from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from egc_tecqa.chain.search import build_simple_connected_chains
from egc_tecqa.data.multitq import (
    load_multitq_kg,
    load_multitq_questions,
    load_schema,
    sample_by_qtype,
)
from egc_tecqa.eval.metrics import hits_at_k
from egc_tecqa.executor.execute import execute_chain
from egc_tecqa.executor.verifier import verify_chain
from egc_tecqa.kg.indexes import TemporalKG
from egc_tecqa.parser.heuristic_grounding import HeuristicGrounder
from egc_tecqa.parser.llm_client import LLMConfig, OpenAICompatibleClient
from egc_tecqa.parser.llm_parser import LLMQuestionParser
from egc_tecqa.retrieval.structure_guided import retrieve_structure_guided


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/MultiTQ")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--kg-split", default="full", choices=["full", "train", "valid", "test"])
    parser.add_argument("--kg-limit", type=int, default=None)
    parser.add_argument("--parser", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out", default="outputs/cases/multitq_debug_predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT / args.data_root
    examples = sample_by_qtype(load_multitq_questions(root, args.split), args.per_type)
    if args.limit is not None:
        examples = examples[: args.limit]
    entities, relations = load_schema(root)
    grounder = HeuristicGrounder(entities, relations)
    llm_parser = None
    if args.parser == "llm":
        llm_parser = LLMQuestionParser(OpenAICompatibleClient(LLMConfig.from_env(ROOT / args.env)))
    kg = TemporalKG(load_multitq_kg(root, args.kg_split, limit=args.kg_limit))

    predictions: list[list[str]] = []
    golds: list[list[str]] = []
    rows: list[dict] = []

    for example in examples:
        if llm_parser is not None:
            parsed = grounder.ground_parsed_question(
                llm_parser.parse(example.question, example.qtype, example.answer_type, example.time_level)
            )
        else:
            parsed = grounder.parse(example.question, example.qtype, example.answer_type, example.time_level)
        candidates = retrieve_structure_guided(kg, parsed)
        chains = build_simple_connected_chains(parsed, candidates, max_facts=80)
        if chains:
            chain = execute_chain(parsed, chains[0])
            verify_chain(chain, example.answers)
            pred = chain.execution_result
            chain_debug = chain.as_debug_dict()
        else:
            pred = []
            chain_debug = {}
        predictions.append(pred)
        golds.append(example.answers)
        rows.append(
            {
                "quid": example.quid,
                "qtype": example.qtype,
                "question": example.question,
                "gold": example.answers,
                "pred": pred,
                "parsed": parsed.__dict__,
                "candidate_count": len(candidates),
                "chain": chain_debug,
            }
        )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(
        json.dumps(
            {
                "split": args.split,
                "parser": args.parser,
                "examples": len(examples),
                "hits@1": hits_at_k(predictions, golds, k=1),
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
