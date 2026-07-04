from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from egc_tecqa.chain.search import build_simple_connected_chains
from egc_tecqa.data.multitq import (
    QuestionExample,
    load_multitq_kg,
    load_multitq_questions,
    load_schema,
    sample_by_qtype,
)
from egc_tecqa.eval.metrics import hits_at_k, normalize_answer
from egc_tecqa.executor.execute import execute_chain
from egc_tecqa.executor.verifier import verify_chain
from egc_tecqa.kg.indexes import TemporalKG
from egc_tecqa.parser.heuristic_grounding import HeuristicGrounder
from egc_tecqa.parser.intent import ParsedQuestion
from egc_tecqa.parser.llm_client import LLMConfig, OpenAICompatibleClient
from egc_tecqa.parser.llm_parser import LLMQuestionParser
from egc_tecqa.retrieval.structure_guided import retrieve_structure_guided


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/MultiTQ")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--kg-split", default="full", choices=["full", "train", "valid", "test"])
    parser.add_argument("--kg-limit", type=int, default=None)
    parser.add_argument("--parser", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-type", type=int, default=None)
    parser.add_argument("--max-facts", type=int, default=80)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--cache", default="outputs/cache/multitq_parse_cache.jsonl")
    parser.add_argument("--out", default="outputs/eval/multitq_eval_predictions.jsonl")
    return parser.parse_args()


def read_parse_cache(path: Path) -> dict[str, ParsedQuestion]:
    if not path.exists():
        return {}
    cache: dict[str, ParsedQuestion] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            cache[row["cache_key"]] = ParsedQuestion(**row["parsed"])
    return cache


def append_parse_cache(path: Path, cache_key: str, parsed: ParsedQuestion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"cache_key": cache_key, "parsed": asdict(parsed)}
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def choose_examples(args: argparse.Namespace, root: Path) -> list[QuestionExample]:
    examples = load_multitq_questions(root, args.split)
    if args.per_type is not None:
        examples = sample_by_qtype(examples, args.per_type)
    if args.limit is not None:
        examples = examples[: args.limit]
    return examples


def parse_example(
    example: QuestionExample,
    args: argparse.Namespace,
    grounder: HeuristicGrounder,
    llm_parser: LLMQuestionParser | None,
    cache: dict[str, ParsedQuestion],
    cache_path: Path,
) -> ParsedQuestion:
    cache_key = f"{args.parser}:{args.split}:{example.quid}"
    if cache_key in cache:
        return cache[cache_key]

    if llm_parser is not None:
        parsed = grounder.ground_parsed_question(
            llm_parser.parse(example.question, example.qtype, example.answer_type, example.time_level)
        )
    else:
        parsed = grounder.parse(example.question, example.qtype, example.answer_type, example.time_level)

    cache[cache_key] = parsed
    append_parse_cache(cache_path, cache_key, parsed)
    return parsed


def has_hit(pred: list[str], gold: list[str], k: int = 1) -> bool:
    pred_set = {normalize_answer(answer) for answer in pred[:k]}
    gold_set = {normalize_answer(answer) for answer in gold}
    return bool(pred_set & gold_set)


def categorize_error(
    parsed: ParsedQuestion,
    candidate_count: int,
    chain_debug: dict,
    pred: list[str],
    gold: list[str],
) -> str:
    if has_hit(pred, gold):
        return "OK"
    if not parsed.entities or not parsed.relations:
        return "PARSE_OR_GROUNDING_ERROR"
    if candidate_count == 0:
        return "RETRIEVAL_MISS"
    if not chain_debug:
        return "NO_CHAIN"
    checks = chain_debug.get("checks", {})
    if not checks.get("anchor_found"):
        return "ANCHOR_ERROR"
    if not pred:
        return "EXECUTION_EMPTY"
    if not checks.get("answer_supported"):
        return "ANSWER_UNSUPPORTED"
    return "WRONG_SUPPORTED_ANSWER"


def summarize(rows: list[dict], predictions: list[list[str]], golds: list[list[str]]) -> dict:
    by_qtype: dict[str, list[int]] = defaultdict(list)
    support_by_qtype: dict[str, list[int]] = defaultdict(list)
    error_counts: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(rows):
        qtype = row["qtype"]
        by_qtype[qtype].append(1 if has_hit(predictions[idx], golds[idx]) else 0)
        checks = row.get("chain", {}).get("checks", {})
        support_by_qtype[qtype].append(1 if checks.get("answer_supported") else 0)
        error_counts[row["error_type"]] += 1

    per_qtype = {}
    for qtype, values in sorted(by_qtype.items()):
        support_values = support_by_qtype[qtype]
        per_qtype[qtype] = {
            "examples": len(values),
            "hits@1": sum(values) / len(values) if values else 0.0,
            "answer_supported_rate": sum(support_values) / len(support_values) if support_values else 0.0,
        }

    return {
        "examples": len(rows),
        "hits@1": hits_at_k(predictions, golds, k=1),
        "per_qtype": per_qtype,
        "error_counts": dict(sorted(error_counts.items())),
    }


def main() -> None:
    args = parse_args()
    root = ROOT / args.data_root
    cache_path = ROOT / args.cache
    out_path = ROOT / args.out

    examples = choose_examples(args, root)
    entities, relations = load_schema(root)
    grounder = HeuristicGrounder(entities, relations)
    llm_parser = None
    if args.parser == "llm":
        llm_parser = LLMQuestionParser(OpenAICompatibleClient(LLMConfig.from_env(ROOT / args.env)))
    parse_cache = read_parse_cache(cache_path)
    kg = TemporalKG(load_multitq_kg(root, args.kg_split, limit=args.kg_limit))

    predictions: list[list[str]] = []
    golds: list[list[str]] = []
    rows: list[dict] = []

    for example in examples:
        parsed = parse_example(example, args, grounder, llm_parser, parse_cache, cache_path)
        candidates = retrieve_structure_guided(kg, parsed)
        chains = build_simple_connected_chains(parsed, candidates, max_facts=args.max_facts)
        if chains:
            chain = execute_chain(parsed, chains[0])
            verify_chain(chain, example.answers)
            pred = chain.execution_result
            chain_debug = chain.as_debug_dict()
        else:
            pred = []
            chain_debug = {}

        row = {
            "quid": example.quid,
            "qtype": example.qtype,
            "question": example.question,
            "gold": example.answers,
            "pred": pred,
            "parsed": asdict(parsed),
            "candidate_count": len(candidates),
            "chain": chain_debug,
        }
        row["error_type"] = categorize_error(parsed, len(candidates), chain_debug, pred, example.answers)

        predictions.append(pred)
        golds.append(example.answers)
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = {
        "split": args.split,
        "parser": args.parser,
        "kg_split": args.kg_split,
        "kg_limit": args.kg_limit,
        "limit": args.limit,
        "per_type": args.per_type,
        "output": str(out_path),
        "cache": str(cache_path),
        **summarize(rows, predictions, golds),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
