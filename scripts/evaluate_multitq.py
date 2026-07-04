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
    parser.add_argument("--badcases", default=None)
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


def reground_cached_llm_parse(parsed: ParsedQuestion, grounder: HeuristicGrounder) -> ParsedQuestion:
    metadata = dict(parsed.metadata)
    llm_raw = metadata.get("llm_raw")
    if not llm_raw:
        return parsed

    raw_parsed = ParsedQuestion(
        question=parsed.question,
        entities=[str(x) for x in metadata.get("llm_entities", parsed.entities)],
        relations=[str(x) for x in metadata.get("llm_relations", parsed.relations)],
        main_entity_candidates=[str(llm_raw["main_entity"])] if llm_raw.get("main_entity") else [],
        answer_type=parsed.answer_type,
        temporal_operator=parsed.temporal_operator,
        anchor_expression=None,
        target_slot=parsed.target_slot,
        metadata={
            "qtype": metadata.get("qtype"),
            "time_level": metadata.get("time_level"),
            "anchor_entity": metadata.get("anchor_entity"),
            "llm_raw": llm_raw,
        },
    )
    return grounder.ground_parsed_question(raw_parsed)


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
        if args.parser == "llm":
            return reground_cached_llm_parse(cache[cache_key], grounder)
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


def diagnosis_for_error(row: dict) -> tuple[str, str]:
    error_type = row["error_type"]
    parsed = row.get("parsed", {})
    metadata = parsed.get("metadata", {})
    anchor_entity = metadata.get("anchor_entity")
    grounded_anchor = metadata.get("grounded_anchor_entity")

    if error_type == "OK":
        return "Prediction matches a gold answer.", "No fix needed."
    if error_type == "PARSE_OR_GROUNDING_ERROR":
        return (
            "Parsed entities or relations are empty or visibly incomplete.",
            "Improve LLM parsing, schema grounding, or fallback relation/entity linking.",
        )
    if error_type == "RETRIEVAL_MISS":
        return (
            "No candidate facts were retrieved after parsing and grounding.",
            "Improve high-recall retrieval using top-m entities/relations and operator-aware expansion.",
        )
    if error_type == "NO_CHAIN":
        return (
            "Candidate facts exist, but chain construction returned no executable chain.",
            "Relax chain construction, add beam search, or inspect connectivity constraints.",
        )
    if error_type == "ANCHOR_ERROR":
        return (
            "The chain lacks a valid temporal anchor for an anchor-relative operator.",
            "Improve anchor mention grounding and anchor fact scoring.",
        )
    if error_type == "EXECUTION_EMPTY":
        return (
            "A chain was built, but the temporal executor returned no answer.",
            "Inspect temporal filter, selector, answer slot, and max-facts truncation.",
        )
    if error_type == "ANSWER_UNSUPPORTED":
        return (
            "A prediction was produced but is not supported by the selected chain facts.",
            "Tighten answer projection and support verification.",
        )
    if anchor_entity and grounded_anchor and str(anchor_entity) != str(grounded_anchor):
        return (
            f"Prediction is supported but wrong; anchor mention {anchor_entity!r} was grounded as {grounded_anchor!r}.",
            "Prioritize fine-grained anchor/entity grounding and exact mention-to-KG matching.",
        )
    return (
        "Prediction is supported by retrieved facts but differs from the gold answer.",
        "Inspect anchor selection, candidate scope, temporal selector, and gold answer coverage.",
    )


def compact_chain_trace(chain_debug: dict) -> dict:
    facts = chain_debug.get("facts", [])
    roles = chain_debug.get("roles", [])
    answer_facts = []
    for role, fact in zip(roles, facts):
        if role == "anchor_fact" or len(answer_facts) < 3:
            answer_facts.append({"role": role, "fact": fact})
    return {
        "operator": chain_debug.get("operator"),
        "anchor_facts": chain_debug.get("facts", [])[:1],
        "selected_or_head_facts": answer_facts,
        "bridge_entities": chain_debug.get("bridge_entities", []),
        "checks": chain_debug.get("checks", {}),
        "execution_result": chain_debug.get("execution_result", []),
    }


def make_badcase(row: dict) -> dict:
    diagnosis, fix_hint = diagnosis_for_error(row)
    return {
        "quid": row["quid"],
        "qtype": row["qtype"],
        "question": row["question"],
        "gold": row["gold"],
        "pred": row["pred"],
        "error_type": row["error_type"],
        "candidate_count": row["candidate_count"],
        "parsed": row["parsed"],
        "trace": compact_chain_trace(row.get("chain", {})),
        "diagnosis": diagnosis,
        "fix_hint": fix_hint,
    }


def default_badcase_path(out_path: Path) -> Path:
    name = out_path.stem
    if name.endswith("_predictions"):
        name = name[: -len("_predictions")]
    return ROOT / "outputs" / "badcases" / f"{name}_badcases.jsonl"


def write_badcases(rows: list[dict], path: Path) -> dict:
    badcases = [make_badcase(row) for row in rows if row["error_type"] != "OK"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in badcases:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    by_error_dir = path.parent / "by_error_type"
    by_error_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in badcases:
        grouped[row["error_type"]].append(row)
    for error_type, group in sorted(grouped.items()):
        error_path = by_error_dir / f"{error_type}.jsonl"
        with error_path.open("w") as handle:
            for row in group:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    markdown_path = path.with_suffix(".md")
    write_badcase_markdown(badcases, markdown_path)
    return {
        "badcase_count": len(badcases),
        "badcase_path": str(path),
        "badcase_markdown": str(markdown_path),
        "badcase_by_error_dir": str(by_error_dir),
    }


def write_badcase_markdown(badcases: list[dict], path: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in badcases:
        counts[row["error_type"]] += 1

    lines = [
        "# MultiTQ Badcase Report",
        "",
        "## Error Counts",
        "",
    ]
    if counts:
        for error_type, count in sorted(counts.items()):
            lines.append(f"- `{error_type}`: {count}")
    else:
        lines.append("- No badcases.")

    lines.extend(["", "## Cases", ""])
    for idx, row in enumerate(badcases[:50], start=1):
        lines.extend(
            [
                f"### {idx}. {row['error_type']} | quid={row['quid']} | qtype={row['qtype']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold: `{row['gold']}`",
                "",
                f"Pred: `{row['pred']}`",
                "",
                f"Candidate count: `{row['candidate_count']}`",
                "",
                f"Diagnosis: {row['diagnosis']}",
                "",
                f"Fix hint: {row['fix_hint']}",
                "",
                "Trace:",
                "",
                "```json",
                json.dumps(row["trace"], ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    badcase_path = ROOT / args.badcases if args.badcases else default_badcase_path(out_path)

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
    summary.update(write_badcases(rows, badcase_path))
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
