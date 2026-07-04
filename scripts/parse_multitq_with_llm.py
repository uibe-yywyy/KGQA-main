from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from egc_tecqa.data.multitq import load_multitq_questions
from egc_tecqa.parser.llm_client import LLMConfig, OpenAICompatibleClient
from egc_tecqa.parser.llm_parser import LLMQuestionParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/MultiTQ")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out", default="outputs/cases/llm_parse_sample.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_multitq_questions(ROOT / args.data_root, args.split)[: args.limit]
    client = OpenAICompatibleClient(LLMConfig.from_env(ROOT / args.env))
    parser = LLMQuestionParser(client)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for example in examples:
            parsed = parser.parse(example.question, example.qtype, example.answer_type, example.time_level)
            row = {
                "quid": example.quid,
                "question": example.question,
                "gold": example.answers,
                "parsed": parsed.__dict__,
            }
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(json.dumps(row, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

