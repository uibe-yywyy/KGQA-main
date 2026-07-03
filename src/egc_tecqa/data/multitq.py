from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from egc_tecqa.kg.fact import Fact


@dataclass(frozen=True)
class QuestionExample:
    quid: int
    question: str
    answers: list[str]
    answer_type: str
    qtype: str
    qlabel: str
    time_level: str | None = None


def load_multitq_questions(root: str | Path, split: str) -> list[QuestionExample]:
    path = Path(root) / "questions" / f"{split}.json"
    data = json.loads(path.read_text())
    return [
        QuestionExample(
            quid=int(row["quid"]),
            question=row["question"],
            answers=[str(answer) for answer in row["answers"]],
            answer_type=row.get("answer_type", "entity"),
            qtype=row.get("qtype", ""),
            qlabel=row.get("qlabel", ""),
            time_level=row.get("time_level"),
        )
        for row in data
    ]


def load_multitq_kg(root: str | Path, split: str = "full", limit: int | None = None) -> list[Fact]:
    path = Path(root) / "kg" / f"{split}.txt"
    facts: list[Fact] = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            subject, relation, object_, timestamp = line.rstrip("\n").split("\t")
            facts.append(Fact.from_values(f"{split}-{idx}", subject, relation, object_, timestamp))
    return facts


def load_schema(root: str | Path) -> tuple[list[str], list[str]]:
    base = Path(root) / "kg"
    entities = list(json.loads((base / "entity2id.json").read_text()).keys())
    relations = list(json.loads((base / "relation2id.json").read_text()).keys())
    return entities, relations


def label_to_text(label: str) -> str:
    text = label.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = label_to_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sample_by_qtype(examples: Iterable[QuestionExample], per_type: int) -> list[QuestionExample]:
    counts: dict[str, int] = {}
    selected: list[QuestionExample] = []
    for example in examples:
        key = example.qtype
        if counts.get(key, 0) >= per_type:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(example)
    return selected

