from __future__ import annotations

import re


def normalize_answer(answer: str) -> str:
    text = str(answer).replace("_", " ").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hits_at_k(predictions: list[list[str]], gold_answers: list[list[str]], k: int = 1) -> float:
    if len(predictions) != len(gold_answers):
        raise ValueError("predictions and gold_answers must have the same length")
    if not predictions:
        return 0.0
    hits = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_set = {normalize_answer(answer) for answer in pred[:k]}
        gold_set = {normalize_answer(answer) for answer in gold}
        if pred_set & gold_set:
            hits += 1
    return hits / len(predictions)
