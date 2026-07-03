from __future__ import annotations


def hits_at_k(predictions: list[list[str]], gold_answers: list[list[str]], k: int = 1) -> float:
    if len(predictions) != len(gold_answers):
        raise ValueError("predictions and gold_answers must have the same length")
    if not predictions:
        return 0.0
    hits = 0
    for pred, gold in zip(predictions, gold_answers):
        if set(pred[:k]) & set(gold):
            hits += 1
    return hits / len(predictions)

