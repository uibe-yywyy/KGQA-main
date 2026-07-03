from __future__ import annotations

from datetime import date

from egc_tecqa.kg.fact import Fact, parse_date


def fact_time(fact: Fact) -> date | None:
    return fact.representative_time


def after(anchor: Fact, candidates: list[Fact]) -> list[Fact]:
    anchor_time = fact_time(anchor)
    if anchor_time is None:
        return []
    return [fact for fact in candidates if fact_time(fact) and fact_time(fact) > anchor_time]


def before(anchor: Fact, candidates: list[Fact]) -> list[Fact]:
    anchor_time = fact_time(anchor)
    if anchor_time is None:
        return []
    return [fact for fact in candidates if fact_time(fact) and fact_time(fact) < anchor_time]


def first(candidates: list[Fact]) -> list[Fact]:
    valid = [fact for fact in candidates if fact_time(fact)]
    return [min(valid, key=lambda f: fact_time(f))] if valid else []


def last(candidates: list[Fact]) -> list[Fact]:
    valid = [fact for fact in candidates if fact_time(fact)]
    return [max(valid, key=lambda f: fact_time(f))] if valid else []


def equal_time(target: str | date | None, candidates: list[Fact]) -> list[Fact]:
    if target is None:
        return candidates
    target_date = parse_date(target)
    return [fact for fact in candidates if fact_time(fact) == target_date]

