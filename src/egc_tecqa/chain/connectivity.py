from __future__ import annotations

from egc_tecqa.kg.fact import Fact


def connected(left: Fact, right: Fact, pivot_entities: set[str] | None = None) -> bool:
    if left.shares_entity_with(right):
        return True
    if pivot_entities and (left.entities & pivot_entities) and (right.entities & pivot_entities):
        return True
    return False


def chain_is_connected(facts: list[Fact], pivot_entities: set[str] | None = None) -> bool:
    if len(facts) <= 1:
        return True
    return all(connected(a, b, pivot_entities) for a, b in zip(facts, facts[1:]))


def bridge_entities(facts: list[Fact]) -> list[str]:
    if len(facts) < 2:
        return []
    bridges: set[str] = set()
    for left, right in zip(facts, facts[1:]):
        bridges.update(left.entities & right.entities)
    return sorted(bridges)

