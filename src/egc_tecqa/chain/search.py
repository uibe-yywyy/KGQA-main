from __future__ import annotations

from egc_tecqa.kg.fact import Fact
from egc_tecqa.parser.intent import ParsedQuestion

from .connectivity import bridge_entities, chain_is_connected
from .model import EvidenceChain


def _fact_time_key(fact: Fact) -> int:
    t = fact.representative_time
    return t.toordinal() if t else 0


def _ordered_for_operator(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    reverse = parsed.temporal_operator in {"last", "before_last"}
    return sorted(facts, key=_fact_time_key, reverse=reverse)


def _relation_matches(fact: Fact, relations: list[str]) -> bool:
    if not relations:
        return True
    rel = fact.relation.lower()
    return any(r.lower() == rel for r in relations)


def _choose_anchor_fact(parsed: ParsedQuestion, facts: list[Fact]) -> Fact:
    """Choose an anchor fact for relative temporal operators.

    For questions such as "after John Baird", the main entity is often the
    timeline pivot (Thailand) while another mentioned entity supplies the
    temporal anchor. Prefer facts touching non-main entities.
    """

    main_entities = set(parsed.main_entity_candidates)
    anchor_entities = [entity for entity in parsed.entities if entity not in main_entities]
    for entity in anchor_entities:
        for fact in facts:
            if entity in fact.entities:
                return fact
    return facts[0]


def build_simple_connected_chains(
    parsed: ParsedQuestion,
    candidate_facts: list[Fact],
    max_facts: int = 20,
) -> list[EvidenceChain]:
    """Build a first-pass graph-connected evidence chain.

    This intentionally starts simple: keep relation-compatible facts that touch
    parsed entities, sort them by time, and verify adjacent connectivity. More
    advanced beam search can replace this without changing downstream APIs.
    """

    entity_set = set(parsed.entities + parsed.main_entity_candidates)
    relevant = [
        fact
        for fact in candidate_facts
        if _relation_matches(fact, parsed.relations) and (fact.entities & entity_set)
    ]
    relevant = _ordered_for_operator(parsed, relevant)
    relevant = relevant[:max_facts]

    if not relevant:
        return []

    pivot_set = set(parsed.main_entity_candidates)
    if not chain_is_connected(relevant, pivot_set):
        # Fall back to facts connected to the first pivot/main fact.
        seed = relevant[0]
        relevant = [fact for fact in relevant if seed.shares_entity_with(fact)]

    anchor_fact = _choose_anchor_fact(parsed, relevant)
    ordered = [anchor_fact] + [fact for fact in relevant if fact.fact_id != anchor_fact.fact_id]
    roles = ["anchor_fact"] + ["context_fact"] * (len(ordered) - 1)

    return [
        EvidenceChain(
            chain_id="chain-0",
            facts=ordered,
            roles=roles,
            operator=parsed.temporal_operator,
            anchor_facts=[anchor_fact],
            bridge_entities=bridge_entities(ordered),
            answer_slot=parsed.target_slot,
        )
    ]
