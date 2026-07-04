from __future__ import annotations

from datetime import date

from egc_tecqa.kg.fact import Fact, parse_date
from egc_tecqa.parser.intent import ParsedQuestion

from .connectivity import bridge_entities, chain_is_connected
from .model import EvidenceChain


def _fact_time_key(fact: Fact) -> int:
    t = fact.representative_time
    return t.toordinal() if t else 0


def _time_matches(target: date | None, fact: Fact, granularity: str | None) -> bool:
    fact_time = fact.representative_time
    if target is None or fact_time is None:
        return False
    if granularity == "year":
        return fact_time.year == target.year
    if granularity == "month":
        return fact_time.year == target.year and fact_time.month == target.month
    return fact_time == target


def _ordered_for_operator(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    reverse = parsed.temporal_operator in {"last", "before_last"}
    return sorted(facts, key=_fact_time_key, reverse=reverse)


def _relation_matches(fact: Fact, relations: list[str]) -> bool:
    if not relations:
        return True
    rel = fact.relation.lower()
    return any(r.lower() == rel for r in relations)


def _entity_coverage(fact: Fact, entities: set[str]) -> int:
    return len(fact.entities & entities)


def _choose_anchor_fact(parsed: ParsedQuestion, facts: list[Fact]) -> Fact:
    """Choose an anchor fact for relative temporal operators.

    For questions such as "after John Baird", the main entity is often the
    timeline pivot (Thailand) while another mentioned entity supplies the
    temporal anchor. Prefer facts touching non-main entities.
    """

    main_entities = set(parsed.main_entity_candidates)
    grounded_anchor = parsed.metadata.get("grounded_anchor_entity")
    anchor_entities = []
    if grounded_anchor:
        anchor_entities.append(grounded_anchor)
    anchor_entities.extend(entity for entity in parsed.entities if entity not in main_entities)

    for entity in anchor_entities:
        entity_facts = [fact for fact in facts if entity in fact.entities]
        if entity_facts:
            ranked = sorted(
                entity_facts,
                key=lambda fact: (-_entity_coverage(fact, set(parsed.entities)), _fact_time_key(fact)),
            )
            return ranked[0]

    ranked = sorted(
        facts,
        key=lambda fact: (-_entity_coverage(fact, set(parsed.entities)), _fact_time_key(fact)),
    )
    return ranked[0]


def _facts_around_anchor(parsed: ParsedQuestion, anchor_fact: Fact, facts: list[Fact]) -> list[Fact]:
    anchor_time = anchor_fact.representative_time
    others = [fact for fact in facts if fact.fact_id != anchor_fact.fact_id]
    pivot_set = set(parsed.main_entity_candidates)
    if anchor_time is None:
        return _ordered_for_operator(parsed, others)

    op = parsed.temporal_operator
    if op in {"after", "after_first"}:
        candidates = sorted(
            [fact for fact in others if fact.representative_time and fact.representative_time > anchor_time],
            key=_fact_time_key,
        )
        pivot_candidates = [fact for fact in candidates if fact.entities & pivot_set]
        return pivot_candidates or candidates
    if op in {"before", "before_last"}:
        candidates = sorted(
            [fact for fact in others if fact.representative_time and fact.representative_time < anchor_time],
            key=_fact_time_key,
            reverse=True,
        )
        pivot_candidates = [fact for fact in candidates if fact.entities & pivot_set]
        return pivot_candidates or candidates
    return _ordered_for_operator(parsed, others)


def _same_time_key(parsed: ParsedQuestion, fact: Fact) -> tuple[int, int, int]:
    anchor = parsed.metadata.get("grounded_anchor_entity")
    anchor_penalty = 1 if anchor and anchor in fact.entities else 0
    pivot_set = set(parsed.main_entity_candidates)
    pivot_bonus = 1 if fact.entities & pivot_set else 0
    return (anchor_penalty, -pivot_bonus, _fact_time_key(fact))


def _facts_for_equal_multi(parsed: ParsedQuestion, anchor_fact: Fact, facts: list[Fact]) -> list[Fact]:
    others = [fact for fact in facts if fact.fact_id != anchor_fact.fact_id]
    pivot_set = set(parsed.main_entity_candidates)
    pivot_facts = [fact for fact in others if fact.entities & pivot_set]
    candidates = pivot_facts or others
    anchor_time = anchor_fact.representative_time
    granularity = parsed.metadata.get("time_level")
    return sorted(
        candidates,
        key=lambda fact: (
            not _time_matches(anchor_time, fact, granularity),
            *_same_time_key(parsed, fact),
        ),
    )


def _facts_for_explicit_time_relative(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    target_time = parse_date(parsed.anchor_expression)
    op = parsed.temporal_operator
    if op == "before":
        return sorted(
            [fact for fact in facts if fact.representative_time and fact.representative_time < target_time],
            key=_fact_time_key,
            reverse=True,
        )
    if op == "after":
        return sorted(
            [fact for fact in facts if fact.representative_time and fact.representative_time > target_time],
            key=_fact_time_key,
        )
    return facts


def _rank_direct_facts(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    entities = set(parsed.entities + parsed.main_entity_candidates)
    reverse_time = parsed.temporal_operator in {"last", "before_last"}
    target_time = parse_date(parsed.anchor_expression) if parsed.anchor_expression else None
    granularity = parsed.metadata.get("time_level")

    def sort_key(fact: Fact) -> tuple[bool, int, int]:
        time_key = _fact_time_key(fact)
        if reverse_time:
            time_key = -time_key
        return (not _time_matches(target_time, fact, granularity), -_entity_coverage(fact, entities), time_key)

    return sorted(facts, key=sort_key)


def _focus_direct_facts(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    entities = set(parsed.entities + parsed.main_entity_candidates)
    if not entities:
        return facts
    max_coverage = max(_entity_coverage(fact, entities) for fact in facts)
    if max_coverage > 1:
        return [fact for fact in facts if _entity_coverage(fact, entities) == max_coverage]
    return facts


def _is_relative_operator(parsed: ParsedQuestion) -> bool:
    return parsed.temporal_operator in {"after", "after_first", "before", "before_last"}


def _is_connected_or_focus_chain(facts: list[Fact], pivot_set: set[str]) -> bool:
    if chain_is_connected(facts, pivot_set):
        return True
    return all(fact.entities & pivot_set for fact in facts)


def _trim_for_connectivity(facts: list[Fact], pivot_set: set[str], max_facts: int) -> list[Fact]:
    trimmed = facts[:max_facts]
    if _is_connected_or_focus_chain(trimmed, pivot_set):
        return trimmed

    for fact in trimmed:
        connected = [other for other in trimmed if fact.shares_entity_with(other)]
        if _is_connected_or_focus_chain(connected, pivot_set):
            return connected
    return trimmed


def build_simple_connected_chains(
    parsed: ParsedQuestion,
    candidate_facts: list[Fact],
    max_facts: int = 20,
) -> list[EvidenceChain]:
    """Build a first-pass graph-connected evidence chain.

    The chain builder is anchor-aware: relative questions first locate the
    anchor fact in the full candidate set, then collect facts before/after that
    anchor. Direct questions prioritize facts that cover more grounded entities.
    """

    entity_set = set(parsed.entities + parsed.main_entity_candidates)
    relevant = [
        fact
        for fact in candidate_facts
        if _relation_matches(fact, parsed.relations) and (fact.entities & entity_set)
    ]
    if not relevant:
        return []

    pivot_set = set(parsed.main_entity_candidates)
    if _is_relative_operator(parsed) and parsed.anchor_expression and not parsed.metadata.get("grounded_anchor_entity"):
        relevant = _trim_for_connectivity(_facts_for_explicit_time_relative(parsed, relevant), pivot_set, max_facts)
    elif _is_relative_operator(parsed):
        anchor_fact = _choose_anchor_fact(parsed, relevant)
        around_anchor = _facts_around_anchor(parsed, anchor_fact, relevant)
        relevant = [anchor_fact] + _trim_for_connectivity(around_anchor, pivot_set, max_facts - 1)
    elif parsed.temporal_operator == "equal_multi":
        anchor_fact = _choose_anchor_fact(parsed, relevant)
        same_time_candidates = _facts_for_equal_multi(parsed, anchor_fact, relevant)
        relevant = [anchor_fact] + _trim_for_connectivity(same_time_candidates, pivot_set, max_facts - 1)
    else:
        focused = _focus_direct_facts(parsed, relevant)
        relevant = _trim_for_connectivity(_rank_direct_facts(parsed, focused), pivot_set, max_facts)

    if not relevant:
        return []

    anchor_fact = relevant[0]
    ordered = relevant[:max_facts]
    roles = ["anchor_fact"] + ["context_fact"] * (len(ordered) - 1)
    anchor_facts = [anchor_fact] if not (parsed.anchor_expression and _is_relative_operator(parsed)) else []

    return [
        EvidenceChain(
            chain_id="chain-0",
            facts=ordered,
            roles=roles,
            operator=parsed.temporal_operator,
            anchor_facts=anchor_facts,
            bridge_entities=bridge_entities(ordered),
            answer_slot=parsed.target_slot,
        )
    ]
