from __future__ import annotations

from egc_tecqa.chain.model import EvidenceChain
from egc_tecqa.kg.fact import Fact
from egc_tecqa.parser.intent import ParsedQuestion

from .temporal_ops import after, before, equal_time, first, last


def _format_time(value: str, granularity: str | None) -> str:
    if granularity == "year":
        return value[:4]
    if granularity == "month":
        return value[:7]
    return value


def _answer_from_fact(
    fact: Fact,
    slot: str,
    granularity: str | None = None,
    main_entities: set[str] | None = None,
) -> str:
    main_entities = main_entities or set()
    if slot == "object":
        if fact.object in main_entities and fact.subject not in main_entities:
            return fact.subject
        return fact.object
    if slot == "time":
        t = fact.representative_time
        return _format_time(t.isoformat(), granularity) if t else ""
    if fact.subject in main_entities and fact.object not in main_entities:
        return fact.object
    return fact.subject


def _filter_answer_direction(parsed: ParsedQuestion, facts: list[Fact]) -> list[Fact]:
    """Keep facts whose subject/object side is compatible with the answer slot.

    The KG often contains mirrored or semantically nearby facts. For questions
    like "Which country praised Iran?", the answer is the fact subject and the
    main entity should be on the object side. Without this filter, a later fact
    such as Iran -> Praise -> Japan can incorrectly produce Japan as a supported
    but wrong answer.
    """

    main_entities = set(parsed.main_entity_candidates)
    if not main_entities or parsed.target_slot == "time":
        return facts
    if parsed.target_slot == "subject":
        filtered = [fact for fact in facts if fact.subject not in main_entities]
        return filtered or facts
    if parsed.target_slot == "object":
        filtered = [fact for fact in facts if fact.object not in main_entities]
        return filtered or facts
    return facts


def execute_chain(parsed: ParsedQuestion, chain: EvidenceChain) -> EvidenceChain:
    facts = chain.facts
    if not facts:
        chain.execution_result = []
        return chain

    op = parsed.temporal_operator
    selected: list[Fact]

    if op == "after":
        selected = (
            _filter_answer_direction(parsed, facts[:1])
            if not chain.anchor_facts and parsed.anchor_expression
            else first(_filter_answer_direction(parsed, after(chain.anchor_facts[0], facts[1:])))
        )
    elif op == "before":
        selected = (
            _filter_answer_direction(parsed, facts[:1])
            if not chain.anchor_facts and parsed.anchor_expression
            else last(_filter_answer_direction(parsed, before(chain.anchor_facts[0], facts[1:])))
        )
    elif op == "first":
        scope = equal_time(parsed.anchor_expression, facts, parsed.metadata.get("time_level"))
        scope = _filter_answer_direction(parsed, scope)
        selected = first(scope)
    elif op == "last":
        scope = equal_time(parsed.anchor_expression, facts, parsed.metadata.get("time_level"))
        scope = _filter_answer_direction(parsed, scope)
        selected = last(scope)
    elif op == "after_first":
        selected = first(_filter_answer_direction(parsed, after(chain.anchor_facts[0], facts[1:])))
    elif op == "before_last":
        selected = last(_filter_answer_direction(parsed, before(chain.anchor_facts[0], facts[1:])))
    elif op == "equal":
        selected = _filter_answer_direction(
            parsed,
            equal_time(parsed.anchor_expression, facts, parsed.metadata.get("time_level")),
        )
    elif op == "equal_multi":
        anchor_time = chain.anchor_facts[0].representative_time if chain.anchor_facts else None
        selected = _filter_answer_direction(
            parsed,
            equal_time(anchor_time, facts[1:], parsed.metadata.get("time_level")),
        )
    else:
        selected = facts[:1]

    blocked_answers = set(parsed.main_entity_candidates)
    grounded_anchor = parsed.metadata.get("grounded_anchor_entity")
    if grounded_anchor:
        blocked_answers.add(grounded_anchor)
    answers = [
        _answer_from_fact(
            fact,
            parsed.target_slot,
            parsed.metadata.get("time_level"),
            set(parsed.main_entity_candidates),
        )
        for fact in selected
    ]
    deduped = []
    for answer in answers:
        if not answer or answer in blocked_answers or answer in deduped:
            continue
        deduped.append(answer)
    chain.candidate_answers = deduped
    chain.execution_result = deduped
    return chain
