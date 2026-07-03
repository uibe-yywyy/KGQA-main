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


def _answer_from_fact(fact: Fact, slot: str, granularity: str | None = None) -> str:
    if slot == "object":
        return fact.object
    if slot == "time":
        t = fact.representative_time
        return _format_time(t.isoformat(), granularity) if t else ""
    return fact.subject


def execute_chain(parsed: ParsedQuestion, chain: EvidenceChain) -> EvidenceChain:
    facts = chain.facts
    if not facts:
        chain.execution_result = []
        return chain

    op = parsed.temporal_operator
    selected: list[Fact]

    if op == "after":
        selected = first(after(chain.anchor_facts[0], facts[1:]))
    elif op == "before":
        selected = last(before(chain.anchor_facts[0], facts[1:]))
    elif op == "first":
        selected = first(facts)
    elif op == "last":
        selected = last(facts)
    elif op == "after_first":
        selected = first(after(chain.anchor_facts[0], facts[1:]))
    elif op == "before_last":
        selected = last(before(chain.anchor_facts[0], facts[1:]))
    elif op == "equal":
        selected = equal_time(parsed.anchor_expression, facts, parsed.metadata.get("time_level"))
    else:
        selected = facts[:1]

    chain.candidate_answers = [
        _answer_from_fact(fact, parsed.target_slot, parsed.metadata.get("time_level"))
        for fact in selected
    ]
    chain.execution_result = [answer for answer in chain.candidate_answers if answer]
    return chain
