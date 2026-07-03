from __future__ import annotations

from egc_tecqa.chain.connectivity import chain_is_connected
from egc_tecqa.chain.model import EvidenceChain
from egc_tecqa.eval.metrics import normalize_answer


def verify_chain(chain: EvidenceChain, gold_answers: list[str] | None = None) -> EvidenceChain:
    fact_entities = {entity for fact in chain.facts for entity in fact.entities}
    fact_times = {
        fact.representative_time.isoformat()
        for fact in chain.facts
        if fact.representative_time is not None
    }
    support_pool = fact_entities | fact_times
    predicted = {normalize_answer(answer) for answer in chain.execution_result}
    gold = {normalize_answer(answer) for answer in gold_answers or []}
    support_pool_norm = {normalize_answer(answer) for answer in support_pool}
    chain.checks = {
        "anchor_found": bool(chain.anchor_facts),
        "graph_connected": chain_is_connected(chain.facts, set(chain.bridge_entities)),
        "answer_supported": bool(predicted) and predicted.issubset(support_pool_norm),
        "gold_hit": bool(predicted & gold) if gold else False,
    }
    return chain
