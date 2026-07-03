from __future__ import annotations

from egc_tecqa.chain.model import EvidenceChain


def boolean_rate(chains: list[EvidenceChain], check_name: str) -> float:
    if not chains:
        return 0.0
    return sum(1 for chain in chains if chain.checks.get(check_name)) / len(chains)

