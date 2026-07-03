from __future__ import annotations

from dataclasses import dataclass, field

from egc_tecqa.kg.fact import Fact


@dataclass
class EvidenceChain:
    chain_id: str
    facts: list[Fact]
    roles: list[str]
    operator: str
    anchor_facts: list[Fact] = field(default_factory=list)
    bridge_entities: list[str] = field(default_factory=list)
    answer_slot: str = "subject"
    candidate_answers: list[str] = field(default_factory=list)
    execution_result: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def as_debug_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "operator": self.operator,
            "facts": [fact.as_tuple() for fact in self.facts],
            "roles": self.roles,
            "bridge_entities": self.bridge_entities,
            "candidate_answers": self.candidate_answers,
            "execution_result": self.execution_result,
            "checks": self.checks,
        }

