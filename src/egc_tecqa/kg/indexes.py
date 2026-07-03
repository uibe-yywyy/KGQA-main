from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .fact import Fact


def _norm(value: str) -> str:
    return value.strip().lower()


@dataclass
class TemporalKG:
    facts: list[Fact]

    def __post_init__(self) -> None:
        self.by_subject: dict[str, list[Fact]] = defaultdict(list)
        self.by_object: dict[str, list[Fact]] = defaultdict(list)
        self.by_entity: dict[str, list[Fact]] = defaultdict(list)
        self.by_relation: dict[str, list[Fact]] = defaultdict(list)
        self.by_entity_relation: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for fact in self.facts:
            self.by_subject[_norm(fact.subject)].append(fact)
            self.by_object[_norm(fact.object)].append(fact)
            self.by_entity[_norm(fact.subject)].append(fact)
            self.by_entity[_norm(fact.object)].append(fact)
            self.by_relation[_norm(fact.relation)].append(fact)
            self.by_entity_relation[(_norm(fact.subject), _norm(fact.relation))].append(fact)
            self.by_entity_relation[(_norm(fact.object), _norm(fact.relation))].append(fact)

    def facts_for_entity(self, entity: str) -> list[Fact]:
        return list(self.by_entity.get(_norm(entity), []))

    def facts_for_relation(self, relation: str) -> list[Fact]:
        return list(self.by_relation.get(_norm(relation), []))

    def facts_for_entity_relation(self, entity: str, relation: str) -> list[Fact]:
        return list(self.by_entity_relation.get((_norm(entity), _norm(relation)), []))

    def retrieve(
        self,
        entities: list[str],
        relations: list[str],
        main_entities: list[str] | None = None,
    ) -> list[Fact]:
        """Structure-guided high-recall retrieval.

        Main-entity facts are prioritized, but auxiliary entity facts are kept
        so later chain search can recover bridge paths.
        """

        result: dict[str, Fact] = {}
        relation_set = {_norm(r) for r in relations}
        pivots = main_entities or entities

        for entity in pivots:
            for fact in self.facts_for_entity(entity):
                if not relation_set or _norm(fact.relation) in relation_set:
                    result[fact.fact_id] = fact

        for entity in entities:
            for fact in self.facts_for_entity(entity):
                if not relation_set or _norm(fact.relation) in relation_set:
                    result[fact.fact_id] = fact

        return list(result.values())

