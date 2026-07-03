from __future__ import annotations

from egc_tecqa.kg.fact import Fact
from egc_tecqa.kg.indexes import TemporalKG
from egc_tecqa.parser.intent import ParsedQuestion


def retrieve_structure_guided(kg: TemporalKG, parsed: ParsedQuestion) -> list[Fact]:
    return kg.retrieve(
        entities=parsed.entities,
        relations=parsed.relations,
        main_entities=parsed.main_entity_candidates,
    )

