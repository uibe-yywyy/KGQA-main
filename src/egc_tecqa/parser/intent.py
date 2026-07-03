from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedQuestion:
    question: str
    entities: list[str]
    relations: list[str]
    main_entity_candidates: list[str]
    answer_type: str = "entity"
    temporal_operator: str = "equal"
    anchor_expression: str | None = None
    target_slot: str = "subject"
    metadata: dict = field(default_factory=dict)

