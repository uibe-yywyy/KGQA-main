from __future__ import annotations

import json
import re
from dataclasses import dataclass

from egc_tecqa.parser.intent import ParsedQuestion
from egc_tecqa.parser.llm_client import OpenAICompatibleClient
from egc_tecqa.parser.rule_parser import infer_operator


SYSTEM_PROMPT = """You parse temporal knowledge graph QA questions.
Return strict JSON only. Do not answer the question.
Fields:
- entities: surface entity mentions from the question
- relation_phrase: the main action/relation phrase
- main_entity: the pivot entity directly connected to the answer
- anchor_entity: entity/event that provides the temporal reference, or null
- temporal_operator: one of equal,before,after,during,first,last,equal_multi,after_first,before_last,time_join
- answer_type: entity or time
- target_slot: subject, object, or time
"""


USER_TEMPLATE = """Question: {question}
Dataset qtype: {qtype}
Answer type: {answer_type}
Time granularity: {time_level}

Return JSON with the required fields."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class LLMQuestionParser:
    client: OpenAICompatibleClient

    def parse(
        self,
        question: str,
        qtype: str = "",
        answer_type: str = "entity",
        time_level: str | None = None,
    ) -> ParsedQuestion:
        content = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        question=question,
                        qtype=qtype,
                        answer_type=answer_type,
                        time_level=time_level or "",
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        data = _extract_json(content)
        entities = [str(x) for x in data.get("entities", []) if str(x).strip()]
        main_entity = data.get("main_entity") or (entities[-1] if entities else "")
        relation_phrase = data.get("relation_phrase") or ""
        operator = data.get("temporal_operator") or infer_operator(question, qtype)
        target_slot = data.get("target_slot") or ("time" if answer_type == "time" else "subject")
        parsed_answer_type = data.get("answer_type") or answer_type
        return ParsedQuestion(
            question=question,
            entities=entities,
            relations=[relation_phrase] if relation_phrase else [],
            main_entity_candidates=[str(main_entity)] if main_entity else [],
            answer_type=parsed_answer_type,
            temporal_operator=operator,
            anchor_expression=None,
            target_slot=target_slot,
            metadata={
                "qtype": qtype,
                "time_level": time_level,
                "anchor_entity": data.get("anchor_entity"),
                "llm_raw": data,
            },
        )

