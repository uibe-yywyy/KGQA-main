from __future__ import annotations

import re

from .intent import ParsedQuestion

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def infer_operator(question: str, question_type: str | None = None) -> str:
    if question_type:
        qt = question_type.lower().replace("/", "_").replace(" ", "_")
        if "after_first" in qt:
            return "after_first"
        if "before_last" in qt:
            return "before_last"
        if "before_after" in qt:
            q = question.lower()
            if "before" in q:
                return "before"
            if "after" in q:
                return "after"
        if qt == "after" or qt.endswith("_after"):
            return "after"
        if qt == "before" or qt.endswith("_before"):
            return "before"
        if "first" in qt and "last" not in qt:
            return "first"
        if "last" in qt:
            return "last"
        if "before" in qt or "after" in qt:
            return "before_after"
        if "time_join" in qt:
            return "time_join"
        if "equal_multi" in qt:
            return "equal_multi"
        if "equal" in qt:
            return "equal"

    q = question.lower()
    if "after" in q and "first" in q:
        return "after_first"
    if "before" in q and "last" in q:
        return "before_last"
    if "first" in q:
        return "first"
    if "last" in q or "latest" in q:
        return "last"
    if "after" in q:
        return "after"
    if "before" in q:
        return "before"
    if "during" in q:
        return "during"
    return "equal"


def extract_time_expression(question: str) -> str | None:
    month_match = re.search(
        r"\b("
        + "|".join(MONTHS)
        + r")\s+(\d{4})\b",
        question,
        flags=re.IGNORECASE,
    )
    if month_match:
        month, year = month_match.groups()
        return f"{year}-{MONTHS[month.lower()]}"

    match = re.search(r"\b(\d{4}(?:-\d{2}(?:-\d{2})?)?)\b", question)
    if match:
        return match.group(1)
    return None


def make_parsed_question(
    question: str,
    entities: list[str],
    relations: list[str],
    main_entity_candidates: list[str] | None = None,
    answer_type: str = "entity",
    question_type: str | None = None,
    target_slot: str = "subject",
) -> ParsedQuestion:
    return ParsedQuestion(
        question=question,
        entities=entities,
        relations=relations,
        main_entity_candidates=main_entity_candidates or entities[:1],
        answer_type=answer_type,
        temporal_operator=infer_operator(question, question_type),
        anchor_expression=extract_time_expression(question),
        target_slot=target_slot,
        metadata={"question_type": question_type},
    )
