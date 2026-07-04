from __future__ import annotations

import re
from dataclasses import dataclass

from egc_tecqa.data.multitq import label_to_text, normalize_text
from egc_tecqa.parser.intent import ParsedQuestion
from egc_tecqa.parser.rule_parser import extract_time_expression, infer_operator

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "did",
    "do",
    "does",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "whom",
    "with",
}

DEMONYMS = {
    "danish": "denmark",
    "kazakh": "kazakhstan",
    "taiwanese": "taiwan",
    "iraqi": "iraq",
    "chinese": "china",
    "ethiopian": "ethiopia",
}


def _tokens(text: str) -> set[str]:
    tokens = []
    for token in normalize_text(text).split():
        if token in STOPWORDS:
            continue
        tokens.append(DEMONYMS.get(token, token))
    return set(tokens)


@dataclass
class HeuristicGrounder:
    entities: list[str]
    relations: list[str]
    max_entities: int = 4

    def __post_init__(self) -> None:
        aliases = []
        for entity in self.entities:
            text = normalize_text(entity)
            if len(text) >= 3:
                aliases.append((text, entity))
        self.entity_aliases = sorted(aliases, key=lambda item: len(item[0]), reverse=True)
        self.entity_tokens = [(entity, _tokens(entity)) for entity in self.entities]
        self.relation_tokens = [(relation, _tokens(relation)) for relation in self.relations]

    def link_entities(self, question: str) -> list[str]:
        qnorm = f" {normalize_text(question)} "
        found: list[str] = []
        occupied: list[tuple[int, int]] = []
        for alias, entity in self.entity_aliases:
            pattern = f" {re.escape(alias)} "
            match = re.search(pattern, qnorm)
            if not match:
                continue
            span = match.span()
            if any(not (span[1] <= old[0] or span[0] >= old[1]) for old in occupied):
                continue
            found.append(entity)
            occupied.append(span)
            if len(found) >= self.max_entities:
                break
        return found

    def ground_entity_mention(self, mention: str) -> str | None:
        mention_norm = normalize_text(mention)
        if not mention_norm:
            return None
        padded = f" {mention_norm} "
        for alias, entity in self.entity_aliases:
            if f" {alias} " in padded or padded.strip() == alias:
                return entity

        mention_tokens = _tokens(mention)
        best_entity = None
        best_score = 0.0
        for entity, ent_tokens in self.entity_tokens:
            if not ent_tokens:
                continue
            overlap = len(mention_tokens & ent_tokens)
            if overlap == 0:
                continue
            score = overlap / ((len(mention_tokens) * len(ent_tokens)) ** 0.5)
            # Prefer exact country/role disambiguation when present.
            if mention_tokens <= ent_tokens:
                score += 0.15
            if score > best_score:
                best_entity = entity
                best_score = score
        return best_entity

    def link_relation(self, question: str) -> list[str]:
        rule_relation = self._rule_relation(question)
        if rule_relation:
            return [rule_relation]

        q_tokens = _tokens(question)
        best_relation = None
        best_score = -1.0
        for relation, rel_tokens in self.relation_tokens:
            if not rel_tokens:
                continue
            overlap = len(q_tokens & rel_tokens)
            if overlap == 0:
                continue
            score = overlap / (len(rel_tokens) ** 0.5)
            if score > best_score:
                best_relation = relation
                best_score = score
        return [best_relation] if best_relation else []

    def _rule_relation(self, question: str) -> str | None:
        q = normalize_text(question)
        rules = [
            (("sign", "agreement"), "Sign_formal_agreement"),
            (("signed", "agreement"), "Sign_formal_agreement"),
            (("visit",), "Make_a_visit"),
            (("visited",), "Make_a_visit"),
            (("request",), "Make_an_appeal_or_request"),
            (("appeal",), "Make_an_appeal_or_request"),
            (("negotiate",), "Express_intent_to_meet_or_negotiate"),
            (("negotiation",), "Express_intent_to_meet_or_negotiate"),
            (("meet",), "Express_intent_to_meet_or_negotiate"),
            (("cooperation",), "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)"),
            (("cooperate",), "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)"),
            (("threaten",), "Threaten"),
            (("conventional", "military"), "Use_conventional_military_force"),
            (("small", "arms"), "fight_with_small_arms_and_light_weapons"),
            (("statement",), "Make_statement"),
        ]
        relation_set = set(self.relations)
        for needles, relation in rules:
            if relation in relation_set and all(needle in q for needle in needles):
                return relation
        return None

    def parse(
        self,
        question: str,
        qtype: str = "",
        answer_type: str = "entity",
        time_level: str | None = None,
    ) -> ParsedQuestion:
        entities = self.link_entities(question)
        relations = self.link_relation(question)
        main_candidates = choose_main_entity(question, entities)
        return ParsedQuestion(
            question=question,
            entities=entities,
            relations=relations,
            main_entity_candidates=main_candidates,
            answer_type=answer_type,
            temporal_operator=infer_operator(question, qtype),
            anchor_expression=extract_time_expression(question),
            target_slot=infer_target_slot(question, answer_type),
            metadata={"qtype": qtype, "time_level": time_level, "entity_text": [label_to_text(e) for e in entities]},
        )

    def ground_parsed_question(self, parsed: ParsedQuestion) -> ParsedQuestion:
        grounded_entities: list[str] = []
        for mention in parsed.entities:
            grounded = self.ground_entity_mention(mention)
            if grounded and grounded not in grounded_entities:
                grounded_entities.append(grounded)

        grounded_main: list[str] = []
        for mention in parsed.main_entity_candidates:
            grounded = self.ground_entity_mention(mention)
            if grounded and grounded not in grounded_main:
                grounded_main.append(grounded)
        if not grounded_main and grounded_entities:
            grounded_main = [grounded_entities[-1]]

        relation_query = " ".join(parsed.relations) or parsed.question
        grounded_relations = self.link_relation(relation_query)

        anchor_entity = parsed.metadata.get("anchor_entity")
        grounded_anchor = self.ground_entity_mention(anchor_entity) if anchor_entity else None
        metadata = dict(parsed.metadata)
        metadata["llm_entities"] = parsed.entities
        metadata["llm_relations"] = parsed.relations
        metadata["grounded_anchor_entity"] = grounded_anchor
        anchor_expression = parsed.anchor_expression or extract_time_expression(parsed.question)
        if anchor_expression is None and anchor_entity:
            anchor_expression = extract_time_expression(str(anchor_entity))

        return ParsedQuestion(
            question=parsed.question,
            entities=grounded_entities,
            relations=grounded_relations,
            main_entity_candidates=grounded_main,
            answer_type=parsed.answer_type,
            temporal_operator=parsed.temporal_operator,
            anchor_expression=anchor_expression,
            target_slot=parsed.target_slot,
            metadata=metadata,
        )


def choose_main_entity(question: str, entities: list[str]) -> list[str]:
    if not entities:
        return []
    q = question.lower()
    if q.startswith(("who ", "what ", "which ", "when ", "in which ")):
        # In questions like "who visited Iraq", the object/focus entity is
        # usually the pivot connected to the answer. Use the last mention.
        return [entities[-1]]
    return [entities[0]]


def infer_target_slot(question: str, answer_type: str) -> str:
    if answer_type == "time":
        return "time"
    q = question.lower()
    if q.startswith(("who ", "which country", "which person", "what country")):
        return "subject"
    if "with whom" in q or "to whom" in q:
        return "object"
    return "subject"
