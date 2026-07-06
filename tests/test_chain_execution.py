import unittest

from egc_tecqa.chain.search import build_simple_connected_chains
from egc_tecqa.chain.model import EvidenceChain
from egc_tecqa.executor.execute import execute_chain
from egc_tecqa.executor.verifier import verify_chain
from egc_tecqa.kg.fact import Fact
from egc_tecqa.parser.heuristic_grounding import HeuristicGrounder
from egc_tecqa.parser.intent import ParsedQuestion
from egc_tecqa.parser.rule_parser import extract_time_expression


class ChainExecutionTest(unittest.TestCase):
    def test_month_name_time_expression(self):
        self.assertEqual(
            extract_time_expression("Who signed an agreement with China in April 2005?"),
            "2005-04",
        )
        self.assertEqual(
            extract_time_expression("Who visited Malaysia on 14 January 2007?"),
            "2007-01-14",
        )
        self.assertEqual(
            extract_time_expression("Who visited after Jun, 2011?"),
            "2011-06",
        )
        self.assertEqual(
            extract_time_expression("Who visited before 2012-06-8?"),
            "2012-06-08",
        )

    def test_grounding_relation_and_demonym_repairs(self):
        grounder = HeuristicGrounder(
            entities=["Criminal_(Africa)", "Criminal_(Somalia)", "China"],
            relations=[
                "Express_intent_to_cooperate",
                "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)",
                "Make_optimistic_comment",
                "Praise_or_endorse",
                "Criticize_or_denounce",
                "Use_conventional_military_force",
            ],
        )

        self.assertEqual(grounder.ground_entity_mention("Somali criminal"), "Criminal_(Somalia)")
        self.assertEqual(grounder.link_relation("expressed optimism about"), ["Make_optimistic_comment"])
        self.assertEqual(grounder.link_relation("praised"), ["Praise_or_endorse"])
        self.assertEqual(grounder.link_relation("condemned"), ["Criticize_or_denounce"])
        self.assertEqual(grounder.link_relation("express interest in cooperating with"), ["Express_intent_to_cooperate"])
        self.assertEqual(
            grounder.link_relation("wish to establish diplomatic cooperation"),
            ["Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)"],
        )
        self.assertEqual(
            grounder.link_relation("made Burundi suffer from conventional military forces"),
            ["Use_conventional_military_force"],
        )

    def test_grounding_prefers_specific_role_entity(self):
        grounder = HeuristicGrounder(
            entities=[
                "Al-Shabaab",
                "Business_(Belgium)",
                "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)",
                "Citizen_(Belgium)",
                "Citizen_(Norway)",
                "Insurgency_(Al-Shabaab)",
                "Kazakhstan",
            ],
            relations=[],
        )

        self.assertEqual(
            grounder.ground_entity_mention("Cabinet Council of Ministers of Kazakhstan"),
            "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)",
        )
        self.assertEqual(grounder.ground_entity_mention("citizens of Belgium"), "Citizen_(Belgium)")
        self.assertEqual(grounder.ground_entity_mention("citizens of Norway"), "Citizen_(Norway)")
        self.assertEqual(grounder.ground_entity_mention("al-Shabaab insurgency"), "Insurgency_(Al-Shabaab)")

    def test_grounding_composes_group_with_country_context(self):
        grounder = HeuristicGrounder(
            entities=[
                "Actor_(United_Kingdom)",
                "Muslim_(Africa)",
                "Muslim_(United_Kingdom)",
            ],
            relations=["Use_unconventional_violence"],
        )
        parsed = ParsedQuestion(
            question="When did the al-Shabaab insurgency use unconventional violence against Muslims in the United Kingdom?",
            entities=["Muslims", "United Kingdom"],
            relations=["use unconventional violence"],
            main_entity_candidates=[],
            answer_type="time",
            temporal_operator="equal",
            target_slot="time",
        )

        grounded = grounder.ground_parsed_question(parsed)

        self.assertIn("Muslim_(United_Kingdom)", grounded.entities)
        self.assertNotIn("Muslim_(Africa)", grounded.entities)

    def test_grounding_skips_time_mentions(self):
        grounder = HeuristicGrounder(
            entities=["Federica_Mogherini", "Theresa_May", "Wang_Jun"],
            relations=["Express_intent_to_meet_or_negotiate"],
        )
        parsed = ParsedQuestion(
            question="With whom did Federica Mogherini announce her intention to negotiate after 7 May 2015?",
            entities=["Federica Mogherini", "7 May 2015"],
            relations=["announce intention to negotiate"],
            main_entity_candidates=["Federica Mogherini"],
            answer_type="entity",
            temporal_operator="after",
            anchor_expression=None,
            target_slot="object",
            metadata={"anchor_entity": "7 May 2015"},
        )

        grounded = grounder.ground_parsed_question(parsed)

        self.assertEqual(grounded.entities, ["Federica_Mogherini"])
        self.assertEqual(grounded.metadata["grounded_anchor_entity"], None)
        self.assertEqual(grounded.anchor_expression, "2015-05-07")

    def test_relative_anchor_prefers_fact_connected_to_main_entity(self):
        facts = [
            Fact.from_values("1", "Defense_/_Security_Ministry_(Denmark)", "Make_a_visit", "Sudan", "2005-11-28"),
            Fact.from_values("2", "Iran", "Make_a_visit", "Iraq", "2005-11-29"),
            Fact.from_values("3", "Defense_/_Security_Ministry_(Denmark)", "Make_a_visit", "Iraq", "2006-01-05"),
            Fact.from_values("4", "Jack_Straw", "Make_a_visit", "Iraq", "2006-01-06"),
        ]
        parsed = ParsedQuestion(
            question="After the Danish Ministry of Defence and Security, who was the first to visit Iraq?",
            entities=["Defense_/_Security_Ministry_(Denmark)", "Iraq"],
            relations=["Make_a_visit"],
            main_entity_candidates=["Iraq"],
            temporal_operator="after_first",
            target_slot="subject",
            metadata={"grounded_anchor_entity": "Defense_/_Security_Ministry_(Denmark)", "time_level": "day"},
        )

        chain = build_simple_connected_chains(parsed, facts, max_facts=4)[0]
        executed = execute_chain(parsed, chain)

        self.assertEqual(chain.anchor_facts[0].object, "Iraq")
        self.assertEqual(executed.execution_result, ["Jack_Straw"])

    def test_equal_multi_uses_anchor_time_and_excludes_anchor_entity(self):
        facts = [
            Fact.from_values("1", "Oleg_Ostapenko", "Make_a_visit", "China", "2014-09-04"),
            Fact.from_values("2", "Barnaby_Joyce", "Make_a_visit", "China", "2014-09-10"),
            Fact.from_values("3", "China", "Make_a_visit", "Vietnam", "2014-09-11"),
            Fact.from_values("4", "Ma_Ying_Jeou", "Make_a_visit", "China", "2005-01-04"),
        ]
        parsed = ParsedQuestion(
            question="Who visited China in the same month as Oleg Ostapenko?",
            entities=["Oleg_Ostapenko", "China"],
            relations=["Make_a_visit"],
            main_entity_candidates=["China"],
            temporal_operator="equal_multi",
            target_slot="subject",
            metadata={"grounded_anchor_entity": "Oleg_Ostapenko", "time_level": "month"},
        )

        chain = build_simple_connected_chains(parsed, facts, max_facts=4)[0]
        executed = execute_chain(parsed, chain)

        self.assertEqual(executed.execution_result, ["Barnaby_Joyce"])

    def test_explicit_relative_time_respects_granularity(self):
        facts = [
            Fact.from_values("1", "Yang_Hyong_Sop", "Make_a_visit", "Norodom_Sihanouk", "2006-05-10"),
            Fact.from_values("2", "Yang_Hyong_Sop", "Make_a_visit", "China", "2006-07-11"),
            Fact.from_values("3", "Yang_Hyong_Sop", "Make_a_visit", "Angola", "2008-03-21"),
            Fact.from_values("4", "Yang_Hyong_Sop", "Make_a_visit", "Cambodia", "2012-10-20"),
        ]
        parsed = ParsedQuestion(
            question="Which country received the visit from Yang Hyong Sop after 2006?",
            entities=["Yang_Hyong_Sop"],
            relations=["Make_a_visit"],
            main_entity_candidates=["Yang_Hyong_Sop"],
            temporal_operator="after",
            anchor_expression="2006",
            target_slot="object",
            metadata={"time_level": "year"},
        )

        chain = build_simple_connected_chains(parsed, facts, max_facts=4)[0]
        executed = execute_chain(parsed, chain)

        self.assertEqual(executed.execution_result, ["Angola"])

    def test_first_with_explicit_year_filters_scope_before_selecting(self):
        facts = [
            Fact.from_values("1", "South_Africa", "Make_a_visit", "Angola", "2010-05-01"),
            Fact.from_values("2", "Foreign_Affairs_(Namibia)", "Make_a_visit", "Angola", "2011-01-03"),
            Fact.from_values("3", "Yang_Hyong_Sop", "Make_a_visit", "Angola", "2011-07-12"),
        ]
        parsed = ParsedQuestion(
            question="Who was the first to visit Angola in 2011?",
            entities=["Angola"],
            relations=["Make_a_visit"],
            main_entity_candidates=["Angola"],
            temporal_operator="first",
            anchor_expression="2011",
            target_slot="subject",
            metadata={"time_level": "year"},
        )
        chain = EvidenceChain(chain_id="c", facts=facts, roles=["context_fact"] * len(facts), operator="first")

        executed = execute_chain(parsed, chain)

        self.assertEqual(executed.execution_result, ["Foreign_Affairs_(Namibia)"])

    def test_subject_answer_direction_excludes_main_as_subject(self):
        facts = [
            Fact.from_values("1", "Iran", "Praise_or_endorse", "Japan", "2009-12-29"),
            Fact.from_values("2", "Vietnam", "Praise_or_endorse", "Iran", "2009-12-24"),
        ]
        parsed = ParsedQuestion(
            question="Which country last praised Iran in 2009?",
            entities=["Iran"],
            relations=["Praise_or_endorse"],
            main_entity_candidates=["Iran"],
            temporal_operator="last",
            anchor_expression="2009",
            target_slot="subject",
            metadata={"time_level": "year"},
        )
        chain = EvidenceChain(chain_id="c", facts=facts, roles=["context_fact"] * len(facts), operator="last")

        executed = execute_chain(parsed, chain)

        self.assertEqual(executed.execution_result, ["Vietnam"])

    def test_object_answer_direction_excludes_main_as_object(self):
        facts = [
            Fact.from_values("1", "Japan", "Express_intent_to_meet_or_negotiate", "France", "2005-04-01"),
            Fact.from_values("2", "France", "Express_intent_to_meet_or_negotiate", "United_Arab_Emirates", "2013-02-01"),
        ]
        parsed = ParsedQuestion(
            question="With which country did France negotiate for the first time in 2013?",
            entities=["France"],
            relations=["Express_intent_to_meet_or_negotiate"],
            main_entity_candidates=["France"],
            temporal_operator="first",
            anchor_expression="2013",
            target_slot="object",
            metadata={"time_level": "year"},
        )
        chain = EvidenceChain(chain_id="c", facts=facts, roles=["context_fact"] * len(facts), operator="first")

        executed = execute_chain(parsed, chain)

        self.assertEqual(executed.execution_result, ["United_Arab_Emirates"])

    def test_verifier_supports_year_and_month_time_answers(self):
        fact = Fact.from_values("1", "Citizen_(Belgium)", "Sign_formal_agreement", "China", "2005-06-10")
        chain = EvidenceChain(
            chain_id="c",
            facts=[fact],
            roles=["answer_fact"],
            operator="equal",
            execution_result=["2005-06", "2005"],
        )

        verified = verify_chain(chain, ["2005-06"])

        self.assertTrue(verified.checks["answer_supported"])
        self.assertTrue(verified.checks["gold_hit"])


if __name__ == "__main__":
    unittest.main()
