import unittest

from egc_tecqa.chain.search import build_simple_connected_chains
from egc_tecqa.executor.execute import execute_chain
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

    def test_grounding_relation_and_demonym_repairs(self):
        grounder = HeuristicGrounder(
            entities=["Criminal_(Africa)", "Criminal_(Somalia)", "China"],
            relations=[
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
        self.assertEqual(
            grounder.link_relation("made Burundi suffer from conventional military forces"),
            ["Use_conventional_military_force"],
        )

    def test_grounding_prefers_specific_role_entity(self):
        grounder = HeuristicGrounder(
            entities=["Kazakhstan", "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"],
            relations=[],
        )

        self.assertEqual(
            grounder.ground_entity_mention("Cabinet Council of Ministers of Kazakhstan"),
            "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)",
        )

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

        self.assertEqual(executed.execution_result[:2], ["Barnaby_Joyce", "Vietnam"])


if __name__ == "__main__":
    unittest.main()
