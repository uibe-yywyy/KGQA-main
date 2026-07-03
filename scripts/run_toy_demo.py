from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from egc_tecqa.chain.search import build_simple_connected_chains
from egc_tecqa.eval.metrics import hits_at_k
from egc_tecqa.executor.execute import execute_chain
from egc_tecqa.executor.verifier import verify_chain
from egc_tecqa.kg.fact import Fact
from egc_tecqa.kg.indexes import TemporalKG
from egc_tecqa.parser.rule_parser import make_parsed_question
from egc_tecqa.retrieval.structure_guided import retrieve_structure_guided


def build_toy_kg() -> TemporalKG:
    facts = [
        Fact.from_values("f1", "John Baird", "visit", "Thailand", "2012-03-28"),
        Fact.from_values("f2", "Malaysia", "visit", "Thailand", "2012-04-07"),
        Fact.from_values("f3", "Cambodia", "visit", "Thailand", "2012-10-19"),
        Fact.from_values("f4", "Ethiopia", "visit", "Thailand", "2013-10-15"),
        Fact.from_values("f5", "George Yeo", "visit", "Thailand", "2007-01-23"),
    ]
    return TemporalKG(facts)


def main() -> None:
    kg = build_toy_kg()
    parsed = make_parsed_question(
        question="Which country was the first to visit Thailand after John Baird?",
        entities=["Thailand", "John Baird"],
        relations=["visit"],
        main_entity_candidates=["Thailand"],
        answer_type="entity",
        question_type="after",
        target_slot="subject",
    )
    candidates = retrieve_structure_guided(kg, parsed)
    chains = build_simple_connected_chains(parsed, candidates)
    predictions: list[str] = []
    for chain in chains:
        execute_chain(parsed, chain)
        verify_chain(chain, gold_answers=["Malaysia"])
        predictions.extend(chain.execution_result)
        print(chain.as_debug_dict())

    print({"hits@1": hits_at_k([predictions], [["Malaysia"]], k=1)})


if __name__ == "__main__":
    main()

