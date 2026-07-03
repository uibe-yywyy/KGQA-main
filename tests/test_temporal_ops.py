from egc_tecqa.executor.temporal_ops import after, before, first, last
from egc_tecqa.kg.fact import Fact
import unittest


class TemporalOpsTest(unittest.TestCase):
    def test_temporal_ops_ordering(self):
        anchor = Fact.from_values("a", "A", "visit", "X", "2020-01-02")
        facts = [
            Fact.from_values("b", "B", "visit", "X", "2020-01-01"),
            Fact.from_values("c", "C", "visit", "X", "2020-01-03"),
        ]
        self.assertEqual(before(anchor, facts)[0].subject, "B")
        self.assertEqual(after(anchor, facts)[0].subject, "C")
        self.assertEqual(first(facts)[0].subject, "B")
        self.assertEqual(last(facts)[0].subject, "C")


if __name__ == "__main__":
    unittest.main()
