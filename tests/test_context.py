from __future__ import annotations

import unittest

from algoace.context import ContextBuilder, FailureAnalyzer
from algoace.schema import ExecutionReport, ExecutionRun, ProblemSpec, TestCase


class ContextTest(unittest.TestCase):
    def test_visible_repair_contains_failed_case(self) -> None:
        report = ExecutionReport(
            True,
            runs=[
                ExecutionRun(
                    "visible-0001",
                    "visible",
                    False,
                    "2 3\n",
                    "5",
                    "-1",
                )
            ],
        )
        failure = FailureAnalyzer().analyze(report)
        packet = ContextBuilder().visible_repair(
            ProblemSpec(id="sum", statement="Sum two integers."),
            [TestCase("2 3\n", "5\n", "visible-0001")],
            "print(-1)",
            failure,
            1,
        )

        self.assertEqual(packet.context_type, "VISIBLE_REPAIR")
        self.assertIn("2 3", packet.prompt)
        self.assertIn("Expected:\n5", packet.prompt)

    def test_hidden_review_does_not_contain_hidden_case(self) -> None:
        packet = ContextBuilder().hidden_review(
            ProblemSpec(id="sum", statement="Sum two integers."),
            [TestCase("2 3\n", "5\n", "visible-0001")],
            "print(5)",
            1,
        )

        self.assertEqual(packet.context_type, "HIDDEN_REVIEW")
        self.assertIn("Hidden test inputs and outputs are withheld", packet.prompt)
        self.assertNotIn("eval-0001", packet.prompt)


if __name__ == "__main__":
    unittest.main()

