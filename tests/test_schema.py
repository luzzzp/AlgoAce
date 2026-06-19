from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from algoace.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite, load_problem, normalize_output, save_problem


class SchemaTest(unittest.TestCase):
    def test_prompt_contains_only_model_visible_problem_data(self) -> None:
        problem = ProblemSpec(
            id="internal-id",
            statement="Read two integers and print their sum.",
            constraints=["1 <= a, b <= 100"],
        )
        prompt = problem.prompt([TestCase("2 3\n", "5\n", "visible-0001")])

        self.assertNotIn("internal-id", prompt)
        self.assertIn("Problem statement:", prompt)
        self.assertIn("Time limit: 2 seconds", prompt)
        self.assertIn("Visible samples:", prompt)

    def test_problem_round_trip(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="sum", statement="Sum."),
            TestSuite(visible_tests=[TestCase("1 2\n", "3\n", "visible-0001")]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sum.json"
            save_problem(bundle, path)
            loaded = load_problem(path)

        self.assertEqual(loaded.spec.id, "sum")
        self.assertEqual(loaded.tests.visible_tests[0].expected_stdout, "3\n")

    def test_normalize_output_ignores_line_edge_whitespace(self) -> None:
        self.assertEqual(normalize_output("        0\n        3\n"), "0\n3")


if __name__ == "__main__":
    unittest.main()
