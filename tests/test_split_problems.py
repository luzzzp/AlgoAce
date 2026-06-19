from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoace.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite


SCRIPT = Path("scripts/split_problems.py").resolve()
SPEC = importlib.util.spec_from_file_location("split_problems", SCRIPT)
assert SPEC and SPEC.loader
split_problems = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split_problems)


class SplitProblemsTest(unittest.TestCase):
    def test_split_is_deterministic_and_non_overlapping(self) -> None:
        ids = [f"p{i:03d}" for i in range(100)]
        import random

        first = ids.copy()
        second = ids.copy()
        random.Random(42).shuffle(first)
        random.Random(42).shuffle(second)
        split_a = split_problems.split_problem_ids(first, 0.8, 0.1)
        split_b = split_problems.split_problem_ids(second, 0.8, 0.1)

        self.assertEqual(split_a, split_b)
        self.assertEqual({name: len(values) for name, values in split_a.items()}, {"train": 80, "dev": 10, "test": 10})
        all_ids = [item for values in split_a.values() for item in values]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_fingerprint_ignores_input_order(self) -> None:
        self.assertEqual(
            split_problems.fingerprint_problem_ids(["b", "a"]),
            split_problems.fingerprint_problem_ids(["a", "b"]),
        )

    def test_requires_all_test_groups_for_formal_split(self) -> None:
        complete = ProblemBundle(
            ProblemSpec(id="p", statement="Solve."),
            TestSuite(
                visible_tests=[TestCase("1", "1")],
                reward_tests=[TestCase("2", "2")],
                eval_tests=[TestCase("3", "3")],
            ),
        )
        incomplete = ProblemBundle(
            ProblemSpec(id="q", statement="Solve."),
            TestSuite(visible_tests=[TestCase("1", "1")]),
        )

        self.assertTrue(split_problems._has_all_test_groups(complete))
        self.assertFalse(split_problems._has_all_test_groups(incomplete))


if __name__ == "__main__":
    unittest.main()
