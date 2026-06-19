from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algoace.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite


SCRIPT = Path("scripts/evaluate_hf_model.py").resolve()
SPEC = importlib.util.spec_from_file_location("evaluate_hf_model", SCRIPT)
assert SPEC and SPEC.loader
evaluate_hf_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_hf_model)


class EvaluateHfModelTest(unittest.TestCase):
    def test_problem_seed_is_stable_and_problem_specific(self) -> None:
        self.assertEqual(
            evaluate_hf_model.problem_seed(42, "p1"),
            evaluate_hf_model.problem_seed(42, "p1"),
        )
        self.assertNotEqual(
            evaluate_hf_model.problem_seed(42, "p1"),
            evaluate_hf_model.problem_seed(42, "p2"),
        )

    def test_dataset_fingerprint_changes_with_tests(self) -> None:
        first = ProblemBundle(
            ProblemSpec(id="p", statement="Solve."),
            TestSuite(eval_tests=[TestCase("1\n", "1\n", "eval-0001")]),
        )
        second = ProblemBundle(
            ProblemSpec(id="p", statement="Solve."),
            TestSuite(eval_tests=[TestCase("2\n", "2\n", "eval-0001")]),
        )

        self.assertNotEqual(
            evaluate_hf_model.dataset_fingerprint([first]),
            evaluate_hf_model.dataset_fingerprint([second]),
        )

    def test_resume_rejects_changed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.meta.json"
            path.write_text(json.dumps({"model": "base", "seed": 42}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                evaluate_hf_model._validate_resume_identity(
                    path,
                    {"model": "other", "seed": 42},
                )

    def test_loads_split_role_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_split_metadata.json").write_text(
                json.dumps({"role": "test", "seed": 42}),
                encoding="utf-8",
            )

            self.assertEqual(
                evaluate_hf_model._load_split_metadata(root)["role"],
                "test",
            )


if __name__ == "__main__":
    unittest.main()
