from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from algoace.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
    save_problem,
)


SCRIPT = Path("scripts/audit_training_data.py").resolve()
SPEC = importlib.util.spec_from_file_location("audit_training_data", SCRIPT)
assert SPEC and SPEC.loader
audit_training_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_training_data)


class AuditTrainingDataTest(unittest.TestCase):
    def test_clean_dataset_has_no_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problems = root / "train"
            problems.mkdir()
            bundle = ProblemBundle(
                ProblemSpec(id="sum", statement="Add two numbers."),
                TestSuite(visible_tests=[TestCase("1 2", "3")]),
                OracleMetadata(
                    [OracleSolution("python3", "print(sum(map(int, input().split())))", True)]
                ),
            )
            save_problem(bundle, problems / "sum.json")
            (problems / "_split_metadata.json").write_text(
                json.dumps({"role": "train"}), encoding="utf-8"
            )
            dataset = root / "sft.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "instruction": "Solve.",
                        "input": "Add two numbers.",
                        "output": "```python\nprint(sum(map(int, input().split())))\n```",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = audit_training_data.audit(problems, dataset)

            self.assertEqual(report["failures"], [])
            self.assertEqual(report["problems"]["split_role"], "train")
            self.assertEqual(report["dataset"]["syntax_valid_records"], 1)

    def test_invalid_code_is_reported(self) -> None:
        self.assertFalse(audit_training_data._syntax_valid("def broken("))

    def test_syntax_warning_is_counted_separately_from_invalid_syntax(self) -> None:
        code = "if 1 is 1:\n    pass"

        self.assertTrue(audit_training_data._syntax_valid(code))
        self.assertTrue(audit_training_data._has_syntax_warning(code))


if __name__ == "__main__":
    unittest.main()
