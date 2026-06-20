from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/compare_eval_reports.py").resolve()
SPEC = importlib.util.spec_from_file_location("compare_eval_reports", SCRIPT)
assert SPEC and SPEC.loader
compare_eval_reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_eval_reports)


def _result(problem_id: str, solved: bool, code: str = "print(1)") -> dict:
    return {
        "problem_id": problem_id,
        "status": "VERIFIED_SOLVED" if solved else "FAILED",
        "failure_reason": None if solved else "WRONG_ANSWER",
        "attempt_records": [
            {
                "selected_index": 0,
                "candidates": [
                    {
                        "code": code,
                        "visible_result": {
                            "syntax_valid": True,
                            "runs": [{"passed": solved}],
                        },
                    }
                ],
            }
        ],
    }


class CompareEvalReportsTest(unittest.TestCase):
    def test_counts_solution_transitions(self) -> None:
        base = {"metrics": {"verified_success_rate": 0.5}, "results": [_result("a", True), _result("b", False)]}
        candidate = {"metrics": {"verified_success_rate": 0.5}, "results": [_result("a", False), _result("b", True)]}

        report = compare_eval_reports.compare_reports(
            base,
            candidate,
            {"a": "stdin", "b": "callable"},
        )

        self.assertEqual(report["transitions"]["base_only_solved"], 1)
        self.assertEqual(report["transitions"]["candidate_only_solved"], 1)
        self.assertEqual(report["by_io_mode"]["stdin"]["base_verified_success_rate"], 1.0)
        self.assertEqual(len(report["regression_examples"]), 1)


if __name__ == "__main__":
    unittest.main()
