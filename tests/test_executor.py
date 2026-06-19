from __future__ import annotations

import unittest

from algoace.executor import PythonExecutor
from algoace.schema import TestCase


class ExecutorTest(unittest.TestCase):
    def test_accepts_correct_stdin_program(self) -> None:
        result = PythonExecutor().evaluate(
            "a,b=map(int,input().split())\nprint(a+b)",
            [TestCase("2 3\n", "5\n", "visible-0001")],
        )
        self.assertTrue(result.all_passed)

    def test_reports_wrong_answer(self) -> None:
        result = PythonExecutor().evaluate(
            "a,b=map(int,input().split())\nprint(a-b)",
            [TestCase("2 3\n", "5\n", "visible-0001")],
        )
        self.assertFalse(result.all_passed)
        self.assertEqual(result.runs[0].actual, "-1")

    def test_reports_runtime_error(self) -> None:
        result = PythonExecutor().evaluate(
            "raise RuntimeError('boom')",
            [TestCase("", "", "visible-0001")],
        )
        self.assertNotEqual(result.runs[0].returncode, 0)
        self.assertIn("RuntimeError", result.runs[0].stderr)

    def test_kills_timeout(self) -> None:
        result = PythonExecutor().evaluate(
            "while True: pass",
            [TestCase("", "", "visible-0001", timeout_sec=0.1)],
        )
        self.assertTrue(result.runs[0].timed_out)

    def test_executes_callable_entry_point(self) -> None:
        result = PythonExecutor().evaluate(
            "def add(a, b):\n    return a + b",
            [TestCase("[2, 3]", "5", "visible-0001")],
            entry_point="add",
        )
        self.assertTrue(result.all_passed)

    def test_executes_solution_class_entry_point(self) -> None:
        result = PythonExecutor().evaluate(
            "class Solution:\n    def add(self, a, b):\n        return a + b",
            [TestCase("[2, 3]", "5", "visible-0001")],
            entry_point="add",
        )
        self.assertTrue(result.all_passed)


if __name__ == "__main__":
    unittest.main()
