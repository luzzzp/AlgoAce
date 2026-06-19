from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/convert_taco.py").resolve()
SPEC = importlib.util.spec_from_file_location("convert_taco", SCRIPT)
assert SPEC and SPEC.loader
convert_taco = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_taco)


class ConvertTacoTest(unittest.TestCase):
    def test_converts_human_readable_limits_and_tests(self) -> None:
        row = {
            "name": "Sum",
            "question": "Read two integers and print their sum.",
            "time_limit": "2.0 seconds",
            "memory_limit": "256 MB",
            "input_output": {
                "inputs": ["2 3", "10 20", "7 8"],
                "outputs": ["5", "30", "15"],
            },
            "solutions": ["a,b=map(int,input().split())\nprint(a+b)"],
        }

        bundle = convert_taco.convert_row(row, 0, 1, 1, 1)

        self.assertEqual(bundle.spec.time_limit_sec, 2.0)
        self.assertEqual(bundle.spec.memory_limit_mb, 256)
        self.assertEqual(len(bundle.tests.visible_tests), 1)
        self.assertEqual(len(bundle.tests.reward_tests), 1)
        self.assertEqual(len(bundle.tests.eval_tests), 1)

    def test_detects_callable_mode(self) -> None:
        row = {
            "question": "Return the sum.",
            "input_output": {
                "fn_name": "add",
                "inputs": [[2, 3], [4, 5], [6, 7]],
                "outputs": [5, 9, 13],
            },
            "solutions": ["def add(a,b): return a+b"],
        }

        bundle = convert_taco.convert_row(row, 1, 1, 1, 1)

        self.assertEqual(bundle.spec.io_mode, "callable")
        self.assertEqual(bundle.spec.entry_point, "add")
        self.assertEqual(bundle.tests.visible_tests[0].stdin, "[2, 3]")


if __name__ == "__main__":
    unittest.main()

