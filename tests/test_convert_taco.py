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

    def test_callable_output_unwraps_singleton_wrapper(self) -> None:
        row = {
            "question": "Create an acronym.",
            "input_output": {
                "fn_name": "make_acronym",
                "inputs": [["My Amazing Story"], ["Portable Network Graphics"], ["Away From Keyboard"]],
                "outputs": [["MAS"], ["PNG"], ["AFK"]],
            },
            "solutions": ["def make_acronym(text): return ''.join(x[0] for x in text.split())"],
        }

        bundle = convert_taco.convert_row(row, 2, 1, 1, 1)

        self.assertEqual(bundle.tests.visible_tests[0].expected_stdout, '"MAS"')

    def test_callable_output_keeps_non_singleton_list(self) -> None:
        self.assertEqual(
            convert_taco._output_text([1, 0], "callable", unwrap_singleton=True),
            "[1, 0]",
        )

    def test_split_reserves_all_groups_for_three_cases(self) -> None:
        cases = [convert_taco.TestCase(str(i), str(i)) for i in range(3)]

        visible, reward, eval_tests = convert_taco._split_cases(cases, 3, 20, 20)

        self.assertEqual((len(visible), len(reward), len(eval_tests)), (1, 1, 1))

    def test_two_cases_reserve_one_reward_test(self) -> None:
        cases = [convert_taco.TestCase(str(i), str(i)) for i in range(2)]

        visible, reward, eval_tests = convert_taco._split_cases(cases, 3, 20, 20)

        self.assertEqual((len(visible), len(reward), len(eval_tests)), (1, 1, 0))

    def test_split_reserves_eval_before_filling_reward(self) -> None:
        cases = [convert_taco.TestCase(str(i), str(i)) for i in range(10)]

        visible, reward, eval_tests = convert_taco._split_cases(cases, 3, 20, 20)

        self.assertEqual((len(visible), len(reward), len(eval_tests)), (3, 5, 2))
        all_inputs = [case.stdin for case in [*visible, *reward, *eval_tests]]
        self.assertEqual(len(all_inputs), len(set(all_inputs)))


if __name__ == "__main__":
    unittest.main()
