from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoace.hf_model import SYSTEM_PROMPT
from algoace.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite


SCRIPT = Path("scripts/make_grpo_prompts.py").resolve()
SPEC = importlib.util.spec_from_file_location("make_grpo_prompts", SCRIPT)
assert SPEC and SPEC.loader
make_grpo_prompts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_grpo_prompts)


class MakeGrpoPromptsTest(unittest.TestCase):
    def test_system_prompt_is_code_only(self) -> None:
        self.assertIn("exactly one complete Python 3 code block", SYSTEM_PROMPT)

    def test_record_uses_conversational_prompt_matching_inference(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="sum", statement="Add two integers."),
            TestSuite(visible_tests=[TestCase("2 3", "5", "visible-0001")]),
        )

        record = make_grpo_prompts._record(bundle)

        self.assertEqual(record["problem_id"], "sum")
        self.assertEqual(record["prompt"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(record["prompt"][1]["role"], "user")
        self.assertIn("Visible samples", record["prompt"][1]["content"])


if __name__ == "__main__":
    unittest.main()
