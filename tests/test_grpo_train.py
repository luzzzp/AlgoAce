from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("training/grpo_train.py").resolve()
SPEC = importlib.util.spec_from_file_location("grpo_train", SCRIPT)
assert SPEC and SPEC.loader
grpo_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grpo_train)


class ExampleConfig:
    def __init__(self, output_dir=None, beta=None):
        pass


class GrpoTrainTest(unittest.TestCase):
    def test_supported_kwargs_filters_version_specific_options(self) -> None:
        filtered = grpo_train._supported_kwargs(
            ExampleConfig,
            {"output_dir": "out", "beta": 0.04, "future_option": True},
        )

        self.assertEqual(filtered, {"output_dir": "out", "beta": 0.04})

    def test_completion_text_supports_conversational_outputs(self) -> None:
        self.assertEqual(
            grpo_train._completion_text([{"role": "assistant", "content": "```python\npass\n```"}]),
            "```python\npass\n```",
        )


if __name__ == "__main__":
    unittest.main()
