from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("training/sft_train.py").resolve()
SPEC = importlib.util.spec_from_file_location("sft_train", SCRIPT)
assert SPEC and SPEC.loader
sft_train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sft_train)


class SftTrainTest(unittest.TestCase):
    def test_completion_only_labels_mask_prompt(self) -> None:
        packed = sft_train._pack_token_ids([1, 2, 3], [4, 5], max_length=8)

        self.assertEqual(packed["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(packed["labels"], [-100, -100, -100, 4, 5])

    def test_long_prompt_keeps_head_and_tail_without_truncating_code(self) -> None:
        packed = sft_train._pack_token_ids(
            list(range(20)),
            [100, 101, 102],
            max_length=11,
        )

        self.assertEqual(len(packed["input_ids"]), 11)
        self.assertEqual(packed["input_ids"][-3:], [100, 101, 102])
        self.assertEqual(packed["input_ids"][:6], [0, 1, 2, 3, 4, 5])
        self.assertEqual(packed["input_ids"][6:8], [18, 19])
        self.assertEqual(packed["labels"][-3:], [100, 101, 102])
        self.assertTrue(all(value == -100 for value in packed["labels"][:-3]))

    def test_rejects_completion_that_fills_context(self) -> None:
        with self.assertRaises(ValueError):
            sft_train._pack_token_ids([1], [2, 3, 4], max_length=3)


if __name__ == "__main__":
    unittest.main()
