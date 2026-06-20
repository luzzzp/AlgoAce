from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/summarize_grpo_rewards.py").resolve()
SPEC = importlib.util.spec_from_file_location("summarize_grpo_rewards", SCRIPT)
assert SPEC and SPEC.loader
summarize_grpo_rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_grpo_rewards)


class SummarizeGrpoRewardsTest(unittest.TestCase):
    def test_reports_zero_variance_groups(self) -> None:
        rows = [
            {"reward_call": 1, "problem_id": "a", "total": 0.1, "code_chars": 10, "code_hash": "a"},
            {"reward_call": 1, "problem_id": "a", "total": 0.9, "code_chars": 20, "code_hash": "b"},
            {"reward_call": 1, "problem_id": "b", "total": 0.2, "code_chars": 10, "code_hash": "c"},
            {"reward_call": 1, "problem_id": "b", "total": 0.2, "code_chars": 10, "code_hash": "c"},
        ]

        report = summarize_grpo_rewards.summarize(rows)

        self.assertEqual(report["groups"], 2)
        self.assertEqual(report["varying_reward_groups"], 1)
        self.assertEqual(report["zero_variance_group_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
