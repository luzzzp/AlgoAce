from __future__ import annotations

import unittest

from algoace.reward import score_completion
from algoace.schema import ProblemBundle, ProblemSpec, TestCase, TestSuite


def _bundle(timeout: float | None = None) -> ProblemBundle:
    return ProblemBundle(
        ProblemSpec(id="sum", statement="Sum two integers."),
        TestSuite(
            visible_tests=[TestCase("2 3\n", "5\n", "visible-0001", timeout)],
            reward_tests=[TestCase("10 20\n", "30\n", "reward-0001", timeout)],
        ),
    )


class RewardTest(unittest.TestCase):
    def test_correct_code_gets_high_reward(self) -> None:
        text = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
        reward = score_completion(text, _bundle())
        self.assertGreaterEqual(reward.total, 1.0)

    def test_wrong_code_gets_lower_reward(self) -> None:
        text = "```python\na,b=map(int,input().split())\nprint(a-b)\n```"
        reward = score_completion(text, _bundle())
        self.assertLess(reward.total, 0.5)

    def test_missing_code_block_is_rejected(self) -> None:
        reward = score_completion("print(1)", _bundle())
        self.assertEqual(reward.total, -1.0)

    def test_timeout_is_strongly_penalized(self) -> None:
        reward = score_completion("```python\nwhile True: pass\n```", _bundle(0.1))
        self.assertIn("timeout", reward.reasons)
        self.assertLess(reward.total, 0.0)

    def test_missing_reward_tests_do_not_get_free_reward(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="sum", statement="Sum two integers."),
            TestSuite(visible_tests=[TestCase("2 3\n", "5\n", "visible-0001")]),
        )
        reward = score_completion(
            "```python\na,b=map(int,input().split())\nprint(a+b)\n```",
            bundle,
        )

        self.assertEqual(reward.reward_tests, 0.0)
        self.assertLess(reward.total, 0.5)


if __name__ == "__main__":
    unittest.main()
