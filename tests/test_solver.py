from __future__ import annotations

import unittest

from algoace.model_client import ModelResponse, RuleBasedModel
from algoace.schema import ProblemSpec, SolveStatus, TestCase, TestSuite
from algoace.solver import AlgoAceSolver


class StaticSequenceModel:
    def __init__(self, codes: list[str]):
        self.codes = codes
        self.calls = 0

    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        code = self.codes[min(self.calls, len(self.codes) - 1)]
        self.calls += 1
        return [ModelResponse(f"```python\n{code}\n```", code) for _ in range(count)]


class MultiCandidateModel:
    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        codes = [
            "a,b=map(int,input().split())\nprint(a-b)",
            "a,b=map(int,input().split())\nprint(a+b)",
        ]
        return [ModelResponse(f"```python\n{code}\n```", code) for code in codes[:count]]


class RewardRerankModel:
    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        codes = [
            "a,b=map(int,input().split())\nprint(5 if (a,b)==(2,3) else 0)",
            "a,b=map(int,input().split())\nprint(a+b)",
        ]
        return [ModelResponse(f"```python\n{code}\n```", code) for code in codes[:count]]


class SolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ProblemSpec(id="sum", statement="Read two integers and print their sum.")
        self.tests = TestSuite(
            visible_tests=[TestCase("2 3\n", "5\n", "visible-0001")],
            reward_tests=[TestCase("10 20\n", "30\n", "reward-0001")],
            eval_tests=[TestCase("7 8\n", "15\n", "eval-0001")],
        )

    def test_repairs_visible_failure(self) -> None:
        result = AlgoAceSolver(RuleBasedModel(), max_repair_turns=1).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.VERIFIED_SOLVED)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.attempt_records[1].context_type, "VISIBLE_REPAIR")

    def test_best_of_n_selects_passing_candidate(self) -> None:
        result = AlgoAceSolver(
            MultiCandidateModel(),
            max_repair_turns=0,
            candidates_per_turn=2,
        ).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.VERIFIED_SOLVED)
        self.assertEqual(result.attempt_records[0].selected_index, 1)

    def test_reward_failure_can_trigger_hidden_review(self) -> None:
        visible_only = "a,b=map(int,input().split())\nprint(5)"
        correct = "a,b=map(int,input().split())\nprint(a+b)"
        model = StaticSequenceModel([visible_only, correct])

        result = AlgoAceSolver(model, max_repair_turns=1).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.VERIFIED_SOLVED)
        self.assertEqual(result.attempt_records[1].context_type, "HIDDEN_REVIEW")

    def test_reward_tests_can_rerank_visible_passing_candidates(self) -> None:
        result = AlgoAceSolver(
            RewardRerankModel(),
            max_repair_turns=0,
            candidates_per_turn=2,
            rerank_with_reward_tests=True,
        ).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.VERIFIED_SOLVED)
        self.assertEqual(result.attempt_records[0].selected_index, 1)
        self.assertTrue(result.attempt_records[0].reward_result.all_passed)

    def test_reward_rerank_failure_does_not_reach_eval_tests(self) -> None:
        visible_only = "a,b=map(int,input().split())\nprint(5)"
        result = AlgoAceSolver(
            StaticSequenceModel([visible_only]),
            max_repair_turns=0,
            candidates_per_turn=1,
            rerank_with_reward_tests=True,
        ).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.FAILED)
        self.assertEqual(result.failure_reason, "INTERNAL_REWARD_FAILED")
        self.assertIsNone(result.attempt_records[0].eval_result)

    def test_eval_failure_does_not_trigger_repair(self) -> None:
        visible_reward_only = (
            "a,b=map(int,input().split())\n"
            "print(5 if (a,b)==(2,3) else 30)"
        )
        model = StaticSequenceModel([visible_reward_only, "a,b=map(int,input().split())\nprint(a+b)"])

        result = AlgoAceSolver(model, max_repair_turns=3).solve(self.problem, self.tests)

        self.assertEqual(result.status, SolveStatus.FAILED)
        self.assertEqual(result.failure_reason, "INTERNAL_EVAL_FAILED")
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
