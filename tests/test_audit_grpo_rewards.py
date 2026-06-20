from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoace.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
)


SCRIPT = Path("scripts/audit_grpo_rewards.py").resolve()
SPEC = importlib.util.spec_from_file_location("audit_grpo_rewards", SCRIPT)
assert SPEC and SPEC.loader
audit_grpo_rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_grpo_rewards)


class AuditGrpoRewardsTest(unittest.TestCase):
    def test_verified_oracle_outranks_bad_completion(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="sum", statement="Add two integers."),
            TestSuite(
                visible_tests=[TestCase("2 3", "5", "visible-0001")],
                reward_tests=[TestCase("10 20", "30", "reward-0001")],
            ),
            OracleMetadata(
                [OracleSolution("python3", "a,b=map(int,input().split())\nprint(a+b)", True)]
            ),
        )

        report = audit_grpo_rewards.audit_rewards([bundle])

        self.assertEqual(report["ordering_violations"], 0)
        self.assertGreater(report["oracle_minus_bad_margin"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
