from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoace.schema import (
    ExecutionReport,
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestSuite,
)


SCRIPT = Path("scripts/verify_oracles.py").resolve()
SPEC = importlib.util.spec_from_file_location("verify_oracles", SCRIPT)
assert SPEC and SPEC.loader
verify_oracles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_oracles)


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, *args, **kwargs) -> ExecutionReport:
        self.calls += 1
        return ExecutionReport(True)


class VerifyOraclesTest(unittest.TestCase):
    def test_stops_after_first_verified_solution(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="p", statement="Solve."),
            TestSuite(),
            OracleMetadata(
                [
                    OracleSolution("python3", "print(1)"),
                    OracleSolution("python3", "print(2)"),
                    OracleSolution("python3", "print(3)"),
                ]
            ),
        )
        executor = CountingExecutor()

        updated, detail = verify_oracles.verify_bundle(bundle, executor, 3)

        self.assertEqual(executor.calls, 1)
        self.assertEqual(len(detail["attempts"]), 1)
        self.assertTrue(updated.oracle.solutions[0].verified)
        self.assertFalse(updated.oracle.solutions[1].verified)


if __name__ == "__main__":
    unittest.main()
