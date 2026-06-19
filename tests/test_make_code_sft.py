from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from algoace.schema import OracleMetadata, OracleSolution, ProblemBundle, ProblemSpec, TestCase, TestSuite


SCRIPT = Path("scripts/make_code_sft.py").resolve()
SPEC = importlib.util.spec_from_file_location("make_code_sft", SCRIPT)
assert SPEC and SPEC.loader
make_code_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_code_sft)


class MakeCodeSftTest(unittest.TestCase):
    def test_record_is_code_only(self) -> None:
        bundle = ProblemBundle(
            ProblemSpec(id="sum", statement="Sum two integers."),
            TestSuite(visible_tests=[TestCase("2 3\n", "5\n", "visible-0001")]),
            OracleMetadata([OracleSolution("python3", "print(sum(map(int,input().split())))", True)]),
        )

        record = make_code_sft._record(bundle)

        self.assertEqual(set(record), {"instruction", "input", "output"})
        self.assertTrue(record["output"].startswith("```python\n"))
        self.assertNotIn("Solution Explanation", record["output"])
        self.assertNotIn("Time Complexity", record["output"])


if __name__ == "__main__":
    unittest.main()

