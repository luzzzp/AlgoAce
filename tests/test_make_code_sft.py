from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_rejects_test_split_for_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_split_metadata.json").write_text(
                json.dumps({"role": "test"}),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                make_code_sft._assert_training_split(root, allow_nontrain=False)

    def test_resume_sidecar_consistency_is_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dataset.jsonl"
            done = out.with_suffix(out.suffix + ".done.jsonl")
            out.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                make_code_sft._validate_resume_files(out, done, resume=True)


if __name__ == "__main__":
    unittest.main()
