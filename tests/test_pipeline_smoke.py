from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTest(unittest.TestCase):
    def test_verify_then_build_code_sft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            verified = root / "verified"
            dataset = root / "code_sft.jsonl"
            raw.mkdir()
            (raw / "sum.json").write_text(
                json.dumps(_problem(), ensure_ascii=False),
                encoding="utf-8",
            )

            verify = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_oracles.py",
                    "--problems",
                    str(raw),
                    "--out-dir",
                    str(verified),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)

            verify_resume = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_oracles.py",
                    "--problems",
                    str(raw),
                    "--out-dir",
                    str(verified),
                    "--resume",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(verify_resume.returncode, 0, verify_resume.stderr)
            self.assertEqual(json.loads(verify_resume.stdout)["skipped_existing"], 1)

            training_only = _problem()
            training_only["problem"]["id"] = "training_only"
            training_only["tests"]["reward_tests"] = []
            training_only["tests"]["eval_tests"] = []
            training_only["oracle"]["solutions"][0]["verified"] = True
            (verified / "training_only.json").write_text(
                json.dumps(training_only, ensure_ascii=False),
                encoding="utf-8",
            )

            split_root = root / "split"
            split = subprocess.run(
                [
                    sys.executable,
                    "scripts/split_problems.py",
                    "--problems",
                    str(verified),
                    "--out-dir",
                    str(split_root),
                    "--train-ratio",
                    "0",
                    "--dev-ratio",
                    "0",
                    "--test-ratio",
                    "1",
                    "--seed",
                    "42",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(split.returncode, 0, split.stderr)
            self.assertTrue((split_root / "test" / "sum.json").exists())
            self.assertTrue((split_root / "train" / "training_only.json").exists())
            self.assertFalse((split_root / "dev" / "training_only.json").exists())
            self.assertFalse((split_root / "test" / "training_only.json").exists())
            role = json.loads((split_root / "test" / "_split_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(role["role"], "test")
            train_role = json.loads(
                (split_root / "train" / "_split_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(train_role["training_only_count"], 1)

            build = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_code_sft.py",
                    "--problems",
                    str(verified),
                    "--out",
                    str(dataset),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            records = dataset.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 2)
            payload = json.loads(records[0])
            self.assertIn("```python", payload["output"])
            self.assertNotIn("Solution Explanation", payload["output"])

            build_resume = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_code_sft.py",
                    "--problems",
                    str(verified),
                    "--out",
                    str(dataset),
                    "--resume",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(build_resume.returncode, 0, build_resume.stderr)
            self.assertEqual(len(dataset.read_text(encoding="utf-8").splitlines()), 2)

            rejected = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_code_sft.py",
                    "--problems",
                    str(split_root / "test"),
                    "--out",
                    str(root / "forbidden.jsonl"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Refusing to build training data", rejected.stderr)


def _problem() -> dict:
    return {
        "problem": {
            "id": "sum",
            "statement": "Read two integers and print their sum.",
            "input_format": "Two integers.",
            "output_format": "Their sum.",
            "constraints": [],
            "language": "python3",
            "time_limit_sec": 2.0,
            "memory_limit_mb": 256,
            "io_mode": "stdin",
            "entry_point": "",
        },
        "tests": {
            "visible_tests": [{"stdin": "2 3\n", "expected_stdout": "5\n"}],
            "reward_tests": [{"stdin": "10 20\n", "expected_stdout": "30\n"}],
            "eval_tests": [{"stdin": "7 8\n", "expected_stdout": "15\n"}],
        },
        "oracle": {
            "solutions": [
                {
                    "language": "python3",
                    "code": "a,b=map(int,input().split())\nprint(a+b)\n",
                    "verified": False,
                }
            ],
            "source": "mock",
            "url": "",
        },
    }


if __name__ == "__main__":
    unittest.main()
