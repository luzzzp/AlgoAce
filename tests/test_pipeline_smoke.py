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
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0])
            self.assertIn("```python", payload["output"])
            self.assertNotIn("Solution Explanation", payload["output"])


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

