from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from algoace.schema import ExecutionReport, ExecutionRun, TestCase, normalize_output


class PythonExecutor:
    def __init__(self, max_stdout_bytes: int = 1_000_000):
        self.max_stdout_bytes = max_stdout_bytes

    def evaluate(
        self,
        code: str,
        tests: list[TestCase],
        default_timeout_sec: float = 2.0,
        suite_name: str = "tests",
        entry_point: str = "",
    ) -> ExecutionReport:
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return ExecutionReport(False, syntax_error=str(exc))

        with tempfile.TemporaryDirectory(prefix="algoace_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(_wrapped_code(code, entry_point), encoding="utf-8")
            runs = [
                self._run_one(script, case, default_timeout_sec, suite_name)
                for case in tests
            ]
        return ExecutionReport(True, runs=runs)

    def _run_one(
        self,
        script: Path,
        case: TestCase,
        default_timeout_sec: float,
        suite_name: str,
    ) -> ExecutionRun:
        timeout = case.timeout_sec or default_timeout_sec
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                input=case.stdin.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionRun(
                test_id=case.id,
                suite=suite_name,
                passed=False,
                stdin=case.stdin,
                expected=normalize_output(case.expected_stdout),
                actual=normalize_output(exc.stdout),
                stderr=normalize_output(exc.stderr),
                timed_out=True,
            )

        stdout = completed.stdout or b""
        truncated = len(stdout) > self.max_stdout_bytes
        if truncated:
            stdout = stdout[: self.max_stdout_bytes]
        actual = normalize_output(stdout)
        expected = normalize_output(case.expected_stdout)
        return ExecutionRun(
            test_id=case.id,
            suite=suite_name,
            passed=completed.returncode == 0 and actual == expected and not truncated,
            stdin=case.stdin,
            expected=expected,
            actual=actual,
            stderr=normalize_output(completed.stderr),
            returncode=completed.returncode,
            output_truncated=truncated,
        )


def _wrapped_code(code: str, entry_point: str) -> str:
    if not entry_point:
        return code
    return (
        f"{code.rstrip()}\n\n"
        "if __name__ == '__main__':\n"
        "    import json as _algoace_json\n"
        "    import sys as _algoace_sys\n"
        "    _payload = _algoace_json.loads(_algoace_sys.stdin.read())\n"
        "    _args = _payload if isinstance(_payload, list) else [_payload]\n"
        f"    _result = {entry_point}(*_args)\n"
        "    print(_algoace_json.dumps(_result, ensure_ascii=False))\n"
    )
