from __future__ import annotations

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
            script.write_text(code, encoding="utf-8")
            runner = Path(tmp) / "callable_runner.py"
            if entry_point:
                runner.write_text(_CALLABLE_RUNNER, encoding="utf-8")
            runs = [
                self._run_one(script, case, default_timeout_sec, suite_name, entry_point, runner)
                for case in tests
            ]
        return ExecutionReport(True, runs=runs)

    def _run_one(
        self,
        script: Path,
        case: TestCase,
        default_timeout_sec: float,
        suite_name: str,
        entry_point: str = "",
        runner: Path | None = None,
    ) -> ExecutionRun:
        timeout = case.timeout_sec or default_timeout_sec
        command = (
            [sys.executable, str(runner), str(script), entry_point]
            if entry_point and runner is not None
            else [sys.executable, str(script)]
        )
        try:
            completed = subprocess.run(
                command,
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


_CALLABLE_RUNNER = r"""
import json
import runpy
import sys


def resolve_target(namespace, entry_point):
    target = namespace.get(entry_point)
    if callable(target):
        return target
    solution_class = namespace.get("Solution")
    if isinstance(solution_class, type):
        target = getattr(solution_class(), entry_point, None)
        if callable(target):
            return target
    for value in list(namespace.values()):
        if not isinstance(value, type):
            continue
        try:
            instance = value()
        except TypeError:
            continue
        target = getattr(instance, entry_point, None)
        if callable(target):
            return target
    raise AttributeError(f"entry point not found: {entry_point}")


def main():
    script = sys.argv[1]
    entry_point = sys.argv[2]
    namespace = runpy.run_path(script)
    target = resolve_target(namespace, entry_point)
    raw = sys.stdin.read().strip()
    args = json.loads(raw) if raw else []
    if not isinstance(args, list):
        args = [args]
    result = target(*args)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
"""
