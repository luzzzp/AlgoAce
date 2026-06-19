from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class SolveStatus(str, Enum):
    VISIBLE_SOLVED = "VISIBLE_SOLVED"
    VERIFIED_SOLVED = "VERIFIED_SOLVED"
    FAILED = "FAILED"


class FailureKind(str, Enum):
    NONE = "NONE"
    MISSING_CODE = "MISSING_CODE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    WRONG_ANSWER = "WRONG_ANSWER"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    OUTPUT_TOO_LONG = "OUTPUT_TOO_LONG"
    INTERNAL_EVAL_FAILED = "INTERNAL_EVAL_FAILED"


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: list[str] = field(default_factory=list)
    language: str = "python3"
    time_limit_sec: float = 2.0
    memory_limit_mb: int = 256
    io_mode: str = "stdin"
    entry_point: str = ""

    def prompt(self, visible_tests: list[TestCase] | None = None) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints) or "- Not specified"
        mode = ""
        if self.io_mode == "callable" and self.entry_point:
            mode = (
                f"Callable entry point: {self.entry_point}\n"
                "Implement this function directly. Do not read stdin unless the statement requires it.\n"
            )
        samples = ""
        if visible_tests:
            blocks = [
                f"Input:\n{case.stdin.rstrip()}\nExpected output:\n{case.expected_stdout.rstrip()}"
                for case in visible_tests
            ]
            samples = "\n\nVisible samples:\n" + "\n\n".join(blocks)
        return (
            f"Problem statement:\n{self.statement}\n\n"
            f"Input format:\n{self.input_format}\n\n"
            f"Output format:\n{self.output_format}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Language: Python 3\n"
            f"Time limit: {self.time_limit_sec:g} seconds\n"
            f"Memory limit: {self.memory_limit_mb} MB\n"
            f"{mode}{samples}"
        )


@dataclass(frozen=True)
class TestCase:
    stdin: str
    expected_stdout: str
    id: str = ""
    timeout_sec: float | None = None


@dataclass(frozen=True)
class TestSuite:
    visible_tests: list[TestCase] = field(default_factory=list)
    reward_tests: list[TestCase] = field(default_factory=list)
    eval_tests: list[TestCase] = field(default_factory=list)


@dataclass(frozen=True)
class OracleSolution:
    language: str
    code: str
    verified: bool = False


@dataclass(frozen=True)
class OracleMetadata:
    solutions: list[OracleSolution] = field(default_factory=list)
    source: str = ""
    url: str = ""

    def best_verified(self, language: str = "python3") -> str:
        for solution in self.solutions:
            if solution.language == language and solution.verified:
                return solution.code
        return ""


@dataclass(frozen=True)
class ProblemBundle:
    spec: ProblemSpec
    tests: TestSuite
    oracle: OracleMetadata = field(default_factory=OracleMetadata)


@dataclass(frozen=True)
class ExecutionRun:
    test_id: str
    suite: str
    passed: bool
    stdin: str
    expected: str
    actual: str
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    output_truncated: bool = False


@dataclass(frozen=True)
class ExecutionReport:
    syntax_valid: bool
    syntax_error: str = ""
    runs: list[ExecutionRun] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.syntax_valid and all(run.passed for run in self.runs)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 1.0 if self.syntax_valid else 0.0
        return sum(run.passed for run in self.runs) / len(self.runs)


@dataclass(frozen=True)
class FailureSummary:
    kind: FailureKind
    message: str
    test_id: str = ""
    stdin: str = ""
    expected: str = ""
    actual: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CandidateAttempt:
    code: str
    visible_result: ExecutionReport
    score: float


@dataclass(frozen=True)
class AttemptRecord:
    turn: int
    context_type: str
    candidates: list[CandidateAttempt]
    selected_index: int
    reward_result: ExecutionReport | None = None
    eval_result: ExecutionReport | None = None


@dataclass(frozen=True)
class SolveResult:
    problem_id: str
    status: SolveStatus
    attempts: int
    code: str | None
    failure_reason: str | None
    diagnostic_summary: str
    attempt_records: list[AttemptRecord] = field(default_factory=list)


def normalize_output(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def load_problems(path: str | Path) -> list[ProblemBundle]:
    root = Path(path)
    if root.is_file():
        return [load_problem(root)]
    return [load_problem(item) for item in sorted(root.glob("*.json")) if not item.name.startswith("_")]


def load_problem(path: str | Path) -> ProblemBundle:
    return problem_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_problem(bundle: ProblemBundle, path: str | Path) -> None:
    Path(path).write_text(json.dumps(problem_to_dict(bundle), ensure_ascii=False, indent=2), encoding="utf-8")


def problem_from_dict(payload: dict[str, Any]) -> ProblemBundle:
    problem = payload["problem"]
    tests = payload.get("tests", {})
    oracle = payload.get("oracle", {})
    spec = ProblemSpec(
        id=str(problem["id"]),
        statement=str(problem.get("statement") or ""),
        input_format=str(problem.get("input_format") or ""),
        output_format=str(problem.get("output_format") or ""),
        constraints=[str(item) for item in problem.get("constraints", [])],
        language=str(problem.get("language") or "python3"),
        time_limit_sec=float(problem.get("time_limit_sec", 2.0)),
        memory_limit_mb=int(problem.get("memory_limit_mb", 256)),
        io_mode=str(problem.get("io_mode") or "stdin"),
        entry_point=str(problem.get("entry_point") or ""),
    )
    return ProblemBundle(
        spec=spec,
        tests=TestSuite(
            visible_tests=_cases(tests.get("visible_tests", []), "visible"),
            reward_tests=_cases(tests.get("reward_tests", []), "reward"),
            eval_tests=_cases(tests.get("eval_tests", []), "eval"),
        ),
        oracle=OracleMetadata(
            solutions=[
                OracleSolution(
                    language=str(item.get("language") or "python3"),
                    code=str(item.get("code") or ""),
                    verified=bool(item.get("verified")),
                )
                for item in oracle.get("solutions", [])
                if item.get("code")
            ],
            source=str(oracle.get("source") or ""),
            url=str(oracle.get("url") or ""),
        ),
    )


def problem_to_dict(bundle: ProblemBundle) -> dict[str, Any]:
    return {
        "problem": asdict(bundle.spec),
        "tests": {
            "visible_tests": [asdict(case) for case in bundle.tests.visible_tests],
            "reward_tests": [asdict(case) for case in bundle.tests.reward_tests],
            "eval_tests": [asdict(case) for case in bundle.tests.eval_tests],
        },
        "oracle": {
            "solutions": [asdict(solution) for solution in bundle.oracle.solutions],
            "source": bundle.oracle.source,
            "url": bundle.oracle.url,
        },
    }


def result_to_dict(result: SolveResult) -> dict[str, Any]:
    return _json_ready(asdict(result))


def _cases(items: list[dict[str, Any]], prefix: str) -> list[TestCase]:
    return [
        TestCase(
            stdin=str(item.get("stdin") or ""),
            expected_stdout=str(item.get("expected_stdout") or ""),
            id=str(item.get("id") or f"{prefix}-{index:04d}"),
            timeout_sec=float(item["timeout_sec"]) if item.get("timeout_sec") is not None else None,
        )
        for index, item in enumerate(items, start=1)
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
