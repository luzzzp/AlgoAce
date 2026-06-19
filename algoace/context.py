from __future__ import annotations

from dataclasses import dataclass

from algoace.schema import (
    ExecutionReport,
    FailureKind,
    FailureSummary,
    ProblemSpec,
    TestCase,
)


CODE_OUTPUT_RULE = (
    "Return exactly one Python code block and no explanation:\n"
    "```python\n"
    "# complete solution\n"
    "```"
)


@dataclass(frozen=True)
class ContextPacket:
    context_type: str
    prompt: str


class FailureAnalyzer:
    def analyze(self, report: ExecutionReport, reveal_case: bool = True) -> FailureSummary:
        if not report.syntax_valid:
            return FailureSummary(FailureKind.SYNTAX_ERROR, report.syntax_error)
        failed = next((run for run in report.runs if not run.passed), None)
        if failed is None:
            return FailureSummary(FailureKind.NONE, "All tests passed.")
        if failed.timed_out:
            kind = FailureKind.TIMEOUT
        elif failed.output_truncated:
            kind = FailureKind.OUTPUT_TOO_LONG
        elif (failed.returncode or 0) != 0:
            kind = FailureKind.RUNTIME_ERROR
        else:
            kind = FailureKind.WRONG_ANSWER
        return FailureSummary(
            kind=kind,
            message=f"{kind.value} on {failed.test_id}",
            test_id=failed.test_id,
            stdin=failed.stdin if reveal_case else "",
            expected=failed.expected if reveal_case else "",
            actual=failed.actual if reveal_case else "",
            stderr=failed.stderr if reveal_case else "",
        )


class ContextBuilder:
    def initial(self, problem: ProblemSpec, visible_tests: list[TestCase]) -> ContextPacket:
        return ContextPacket(
            "INITIAL_SOLVE",
            f"{problem.prompt(visible_tests)}\n\n{CODE_OUTPUT_RULE}",
        )

    def visible_repair(
        self,
        problem: ProblemSpec,
        visible_tests: list[TestCase],
        previous_code: str,
        failure: FailureSummary,
        turn: int,
    ) -> ContextPacket:
        diagnostic = (
            f"Failure type: {failure.kind.value}\n"
            f"Failed test: {failure.test_id}\n"
            f"Input:\n{failure.stdin}\n"
            f"Expected:\n{failure.expected}\n"
            f"Actual:\n{failure.actual}\n"
            f"Stderr:\n{failure.stderr}"
        )
        return ContextPacket(
            "VISIBLE_REPAIR",
            (
                f"{problem.prompt(visible_tests)}\n\n"
                f"Previous candidate:\n```python\n{previous_code}\n```\n\n"
                f"Execution feedback for repair turn {turn}:\n{diagnostic}\n\n"
                "Identify the root cause internally, then return a corrected complete program. "
                "Do not hardcode the visible examples.\n\n"
                f"{CODE_OUTPUT_RULE}"
            ),
        )

    def hidden_review(
        self,
        problem: ProblemSpec,
        visible_tests: list[TestCase],
        previous_code: str,
        turn: int,
    ) -> ContextPacket:
        return ContextPacket(
            "HIDDEN_REVIEW",
            (
                f"{problem.prompt(visible_tests)}\n\n"
                f"Previous candidate:\n```python\n{previous_code}\n```\n\n"
                f"Internal verification failed after turn {turn}. Hidden test inputs and outputs are withheld. "
                "Re-check the algorithm against the full statement, boundary cases, input/output handling, "
                "numeric limits, duplicate values, empty or singleton cases, and time complexity. "
                "Do not hardcode visible examples. Return a revised complete program only.\n\n"
                f"{CODE_OUTPUT_RULE}"
            ),
        )
