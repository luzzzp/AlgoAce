from __future__ import annotations

from dataclasses import dataclass, field
import re

from algoace.executor import PythonExecutor
from algoace.hf_model import extract_python_code
from algoace.schema import ExecutionReport, ProblemBundle


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    syntax: float = 0.0
    visible: float = 0.0
    reward_tests: float = 0.0
    stability: float = 0.0
    penalties: float = 0.0
    reasons: list[str] = field(default_factory=list)


def score_completion(
    text: str,
    bundle: ProblemBundle,
    executor: PythonExecutor | None = None,
) -> RewardBreakdown:
    executor = executor or PythonExecutor()
    code = extract_python_code(text)
    if not code:
        return RewardBreakdown(-1.0, penalties=-1.0, reasons=["missing_python_code_block"])
    entry_point = bundle.spec.entry_point if bundle.spec.io_mode == "callable" else ""
    visible = executor.evaluate(
        code,
        bundle.tests.visible_tests,
        bundle.spec.time_limit_sec,
        "visible",
        entry_point,
    )
    reward = executor.evaluate(
        code,
        bundle.tests.reward_tests,
        bundle.spec.time_limit_sec,
        "reward",
        entry_point,
    )
    return compute_reward(code, bundle, visible, reward)


def compute_reward(
    code: str,
    bundle: ProblemBundle,
    visible_result: ExecutionReport,
    reward_result: ExecutionReport,
) -> RewardBreakdown:
    reasons: list[str] = []
    syntax = 0.1 if visible_result.syntax_valid else 0.0
    visible = 0.2 * visible_result.pass_rate
    reward_tests = 0.7 * reward_result.pass_rate if reward_result.runs else 0.0
    has_timeout = _any_timeout(visible_result) or _any_timeout(reward_result)
    has_runtime_error = _any_runtime_error(visible_result) or _any_runtime_error(reward_result)
    stability = 0.1 if visible_result.syntax_valid and not has_timeout and not has_runtime_error else 0.0
    penalties = 0.0
    if has_timeout:
        penalties -= 0.7
        reasons.append("timeout")
    if has_runtime_error:
        penalties -= 0.4
        reasons.append("runtime_error")
    if _looks_like_sample_hardcode(code, bundle):
        penalties -= 0.8
        reasons.append("possible_sample_hardcode")
    if _has_truncated_output(visible_result) or _has_truncated_output(reward_result):
        penalties -= 0.5
        reasons.append("output_too_long")
    total = syntax + visible + reward_tests + stability + penalties
    return RewardBreakdown(
        total=round(total, 6),
        syntax=syntax,
        visible=visible,
        reward_tests=reward_tests,
        stability=stability,
        penalties=penalties,
        reasons=reasons,
    )


def _looks_like_sample_hardcode(code: str, bundle: ProblemBundle) -> bool:
    if bundle.spec.io_mode == "stdin" and not re.search(r"\binput\s*\(|sys\.stdin", code):
        return True
    compact = re.sub(r"\s+", "", code)
    for case in bundle.tests.visible_tests:
        expected = re.sub(r"\s+", "", case.expected_stdout)
        if len(expected) >= 3 and expected in compact:
            return True
    return False


def _any_timeout(report: ExecutionReport) -> bool:
    return any(run.timed_out for run in report.runs)


def _any_runtime_error(report: ExecutionReport) -> bool:
    return any((run.returncode or 0) != 0 and not run.timed_out for run in report.runs)


def _has_truncated_output(report: ExecutionReport) -> bool:
    return any(run.output_truncated for run in report.runs)
