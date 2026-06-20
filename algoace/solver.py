from __future__ import annotations

from algoace.context import ContextBuilder, ContextPacket, FailureAnalyzer
from algoace.executor import PythonExecutor
from algoace.model_client import ModelClient
from algoace.schema import (
    AttemptRecord,
    CandidateAttempt,
    ExecutionReport,
    FailureKind,
    ProblemSpec,
    SolveResult,
    SolveStatus,
    TestCase,
    TestSuite,
)


class AlgoAceSolver:
    def __init__(
        self,
        model: ModelClient,
        executor: PythonExecutor | None = None,
        max_repair_turns: int = 3,
        candidates_per_turn: int = 1,
        rerank_with_reward_tests: bool = False,
    ):
        self.model = model
        self.executor = executor or PythonExecutor()
        self.max_repair_turns = max_repair_turns
        self.candidates_per_turn = candidates_per_turn
        self.rerank_with_reward_tests = rerank_with_reward_tests
        self.context_builder = ContextBuilder()
        self.failure_analyzer = FailureAnalyzer()

    def solve(self, problem: ProblemSpec, tests: TestSuite) -> SolveResult:
        if not tests.visible_tests:
            return SolveResult(
                problem_id=problem.id,
                status=SolveStatus.FAILED,
                attempts=0,
                code=None,
                failure_reason="MISSING_VISIBLE_TESTS",
                diagnostic_summary="At least one visible test is required for candidate selection.",
            )

        records: list[AttemptRecord] = []
        packet = self.context_builder.initial(problem, tests.visible_tests)
        last_diagnostic = ""

        for turn in range(1, self.max_repair_turns + 2):
            candidates = self._generate_and_evaluate(problem, tests, packet)
            selected_index = max(range(len(candidates)), key=lambda index: candidates[index].score)
            reward_result = None
            if self.rerank_with_reward_tests and tests.reward_tests:
                selected_index, reward_result = self._reward_rerank(
                    problem,
                    tests,
                    candidates,
                    selected_index,
                )
            selected = candidates[selected_index]
            records.append(
                AttemptRecord(
                    turn=turn,
                    context_type=packet.context_type,
                    candidates=candidates,
                    selected_index=selected_index,
                    reward_result=reward_result,
                )
            )

            if selected.visible_result.all_passed:
                if tests.reward_tests:
                    if reward_result is None:
                        reward_result = self._evaluate(problem, selected.code, tests.reward_tests, "reward")
                        records[-1] = AttemptRecord(
                            turn=turn,
                            context_type=packet.context_type,
                            candidates=candidates,
                            selected_index=selected_index,
                            reward_result=reward_result,
                        )
                    if not reward_result.all_passed:
                        last_diagnostic = "Internal reward verification failed; cases are withheld."
                        packet = self.context_builder.hidden_review(
                            problem,
                            tests.visible_tests,
                            selected.code,
                            turn,
                        )
                        continue

                if not tests.eval_tests:
                    return SolveResult(
                        problem_id=problem.id,
                        status=SolveStatus.VISIBLE_SOLVED,
                        attempts=turn,
                        code=selected.code,
                        failure_reason=None,
                        diagnostic_summary="Visible and reward tests passed; no held-out tests configured.",
                        attempt_records=records,
                    )
                eval_result = self._evaluate(problem, selected.code, tests.eval_tests, "eval")
                records[-1] = AttemptRecord(
                    turn=turn,
                    context_type=packet.context_type,
                    candidates=candidates,
                    selected_index=selected_index,
                    reward_result=reward_result,
                    eval_result=eval_result,
                )
                if eval_result.all_passed:
                    return SolveResult(
                        problem_id=problem.id,
                        status=SolveStatus.VERIFIED_SOLVED,
                        attempts=turn,
                        code=selected.code,
                        failure_reason=None,
                        diagnostic_summary="Visible and held-out tests passed.",
                        attempt_records=records,
                    )
                return SolveResult(
                    problem_id=problem.id,
                    status=SolveStatus.FAILED,
                    attempts=turn,
                    code=None,
                    failure_reason=FailureKind.INTERNAL_EVAL_FAILED.value,
                    diagnostic_summary="Final held-out evaluation failed; no repair feedback was provided.",
                    attempt_records=records,
                )

            failure = self.failure_analyzer.analyze(selected.visible_result, reveal_case=True)
            last_diagnostic = failure.message
            packet = self.context_builder.visible_repair(
                problem,
                tests.visible_tests,
                selected.code,
                failure,
                turn,
            )

        return SolveResult(
            problem_id=problem.id,
            status=SolveStatus.FAILED,
            attempts=len(records),
            code=None,
            failure_reason=_final_failure(records),
            diagnostic_summary=last_diagnostic,
            attempt_records=records,
        )

    def _generate_and_evaluate(
        self,
        problem: ProblemSpec,
        tests: TestSuite,
        packet: ContextPacket,
    ) -> list[CandidateAttempt]:
        responses = self.model.generate_codes(packet.prompt, self.candidates_per_turn)
        candidates = []
        for response in responses:
            if response.code:
                report = self._evaluate(problem, response.code, tests.visible_tests, "visible")
            else:
                report = ExecutionReport(False, syntax_error="Missing Python code block.")
            candidates.append(
                CandidateAttempt(
                    code=response.code,
                    visible_result=report,
                    score=_candidate_score(report),
                )
            )
        if not candidates:
            candidates.append(
                CandidateAttempt(
                    code="",
                    visible_result=ExecutionReport(False, syntax_error="Model returned no candidates."),
                    score=-1.0,
                )
            )
        return candidates

    def _reward_rerank(
        self,
        problem: ProblemSpec,
        tests: TestSuite,
        candidates: list[CandidateAttempt],
        default_index: int,
    ) -> tuple[int, ExecutionReport | None]:
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.visible_result.all_passed
        ]
        if not eligible:
            return default_index, None
        reward_results = {
            index: self._evaluate(
                problem,
                candidates[index].code,
                tests.reward_tests,
                "reward",
            )
            for index in eligible
        }
        selected_index = max(
            eligible,
            key=lambda index: (
                int(reward_results[index].all_passed),
                reward_results[index].pass_rate,
                candidates[index].score,
                -index,
            ),
        )
        return selected_index, reward_results[selected_index]

    def _evaluate(
        self,
        problem: ProblemSpec,
        code: str,
        cases: list[TestCase],
        suite_name: str,
    ) -> ExecutionReport:
        return self.executor.evaluate(
            code,
            cases,
            default_timeout_sec=problem.time_limit_sec,
            suite_name=suite_name,
            entry_point=problem.entry_point if problem.io_mode == "callable" else "",
        )


def _candidate_score(report: ExecutionReport) -> float:
    if not report.syntax_valid:
        return -1.0
    penalty = 0.0
    if any(run.timed_out for run in report.runs):
        penalty += 0.5
    if any((run.returncode or 0) != 0 and not run.timed_out for run in report.runs):
        penalty += 0.25
    if any(run.output_truncated for run in report.runs):
        penalty += 0.5
    return report.pass_rate - penalty


def _final_failure(records: list[AttemptRecord]) -> str:
    if not records:
        return FailureKind.MISSING_CODE.value
    last = records[-1]
    if last.eval_result is not None and not last.eval_result.all_passed:
        return FailureKind.INTERNAL_EVAL_FAILED.value
    if last.reward_result is not None and not last.reward_result.all_passed:
        return "INTERNAL_REWARD_FAILED"
    selected = last.candidates[last.selected_index]
    if not selected.visible_result.syntax_valid:
        return FailureKind.SYNTAX_ERROR.value
    return FailureAnalyzer().analyze(selected.visible_result).kind.value
