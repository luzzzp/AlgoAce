from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.executor import PythonExecutor, has_syntax_warning
from algoace.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    load_problem,
    load_problems,
    save_problem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python oracle solutions with all available tests.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-solutions-per-problem", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of problems verified concurrently. Each problem still runs its tests sequentially.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    bundles = load_problems(args.problems)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "_failed.jsonl"
    verified = 0
    failed = 0
    no_solution = 0
    processed_new = 0
    skipped_existing = 0
    pending: list[ProblemBundle] = []

    failed_mode = "a" if args.resume else "w"
    with failed_path.open(failed_mode, encoding="utf-8") as failed_log:
        for bundle in bundles:
            target = out_dir / f"{bundle.spec.id}.json"
            if args.resume and target.exists():
                updated = load_problem(target)
                skipped_existing += 1
                verified, failed, no_solution = _update_counts(
                    updated,
                    bundle,
                    verified,
                    failed,
                    no_solution,
                )
            else:
                pending.append(bundle)

        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures: dict[Future, ProblemBundle] = {}
        try:
            futures = {
                pool.submit(
                    _verify_one,
                    bundle,
                    args.max_solutions_per_problem,
                ): bundle
                for bundle in pending
            }
            for future in as_completed(futures):
                bundle = futures[future]
                updated, detail = future.result()
                save_problem(updated, out_dir / f"{bundle.spec.id}.json")
                processed_new += 1
                verified, failed, no_solution = _update_counts(
                    updated,
                    bundle,
                    verified,
                    failed,
                    no_solution,
                )
                if not updated.oracle.best_verified() and bundle.oracle.solutions:
                    failed_log.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    failed_log.flush()
                completed = skipped_existing + processed_new
                if processed_new == 1 or processed_new % 100 == 0 or completed == len(bundles):
                    print(
                        f"[{completed}/{len(bundles)}] verified={verified} failed={failed} ",
                        f"no_solution={no_solution}",
                        file=sys.stderr,
                        flush=True,
                    )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            print(
                "Oracle verification interrupted. Completed problem files are safe; rerun with --resume.",
                file=sys.stderr,
                flush=True,
            )
            raise
        else:
            pool.shutdown(wait=True)
    for manifest in Path(args.problems).glob("_*.json"):
        if manifest.name != "_failed.jsonl":
            shutil.copy2(manifest, out_dir / manifest.name)

    report = {
        "stage": "verify_oracles",
        "total": len(bundles),
        "verified": verified,
        "failed": failed,
        "no_solution": no_solution,
        "processed_new": processed_new,
        "skipped_existing": skipped_existing,
        "workers": args.workers,
        "failed_log": "_failed.jsonl",
    }
    (out_dir / "_verify_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def verify_bundle(
    bundle: ProblemBundle,
    executor: PythonExecutor,
    max_solutions: int,
) -> tuple[ProblemBundle, dict]:
    tests = [*bundle.tests.visible_tests, *bundle.tests.reward_tests, *bundle.tests.eval_tests]
    entry_point = bundle.spec.entry_point if bundle.spec.io_mode == "callable" else ""
    attempts = []
    selected_index: int | None = None
    warning_fallback: int | None = None
    for index, solution in enumerate(bundle.oracle.solutions[:max_solutions]):
        result = executor.evaluate(
            solution.code,
            tests,
            bundle.spec.time_limit_sec,
            "oracle",
            entry_point,
        )
        syntax_warning = has_syntax_warning(solution.code)
        attempts.append(
            {
                "solution_index": index,
                "syntax_valid": result.syntax_valid,
                "syntax_warning": syntax_warning,
                "pass_rate": result.pass_rate,
                "first_failed": _first_failed(result),
            }
        )
        if not result.all_passed:
            continue
        if not syntax_warning:
            selected_index = index
            break
        if warning_fallback is None:
            warning_fallback = index
    if selected_index is None:
        selected_index = warning_fallback
    updated_solutions = [
        OracleSolution(solution.language, solution.code, index == selected_index)
        for index, solution in enumerate(bundle.oracle.solutions)
    ]
    found = selected_index is not None
    updated = ProblemBundle(
        spec=bundle.spec,
        tests=bundle.tests,
        oracle=OracleMetadata(updated_solutions, bundle.oracle.source, bundle.oracle.url),
    )
    return updated, {"problem_id": bundle.spec.id, "verified": found, "attempts": attempts}


def _verify_one(bundle: ProblemBundle, max_solutions: int) -> tuple[ProblemBundle, dict]:
    return verify_bundle(bundle, PythonExecutor(), max_solutions)


def _update_counts(
    updated: ProblemBundle,
    original: ProblemBundle,
    verified: int,
    failed: int,
    no_solution: int,
) -> tuple[int, int, int]:
    if updated.oracle.best_verified():
        verified += 1
    elif not original.oracle.solutions:
        no_solution += 1
    else:
        failed += 1
    return verified, failed, no_solution


def _first_failed(report) -> dict | None:
    if not report.syntax_valid:
        return {"kind": "SYNTAX_ERROR", "error": report.syntax_error}
    failed = next((run for run in report.runs if not run.passed), None)
    if failed is None:
        return None
    return {
        "test_id": failed.test_id,
        "timed_out": failed.timed_out,
        "returncode": failed.returncode,
        "expected": failed.expected[:500],
        "actual": failed.actual[:500],
        "stderr": failed.stderr[:500],
    }


if __name__ == "__main__":
    main()
