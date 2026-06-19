from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.executor import PythonExecutor
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    executor = PythonExecutor()
    failed_path = out_dir / "_failed.jsonl"
    verified = 0
    failed = 0
    no_solution = 0
    processed_new = 0
    skipped_existing = 0

    failed_mode = "a" if args.resume else "w"
    with failed_path.open(failed_mode, encoding="utf-8") as failed_log:
        for bundle in bundles:
            target = out_dir / f"{bundle.spec.id}.json"
            if args.resume and target.exists():
                updated = load_problem(target)
                detail = None
                skipped_existing += 1
            else:
                updated, detail = verify_bundle(bundle, executor, args.max_solutions_per_problem)
                save_problem(updated, target)
                processed_new += 1
            if updated.oracle.best_verified():
                verified += 1
            elif not bundle.oracle.solutions:
                no_solution += 1
            else:
                failed += 1
                if detail is not None:
                    failed_log.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    failed_log.flush()
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
    updated_solutions = []
    found = False
    for index, solution in enumerate(bundle.oracle.solutions):
        verified = False
        if index < max_solutions:
            result = executor.evaluate(
                solution.code,
                tests,
                bundle.spec.time_limit_sec,
                "oracle",
                entry_point,
            )
            verified = result.all_passed
            attempts.append(
                {
                    "solution_index": index,
                    "syntax_valid": result.syntax_valid,
                    "pass_rate": result.pass_rate,
                    "first_failed": _first_failed(result),
                }
            )
            found = found or verified
        updated_solutions.append(OracleSolution(solution.language, solution.code, verified))
    updated = ProblemBundle(
        spec=bundle.spec,
        tests=bundle.tests,
        oracle=OracleMetadata(updated_solutions, bundle.oracle.source, bundle.oracle.url),
    )
    return updated, {"problem_id": bundle.spec.id, "verified": found, "attempts": attempts}


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
