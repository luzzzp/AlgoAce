from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.hf_model import HuggingFaceCodeModel
from algoace.schema import SolveStatus, load_problems, result_to_dict
from algoace.solver import AlgoAceSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face code model with AlgoAce.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-repair-turns", type=int, default=0)
    parser.add_argument("--candidates-per-turn", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    model = HuggingFaceCodeModel(
        args.model,
        adapter_path=args.adapter,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    solver = AlgoAceSolver(
        model,
        max_repair_turns=args.max_repair_turns,
        candidates_per_turn=args.candidates_per_turn,
    )
    results = []
    for index, bundle in enumerate(bundles, start=1):
        result = solver.solve(bundle.spec, bundle.tests)
        results.append(result_to_dict(result))
        print(
            f"[{index}/{len(bundles)}] {bundle.spec.id} "
            f"status={result.status.value} attempts={result.attempts}"
        )

    report = {
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "max_repair_turns": args.max_repair_turns,
            "candidates_per_turn": args.candidates_per_turn,
        },
        "metrics": metrics(results),
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    initial_visible = []
    initial_verified = []
    final_visible = []
    verified = []
    visible_rates = []
    reward_rates = []
    eval_rates = []
    pass_at_k = []
    syntax_errors = 0
    runtime_errors = 0
    timeouts = 0

    for result in results:
        records = result["attempt_records"]
        first = records[0] if records else None
        last = records[-1] if records else None
        first_selected = _selected(first)
        last_selected = _selected(last)
        initial_visible.append(bool(first_selected and first_selected["visible_result"]["syntax_valid"] and _all_runs_pass(first_selected["visible_result"])))
        initial_verified.append(bool(first and first.get("eval_result") and _report_passed(first["eval_result"])))
        pass_at_k.append(bool(first and any(_report_passed(item["visible_result"]) for item in first["candidates"])))
        final_visible.append(bool(last_selected and _report_passed(last_selected["visible_result"])))
        verified.append(result["status"] == SolveStatus.VERIFIED_SOLVED.value)
        if last_selected:
            visible_rates.append(_pass_rate(last_selected["visible_result"]))
            syntax_errors += int(not last_selected["visible_result"]["syntax_valid"])
            runtime_errors += int(
                any(
                    (run.get("returncode") or 0) != 0 and not run.get("timed_out")
                    for run in last_selected["visible_result"]["runs"]
                )
            )
            timeouts += int(any(run.get("timed_out") for run in last_selected["visible_result"]["runs"]))
        else:
            visible_rates.append(0.0)
            syntax_errors += 1
        reward_rates.append(_pass_rate(last.get("reward_result")) if last and last.get("reward_result") else 0.0)
        eval_rates.append(_pass_rate(last.get("eval_result")) if last and last.get("eval_result") else 0.0)

    initial_verified_rate = _mean(initial_verified)
    verified_rate = _mean(verified)
    return {
        "num_problems": len(results),
        "initial_visible_success_rate": _mean(initial_visible),
        "initial_candidate_visible_pass_at_k": _mean(pass_at_k),
        "initial_verified_success_rate": initial_verified_rate,
        "final_visible_success_rate": _mean(final_visible),
        "verified_success_rate": verified_rate,
        "visible_test_pass_rate": _mean(visible_rates),
        "reward_test_pass_rate": _mean(reward_rates),
        "eval_test_pass_rate": _mean(eval_rates),
        "avg_attempts": sum(item["attempts"] for item in results) / len(results),
        "syntax_error_rate": syntax_errors / len(results),
        "runtime_error_rate": runtime_errors / len(results),
        "timeout_rate": timeouts / len(results),
        "repair_gain": verified_rate - initial_verified_rate,
    }


def _selected(record: dict | None) -> dict | None:
    if not record or not record["candidates"]:
        return None
    return record["candidates"][record["selected_index"]]


def _report_passed(report: dict) -> bool:
    return bool(report["syntax_valid"] and _all_runs_pass(report))


def _all_runs_pass(report: dict) -> bool:
    return all(run["passed"] for run in report["runs"])


def _pass_rate(report: dict | None) -> float:
    if not report:
        return 0.0
    runs = report["runs"]
    if not runs:
        return 1.0 if report["syntax_valid"] else 0.0
    return sum(run["passed"] for run in runs) / len(runs)


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
