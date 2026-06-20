from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two AlgoAce evaluation reports problem by problem.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--problems", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    io_modes = {}
    if args.problems:
        io_modes = {bundle.spec.id: bundle.spec.io_mode for bundle in load_problems(args.problems)}
    report = compare_reports(
        _load(Path(args.base)),
        _load(Path(args.candidate)),
        io_modes,
        args.max_examples,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "regression_examples"}, ensure_ascii=False, indent=2))


def compare_reports(
    base_report: dict,
    candidate_report: dict,
    io_modes: dict[str, str] | None = None,
    max_examples: int = 10,
) -> dict:
    io_modes = io_modes or {}
    base = {str(item["problem_id"]): item for item in base_report.get("results", [])}
    candidate = {str(item["problem_id"]): item for item in candidate_report.get("results", [])}
    shared_ids = sorted(set(base) & set(candidate))
    missing = {
        "base_only": sorted(set(base) - set(candidate)),
        "candidate_only": sorted(set(candidate) - set(base)),
    }
    transitions = Counter()
    failure_reasons = {"base": Counter(), "candidate": Counter()}
    lengths = {"base": [], "candidate": []}
    by_io = defaultdict(lambda: {"problems": 0, "base_solved": 0, "candidate_solved": 0})
    regressions = []

    for problem_id in shared_ids:
        base_item = base[problem_id]
        candidate_item = candidate[problem_id]
        base_solved = _solved(base_item)
        candidate_solved = _solved(candidate_item)
        transition = (
            "both_solved" if base_solved and candidate_solved else
            "base_only_solved" if base_solved else
            "candidate_only_solved" if candidate_solved else
            "neither_solved"
        )
        transitions[transition] += 1
        failure_reasons["base"][str(base_item.get("failure_reason") or "SOLVED")] += 1
        failure_reasons["candidate"][str(candidate_item.get("failure_reason") or "SOLVED")] += 1
        base_code = _selected_code(base_item)
        candidate_code = _selected_code(candidate_item)
        lengths["base"].append(len(base_code))
        lengths["candidate"].append(len(candidate_code))
        mode = io_modes.get(problem_id, "unknown")
        by_io[mode]["problems"] += 1
        by_io[mode]["base_solved"] += int(base_solved)
        by_io[mode]["candidate_solved"] += int(candidate_solved)
        if base_solved and not candidate_solved and len(regressions) < max_examples:
            regressions.append(
                {
                    "problem_id": problem_id,
                    "io_mode": mode,
                    "candidate_failure_reason": candidate_item.get("failure_reason"),
                    "base_code_chars": len(base_code),
                    "candidate_code_chars": len(candidate_code),
                    "candidate_visible_pass_rate": _selected_visible_pass_rate(candidate_item),
                    "base_code_preview": base_code[:500],
                    "candidate_code_preview": candidate_code[:500],
                }
            )

    io_summary = {}
    for mode, values in sorted(by_io.items()):
        total = values["problems"]
        io_summary[mode] = {
            **values,
            "base_verified_success_rate": values["base_solved"] / total if total else 0.0,
            "candidate_verified_success_rate": values["candidate_solved"] / total if total else 0.0,
        }
    return {
        "stage": "compare_eval_reports",
        "shared_problems": len(shared_ids),
        "missing_problems": missing,
        "base_metrics": base_report.get("metrics", {}),
        "candidate_metrics": candidate_report.get("metrics", {}),
        "transitions": dict(transitions),
        "failure_reasons": {
            name: dict(counter) for name, counter in failure_reasons.items()
        },
        "selected_code_chars": {
            name: _length_stats(values) for name, values in lengths.items()
        },
        "by_io_mode": io_summary,
        "regression_examples": regressions,
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _solved(item: dict) -> bool:
    return item.get("status") == "VERIFIED_SOLVED"


def _selected_code(item: dict) -> str:
    records = item.get("attempt_records") or []
    if not records:
        return ""
    record = records[-1]
    candidates = record.get("candidates") or []
    index = int(record.get("selected_index") or 0)
    if not candidates or index >= len(candidates):
        return ""
    return str(candidates[index].get("code") or "")


def _selected_visible_pass_rate(item: dict) -> float:
    records = item.get("attempt_records") or []
    if not records:
        return 0.0
    record = records[-1]
    candidates = record.get("candidates") or []
    index = int(record.get("selected_index") or 0)
    if not candidates or index >= len(candidates):
        return 0.0
    report = candidates[index].get("visible_result") or {}
    runs = report.get("runs") or []
    if not runs:
        return 1.0 if report.get("syntax_valid") else 0.0
    return sum(bool(run.get("passed")) for run in runs) / len(runs)


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"median": 0, "mean": 0, "max": 0}
    return {
        "median": statistics.median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
