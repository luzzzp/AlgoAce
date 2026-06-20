from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.reward import RewardBreakdown, score_completion
from algoace.schema import ProblemBundle, load_problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit executable GRPO reward ordering before starting RL."
    )
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    bundles = [
        bundle
        for bundle in load_problems(args.problems)
        if bundle.oracle.best_verified() and bundle.tests.reward_tests
    ]
    if args.limit:
        bundles = bundles[: args.limit]
    report = audit_rewards(bundles)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(rendered)


def audit_rewards(bundles: list[ProblemBundle]) -> dict:
    oracle_scores: list[float] = []
    bad_scores: list[float] = []
    margins: list[float] = []
    oracle_reasons: Counter[str] = Counter()
    bad_reasons: Counter[str] = Counter()
    ordering_violations = []
    for bundle in bundles:
        oracle = score_completion(
            f"```python\n{bundle.oracle.best_verified()}\n```",
            bundle,
        )
        bad = score_completion(_bad_completion(bundle), bundle)
        oracle_scores.append(oracle.total)
        bad_scores.append(bad.total)
        margins.append(oracle.total - bad.total)
        oracle_reasons.update(oracle.reasons)
        bad_reasons.update(bad.reasons)
        if oracle.total <= bad.total:
            ordering_violations.append(
                {
                    "problem_id": bundle.spec.id,
                    "oracle": _breakdown(oracle),
                    "bad": _breakdown(bad),
                }
            )
    return {
        "stage": "audit_grpo_rewards",
        "problems": len(bundles),
        "oracle_reward": _stats(oracle_scores),
        "bad_reward": _stats(bad_scores),
        "oracle_minus_bad_margin": _stats(margins),
        "oracle_reason_counts": dict(oracle_reasons),
        "bad_reason_counts": dict(bad_reasons),
        "ordering_violations": len(ordering_violations),
        "violation_examples": ordering_violations[:10],
    }


def _bad_completion(bundle: ProblemBundle) -> str:
    if bundle.spec.io_mode == "callable" and bundle.spec.entry_point:
        code = f"def {bundle.spec.entry_point}(*args, **kwargs):\n    return None"
    else:
        code = "print(0)"
    return f"```python\n{code}\n```"


def _breakdown(item: RewardBreakdown) -> dict:
    return {
        "total": item.total,
        "syntax": item.syntax,
        "visible": item.visible,
        "reward_tests": item.reward_tests,
        "correctness_bonus": item.correctness_bonus,
        "penalties": item.penalties,
        "reasons": item.reasons,
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
