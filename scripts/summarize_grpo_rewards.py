from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AlgoAce GRPO reward logs.")
    parser.add_argument("--reward-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.reward_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(rendered)


def summarize(rows: list[dict]) -> dict:
    rewards = [float(row.get("total") or 0.0) for row in rows]
    code_lengths = [int(row.get("code_chars") or 0) for row in rows]
    reasons: Counter[str] = Counter()
    groups = defaultdict(list)
    hashes = set()
    for row in rows:
        reasons.update(map(str, row.get("reasons") or []))
        groups[(row.get("reward_call"), row.get("problem_id"))].append(
            float(row.get("total") or 0.0)
        )
        if row.get("code_hash"):
            hashes.add(str(row["code_hash"]))
    varying_groups = sum(
        max(values) - min(values) > 1e-9 for values in groups.values() if values
    )
    return {
        "stage": "summarize_grpo_rewards",
        "completions": len(rows),
        "groups": len(groups),
        "varying_reward_groups": varying_groups,
        "zero_variance_group_rate": (
            1.0 - varying_groups / len(groups) if groups else 0.0
        ),
        "reward": _stats(rewards),
        "fully_correct_rate": (
            sum(value >= 0.999 for value in rewards) / len(rewards) if rewards else 0.0
        ),
        "reason_counts": dict(reasons),
        "unique_code_hash_rate": len(hashes) / len(rows) if rows else 0.0,
        "code_chars": _stats(code_lengths),
    }


def _stats(values: list[float | int]) -> dict[str, float]:
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
