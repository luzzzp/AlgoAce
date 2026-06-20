from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically split problem JSON files by problem id.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--allow-missing-test-groups", action="store_true")
    args = parser.parse_args()

    _validate_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)
    source = Path(args.problems)
    out_dir = Path(args.out_dir)
    all_bundles = load_problems(source)
    verified_candidates = (
        all_bundles
        if args.include_unverified
        else [bundle for bundle in all_bundles if bundle.oracle.best_verified()]
    )
    verified_bundles, deduplicated_equivalent = _deduplicate_equivalent_problems(
        verified_candidates
    )
    if args.allow_missing_test_groups:
        benchmark_bundles = verified_bundles
        training_only_bundles = []
    else:
        benchmark_bundles = [
            bundle for bundle in verified_bundles if _has_all_test_groups(bundle)
        ]
        training_only_bundles = [
            bundle for bundle in verified_bundles if not _has_all_test_groups(bundle)
        ]
    if not benchmark_bundles:
        raise SystemExit("No eligible problems found for splitting.")
    ids = sorted(bundle.spec.id for bundle in benchmark_bundles)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    split_ids = split_problem_ids(ids, args.train_ratio, args.dev_ratio)
    split_ids["train"] = sorted(
        [*split_ids["train"], *(bundle.spec.id for bundle in training_only_bundles)]
    )
    source_files = {path.stem: path for path in source.glob("*.json") if not path.name.startswith("_")}

    for split_name, problem_ids in split_ids.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = {path.stem for path in split_dir.glob("*.json") if not path.name.startswith("_")}
        unexpected = existing - set(problem_ids)
        if unexpected:
            raise SystemExit(
                f"{split_dir} contains stale problem files not in this deterministic split: "
                f"{sorted(unexpected)[:5]}"
            )
        for problem_id in problem_ids:
            source_path = source_files.get(problem_id)
            if source_path is None:
                raise SystemExit(f"Missing source JSON for problem id: {problem_id}")
            target = split_dir / source_path.name
            if not target.exists():
                shutil.copy2(source_path, target)
        (split_dir / "_split_metadata.json").write_text(
            json.dumps(
                {
                    "role": split_name,
                    "seed": args.seed,
                    "problem_count": len(problem_ids),
                    "problem_id_fingerprint": fingerprint_problem_ids(problem_ids),
                    "training_only_count": (
                        len(training_only_bundles) if split_name == "train" else 0
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    all_ids = [item for values in split_ids.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Problem id overlap detected across splits.")
    report = {
        "stage": "split_problems",
        "source": str(source.resolve()),
        "seed": args.seed,
        "verified_only": not args.include_unverified,
        "require_all_test_groups_for_dev_test": not args.allow_missing_test_groups,
        "excluded_unverified": len(all_bundles) - len(verified_candidates),
        "deduplicated_equivalent_problems": deduplicated_equivalent,
        "benchmark_eligible": len(benchmark_bundles),
        "training_only_missing_test_groups": len(training_only_bundles),
        "ratios": {
            "train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": args.test_ratio,
        },
        "counts": {name: len(values) for name, values in split_ids.items()},
        "fingerprints": {
            name: fingerprint_problem_ids(values) for name, values in split_ids.items()
        },
        "problem_ids": split_ids,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_split_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "problem_ids"}, indent=2))


def split_problem_ids(
    shuffled_ids: list[str],
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, list[str]]:
    total = len(shuffled_ids)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    return {
        "train": sorted(shuffled_ids[:train_end]),
        "dev": sorted(shuffled_ids[train_end:dev_end]),
        "test": sorted(shuffled_ids[dev_end:]),
    }


def fingerprint_problem_ids(problem_ids: list[str]) -> str:
    payload = "\n".join(sorted(problem_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_ratios(train: float, dev: float, test: float) -> None:
    values = (train, dev, test)
    if any(value < 0 for value in values):
        raise SystemExit("Split ratios must be non-negative.")
    if abs(sum(values) - 1.0) > 1e-9:
        raise SystemExit("Split ratios must sum to 1.0.")


def _has_all_test_groups(bundle) -> bool:
    return bool(
        bundle.tests.visible_tests
        and bundle.tests.reward_tests
        and bundle.tests.eval_tests
    )


def _deduplicate_equivalent_problems(bundles):
    groups = {}
    for bundle in bundles:
        groups.setdefault(_canonical_problem_key(bundle), []).append(bundle)
    selected = [max(group, key=_representative_score) for group in groups.values()]
    return sorted(selected, key=lambda item: item.spec.id), len(bundles) - len(selected)


def _canonical_problem_key(bundle) -> str:
    statement = re.sub(r"\s+", " ", bundle.spec.statement).strip().casefold()
    if not statement:
        return f"id:{bundle.spec.id}"
    callable_signature = (
        bundle.spec.entry_point.strip().casefold()
        if bundle.spec.io_mode == "callable"
        else ""
    )
    return f"{bundle.spec.io_mode}|{callable_signature}|{statement}"


def _representative_score(bundle) -> tuple[int, int, str]:
    test_count = (
        len(bundle.tests.visible_tests)
        + len(bundle.tests.reward_tests)
        + len(bundle.tests.eval_tests)
    )
    return int(_has_all_test_groups(bundle)), test_count, bundle.spec.id


if __name__ == "__main__":
    main()
