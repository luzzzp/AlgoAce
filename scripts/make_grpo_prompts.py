from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.context import CODE_OUTPUT_RULE
from algoace.hf_model import SYSTEM_PROMPT
from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GRPO prompts linked to verified problem ids.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-reward-tests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nontrain-split", action="store_true")
    args = parser.parse_args()

    _assert_training_split(Path(args.problems), args.allow_nontrain_split)
    all_bundles = [
        bundle for bundle in load_problems(args.problems) if bundle.oracle.best_verified()
    ]
    bundles = [
        bundle
        for bundle in all_bundles
        if len(bundle.tests.reward_tests) >= args.min_reward_tests
    ]
    skipped_missing_reward = len(all_bundles) - len(bundles)
    random.Random(args.seed).shuffle(bundles)
    if args.limit:
        bundles = bundles[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_prompt_ids(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    written = 0
    with out_path.open(mode, encoding="utf-8") as handle:
        for bundle in bundles:
            if bundle.spec.id in done:
                continue
            handle.write(
                json.dumps(
                    _record(bundle),
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            written += 1
    print(json.dumps({
        "stage": "make_grpo_prompts",
        "records": _count_lines(out_path),
        "new_records": written,
        "skipped_insufficient_reward_tests": skipped_missing_reward,
        "min_reward_tests": args.min_reward_tests,
        "seed": args.seed,
        "out": str(out_path),
    }, indent=2))


def _record(bundle) -> dict:
    return {
        "problem_id": bundle.spec.id,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{bundle.spec.prompt(bundle.tests.visible_tests)}"
                    f"\n\n{CODE_OUTPUT_RULE}"
                ),
            },
        ],
    }


def _load_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("problem_id"):
                ids.add(str(payload["problem_id"]))
    return ids


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _assert_training_split(path: Path, allow_nontrain: bool) -> None:
    metadata_path = path / "_split_metadata.json"
    if allow_nontrain or not metadata_path.exists():
        return
    role = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("role") or "")
    if role != "train":
        raise SystemExit(
            f"Refusing to build GRPO prompts from split role={role!r}. "
            "Use the train split or explicitly pass --allow-nontrain-split for diagnostics."
        )


if __name__ == "__main__":
    main()
