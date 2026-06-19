from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.schema import ProblemBundle, load_problems


INSTRUCTION = (
    "Solve the algorithm problem in Python 3. "
    "Return exactly one complete python code block and no explanation."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build code-only SFT data from verified Python solutions.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nontrain-split", action="store_true")
    args = parser.parse_args()

    _assert_training_split(Path(args.problems), args.allow_nontrain_split)
    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(out_path.suffix + ".done.jsonl")
    _validate_resume_files(out_path, done_path, args.resume)
    done = _load_done_ids(done_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    written = 0
    skipped_existing = 0
    skipped_unverified = 0
    with out_path.open(mode, encoding="utf-8") as handle, done_path.open(mode, encoding="utf-8") as done_log:
        for bundle in bundles:
            if bundle.spec.id in done:
                skipped_existing += 1
                continue
            if bundle.oracle.best_verified():
                handle.write(json.dumps(_record(bundle), ensure_ascii=False) + "\n")
                written += 1
            else:
                skipped_unverified += 1
            done_log.write(json.dumps({"problem_id": bundle.spec.id}) + "\n")
            handle.flush()
            done_log.flush()
    total_records = _count_lines(out_path)
    print(json.dumps({
        "stage": "make_code_sft",
        "records": total_records,
        "new_records": written,
        "skipped_existing": skipped_existing,
        "skipped_unverified": skipped_unverified,
        "out": str(out_path),
    }, indent=2))


def _record(bundle: ProblemBundle) -> dict[str, str]:
    code = bundle.oracle.best_verified().strip()
    return {
        "instruction": INSTRUCTION,
        "input": bundle.spec.prompt(bundle.tests.visible_tests),
        "output": f"```python\n{code}\n```",
    }


def _load_done_ids(path: Path) -> set[str]:
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
            f"Refusing to build training data from split role={role!r}. "
            "Use the train split or explicitly pass --allow-nontrain-split for diagnostics."
        )


def _validate_resume_files(out_path: Path, done_path: Path, resume: bool) -> None:
    if resume and out_path.exists() != done_path.exists():
        raise SystemExit(
            "Cannot resume because the dataset and sidecar are inconsistent: "
            f"dataset_exists={out_path.exists()}, sidecar_exists={done_path.exists()}"
        )


if __name__ == "__main__":
    main()
