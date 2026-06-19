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
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    records = [_record(bundle) for bundle in bundles if bundle.oracle.best_verified()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "make_code_sft", "records": len(records), "out": str(out_path)}, indent=2))


def _record(bundle: ProblemBundle) -> dict[str, str]:
    code = bundle.oracle.best_verified().strip()
    return {
        "instruction": INSTRUCTION,
        "input": bundle.spec.prompt(bundle.tests.visible_tests),
        "output": f"```python\n{code}\n```",
    }


if __name__ == "__main__":
    main()

