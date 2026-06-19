from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.context import CODE_OUTPUT_RULE
from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GRPO prompts linked to verified problem ids.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    bundles = [bundle for bundle in load_problems(args.problems) if bundle.oracle.best_verified()]
    if args.limit:
        bundles = bundles[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            handle.write(
                json.dumps(
                    {
                        "problem_id": bundle.spec.id,
                        "prompt": f"{bundle.spec.prompt(bundle.tests.visible_tests)}\n\n{CODE_OUTPUT_RULE}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"stage": "make_grpo_prompts", "records": len(bundles), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

