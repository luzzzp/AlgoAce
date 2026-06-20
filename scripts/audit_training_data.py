from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.schema import load_problems
from algoace.executor import has_syntax_warning


CODE_BLOCK = re.compile(r"^```python\s*\n(.*)\n```\s*$", re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit verified train problems and code-only SFT JSONL before training."
    )
    parser.add_argument("--problems", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = audit(Path(args.problems), Path(args.dataset))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    if args.strict and report["failures"]:
        raise SystemExit(1)


def audit(problems_path: Path, dataset_path: Path) -> dict:
    bundles = load_problems(problems_path)
    problem_ids = [bundle.spec.id for bundle in bundles]
    statements = [bundle.spec.statement.strip() for bundle in bundles]
    oracle_codes = [bundle.oracle.best_verified().strip() for bundle in bundles]
    io_modes = Counter(bundle.spec.io_mode for bundle in bundles)
    problem_failures: list[str] = []

    if len(problem_ids) != len(set(problem_ids)):
        problem_failures.append("duplicate_problem_ids")
    unverified = sum(not code for code in oracle_codes)
    if unverified:
        problem_failures.append("unverified_train_problems")
    invalid_oracle_syntax = sum(
        bool(code) and not _syntax_valid(code) for code in oracle_codes
    )
    oracle_syntax_warnings = sum(
        bool(code) and has_syntax_warning(code) for code in oracle_codes
    )
    if invalid_oracle_syntax:
        problem_failures.append("invalid_oracle_syntax")

    dataset = _audit_jsonl(dataset_path)
    failures = [*problem_failures]
    if dataset["malformed_json_records"]:
        failures.append("malformed_sft_json")
    if dataset["invalid_schema_records"]:
        failures.append("invalid_sft_schema")
    if dataset["valid_code_block_records"] != dataset["records"]:
        failures.append("invalid_code_block_format")
    if dataset["syntax_valid_records"] != dataset["records"]:
        failures.append("invalid_sft_python_syntax")
    if dataset["records"] != len(bundles):
        failures.append("problem_dataset_count_mismatch")

    metadata_path = problems_path / "_split_metadata.json"
    split_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return {
        "stage": "audit_training_data",
        "problems": {
            "path": str(problems_path.resolve()),
            "split_role": split_metadata.get("role", ""),
            "records": len(bundles),
            "verified_records": len(bundles) - unverified,
            "missing_visible_tests": sum(not item.tests.visible_tests for item in bundles),
            "missing_reward_tests": sum(not item.tests.reward_tests for item in bundles),
            "missing_eval_tests": sum(not item.tests.eval_tests for item in bundles),
            "invalid_oracle_syntax": invalid_oracle_syntax,
            "syntax_warning_records": oracle_syntax_warnings,
            "duplicate_statements": _duplicate_count(statements),
            "duplicate_oracle_codes": _duplicate_count(oracle_codes),
            "io_modes": dict(sorted(io_modes.items())),
            "fingerprint": _fingerprint(problem_ids),
        },
        "dataset": dataset,
        "failures": sorted(set(failures)),
    }


def _audit_jsonl(path: Path) -> dict:
    records = 0
    malformed = 0
    invalid_schema = 0
    valid_blocks = 0
    syntax_valid = 0
    syntax_warnings = 0
    inputs: list[str] = []
    outputs: list[str] = []
    input_lengths: list[int] = []
    code_lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key].strip()
                for key in ("instruction", "input", "output")
            ):
                invalid_schema += 1
                continue
            inputs.append(row["input"])
            outputs.append(row["output"])
            input_lengths.append(len(row["input"]))
            match = CODE_BLOCK.fullmatch(row["output"].strip())
            if not match:
                continue
            valid_blocks += 1
            code = match.group(1)
            code_lengths.append(len(code))
            syntax_valid += int(_syntax_valid(code))
            syntax_warnings += int(has_syntax_warning(code))
    return {
        "path": str(path.resolve()),
        "records": records,
        "malformed_json_records": malformed,
        "invalid_schema_records": invalid_schema,
        "valid_code_block_records": valid_blocks,
        "syntax_valid_records": syntax_valid,
        "syntax_warning_records": syntax_warnings,
        "duplicate_inputs": _duplicate_count(inputs),
        "duplicate_outputs": _duplicate_count(outputs),
        "input_chars": _length_stats(input_lengths),
        "code_chars": _length_stats(code_lengths),
        "fingerprint": _fingerprint(outputs),
    }


def _syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _duplicate_count(values: list[str]) -> int:
    normalized = [value for value in values if value]
    return len(normalized) - len(set(normalized))


def _length_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _fingerprint(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
