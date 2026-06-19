from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.schema import (
    OracleMetadata,
    OracleSolution,
    ProblemBundle,
    ProblemSpec,
    TestCase,
    TestSuite,
    save_problem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TACO-style data into AlgoAce problem JSON.")
    parser.add_argument("--dataset", default="likaixin/TACO-verified")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-visible-tests", type=int, default=3)
    parser.add_argument("--max-reward-tests", type=int, default=20)
    parser.add_argument("--max-eval-tests", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before converting TACO.") from exc

    dataset = load_dataset(args.dataset, split=args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = out_dir / "_skipped.jsonl"
    done_path = out_dir / "_convert_done.jsonl"
    done_indices = _load_done_indices(done_path) if args.resume else set()
    new_written = 0
    skipped_existing = 0
    skipped = Counter()

    mode = "a" if args.resume else "w"
    with skipped_path.open(mode, encoding="utf-8") as failure_log, done_path.open(mode, encoding="utf-8") as done_log:
        for index, row in enumerate(dataset):
            if args.limit and index >= args.limit:
                break
            if index in done_indices:
                skipped_existing += 1
                continue
            try:
                bundle = convert_row(
                    row,
                    index,
                    args.max_visible_tests,
                    args.max_reward_tests,
                    args.max_eval_tests,
                )
                save_problem(bundle, out_dir / f"{bundle.spec.id}.json")
                new_written += 1
                status = "written"
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                reason = str(exc)
                skipped[reason] += 1
                status = "skipped"
                failure_log.write(
                    json.dumps(
                        {
                            "index": index,
                            "reason": reason,
                            "name": row.get("name") or row.get("title") or "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done_log.write(json.dumps({"index": index, "status": status}) + "\n")
            failure_log.flush()
            done_log.flush()

    written_total = len([path for path in out_dir.glob("*.json") if not path.name.startswith("_")])
    report = {
        "stage": "convert_taco",
        "dataset": args.dataset,
        "split": args.split,
        "written": written_total,
        "new_written": new_written,
        "skipped_existing": skipped_existing,
        "skipped": sum(skipped.values()),
        "skip_reasons": dict(skipped),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def convert_row(
    row: dict[str, Any],
    index: int,
    max_visible: int,
    max_reward: int,
    max_eval: int,
) -> ProblemBundle:
    statement = str(row.get("question") or row.get("statement") or "").strip()
    if not statement:
        raise ValueError("missing statement")
    io_payload = _jsonish(row.get("input_output") or {})
    inputs = list(io_payload.get("inputs") or [])
    outputs = list(io_payload.get("outputs") or [])
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("invalid input_output pairs")
    entry_point = str(io_payload.get("fn_name") or io_payload.get("entry_point") or "")
    io_mode = "callable" if entry_point else "stdin"
    cases = [
        TestCase(
            stdin=_input_text(item, io_mode),
            expected_stdout=_output_text(expected, io_mode),
        )
        for item, expected in zip(inputs, outputs)
    ]
    visible, reward, eval_tests = _split_cases(cases, max_visible, max_reward, max_eval)
    solutions = [
        OracleSolution("python3", code, False)
        for code in _extract_solutions(row.get("solutions") or row.get("python_solutions") or [])
    ]
    if not solutions:
        raise ValueError("missing Python solutions")
    source_name = str(row.get("name") or row.get("title") or f"problem_{index}")
    problem_id = _safe_id(index, source_name)
    return ProblemBundle(
        spec=ProblemSpec(
            id=problem_id,
            statement=statement,
            input_format=str(row.get("input_format") or ""),
            output_format=str(row.get("output_format") or ""),
            constraints=_string_list(row.get("constraints") or []),
            time_limit_sec=_parse_number(row.get("time_limit"), 2.0),
            memory_limit_mb=int(_parse_number(row.get("memory_limit"), 256.0)),
            io_mode=io_mode,
            entry_point=entry_point,
        ),
        tests=TestSuite(visible, reward, eval_tests),
        oracle=OracleMetadata(
            solutions=solutions,
            source=str(row.get("source") or "TACO-verified"),
            url=str(row.get("url") or ""),
        ),
    )


def _split_cases(
    cases: list[TestCase],
    max_visible: int,
    max_reward: int,
    max_eval: int,
) -> tuple[list[TestCase], list[TestCase], list[TestCase]]:
    visible_count = min(max_visible, len(cases))
    visible = _numbered(cases[:visible_count], "visible")
    cursor = visible_count
    reward = _numbered(cases[cursor : cursor + max_reward], "reward")
    cursor += len(reward)
    eval_tests = _numbered(cases[cursor : cursor + max_eval], "eval")
    return visible, reward, eval_tests


def _numbered(cases: list[TestCase], prefix: str) -> list[TestCase]:
    return [
        TestCase(case.stdin, case.expected_stdout, f"{prefix}-{index:04d}", case.timeout_sec)
        for index, case in enumerate(cases, start=1)
    ]


def _extract_solutions(value: Any) -> list[str]:
    value = _jsonish(value)
    if isinstance(value, dict):
        value = value.get("python") or value.get("solutions") or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _input_text(value: Any, io_mode: str) -> str:
    if io_mode == "callable":
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) + "\n"
    return str(value).rstrip() + "\n"


def _output_text(value: Any, io_mode: str) -> str:
    if io_mode == "callable":
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _string_list(value: Any) -> list[str]:
    value = _jsonish(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


def _parse_number(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def _safe_id(index: int, name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()[:80]
    return f"taco_{index}_{slug or 'problem'}"


def _load_done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("index"), int):
                indices.add(payload["index"])
    return indices


if __name__ == "__main__":
    main()
