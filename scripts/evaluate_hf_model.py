from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.hf_model import HuggingFaceCodeModel
from algoace.schema import SolveStatus, load_problems, problem_to_dict, result_to_dict
from algoace.solver import AlgoAceSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face code model with AlgoAce.")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-repair-turns", type=int, default=0)
    parser.add_argument("--candidates-per-turn", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward-rerank", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundles = load_problems(args.problems)
    if args.limit:
        bundles = bundles[: args.limit]
    split_metadata = _load_split_metadata(Path(args.problems))
    if split_metadata and split_metadata.get("role") != args.split_name:
        raise SystemExit(
            f"Requested split-name={args.split_name!r} does not match dataset role="
            f"{split_metadata.get('role')!r}."
        )
    run_identity = {
        "model": args.model,
        "adapter": args.adapter,
        "problems": str(Path(args.problems).resolve()),
        "split_name": args.split_name,
        "dataset_fingerprint": dataset_fingerprint(bundles),
        "split_metadata": split_metadata,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward_rerank": args.reward_rerank,
        "max_repair_turns": args.max_repair_turns,
        "candidates_per_turn": args.candidates_per_turn,
    }
    _set_seed(args.seed)
    model = HuggingFaceCodeModel(
        args.model,
        adapter_path=args.adapter,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.load_in_4bit,
    )
    solver = AlgoAceSolver(
        model,
        max_repair_turns=args.max_repair_turns,
        candidates_per_turn=args.candidates_per_turn,
        rerank_with_reward_tests=args.reward_rerank,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial.jsonl")
    partial_meta_path = out_path.with_suffix(out_path.suffix + ".partial.meta.json")
    if args.resume:
        _validate_resume_identity(partial_meta_path, run_identity)
    else:
        partial_meta_path.write_text(
            json.dumps(run_identity, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    completed = _load_partial(partial_path) if args.resume else {}
    if not args.resume:
        partial_path.write_text("", encoding="utf-8")
    results_by_id = dict(completed)
    for index, bundle in enumerate(bundles, start=1):
        if bundle.spec.id in completed:
            print(f"[{index}/{len(bundles)}] {bundle.spec.id} resumed")
            continue
        _set_seed(problem_seed(args.seed, bundle.spec.id))
        result = solver.solve(bundle.spec, bundle.tests)
        payload = result_to_dict(result)
        results_by_id[bundle.spec.id] = payload
        with partial_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(payload, ensure_ascii=False) + "\n")
            checkpoint.flush()
        print(
            f"[{index}/{len(bundles)}] {bundle.spec.id} "
            f"status={result.status.value} attempts={result.attempts}"
        )

    results = [results_by_id[bundle.spec.id] for bundle in bundles if bundle.spec.id in results_by_id]
    report = {
        "config": run_identity,
        "metrics": metrics(results),
        "results": results,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    initial_visible = []
    initial_verified = []
    final_visible = []
    verified = []
    visible_rates = []
    reward_rates = []
    eval_rates = []
    pass_at_k = []
    syntax_errors = 0
    runtime_errors = 0
    timeouts = 0

    for result in results:
        records = result["attempt_records"]
        first = records[0] if records else None
        last = records[-1] if records else None
        first_selected = _selected(first)
        last_selected = _selected(last)
        initial_visible.append(bool(first_selected and first_selected["visible_result"]["syntax_valid"] and _all_runs_pass(first_selected["visible_result"])))
        initial_verified.append(bool(first and first.get("eval_result") and _report_passed(first["eval_result"])))
        pass_at_k.append(bool(first and any(_report_passed(item["visible_result"]) for item in first["candidates"])))
        final_visible.append(bool(last_selected and _report_passed(last_selected["visible_result"])))
        verified.append(result["status"] == SolveStatus.VERIFIED_SOLVED.value)
        if last_selected:
            visible_rates.append(_pass_rate(last_selected["visible_result"]))
            syntax_errors += int(not last_selected["visible_result"]["syntax_valid"])
            runtime_errors += int(
                any(
                    (run.get("returncode") or 0) != 0 and not run.get("timed_out")
                    for run in last_selected["visible_result"]["runs"]
                )
            )
            timeouts += int(any(run.get("timed_out") for run in last_selected["visible_result"]["runs"]))
        else:
            visible_rates.append(0.0)
            syntax_errors += 1
        reward_rates.append(_pass_rate(last.get("reward_result")) if last and last.get("reward_result") else 0.0)
        eval_rates.append(_pass_rate(last.get("eval_result")) if last and last.get("eval_result") else 0.0)

    initial_verified_rate = _mean(initial_verified)
    verified_rate = _mean(verified)
    return {
        "num_problems": len(results),
        "initial_visible_success_rate": _mean(initial_visible),
        "initial_candidate_visible_pass_at_k": _mean(pass_at_k),
        "initial_verified_success_rate": initial_verified_rate,
        "final_visible_success_rate": _mean(final_visible),
        "verified_success_rate": verified_rate,
        "visible_test_pass_rate": _mean(visible_rates),
        "reward_test_pass_rate": _mean(reward_rates),
        "eval_test_pass_rate": _mean(eval_rates),
        "avg_attempts": sum(item["attempts"] for item in results) / len(results),
        "syntax_error_rate": syntax_errors / len(results),
        "runtime_error_rate": runtime_errors / len(results),
        "timeout_rate": timeouts / len(results),
        "repair_gain": verified_rate - initial_verified_rate,
    }


def _selected(record: dict | None) -> dict | None:
    if not record or not record["candidates"]:
        return None
    return record["candidates"][record["selected_index"]]


def _report_passed(report: dict) -> bool:
    return bool(report["syntax_valid"] and _all_runs_pass(report))


def _all_runs_pass(report: dict) -> bool:
    return all(run["passed"] for run in report["runs"])


def _pass_rate(report: dict | None) -> float:
    if not report:
        return 0.0
    runs = report["runs"]
    if not runs:
        return 1.0 if report["syntax_valid"] else 0.0
    return sum(run["passed"] for run in runs) / len(runs)


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def problem_seed(base_seed: int, problem_id: str) -> int:
    digest = hashlib.sha256(problem_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def dataset_fingerprint(bundles) -> str:
    records = [problem_to_dict(bundle) for bundle in sorted(bundles, key=lambda item: item.spec.id)]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_partial(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("problem_id"):
                results[str(payload["problem_id"])] = payload
    return results


def _validate_resume_identity(path: Path, expected: dict) -> None:
    if not path.exists():
        raise SystemExit(f"Cannot resume without run identity file: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        mismatches = {
            key: {"existing": actual.get(key), "requested": expected.get(key)}
            for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)
        }
        raise SystemExit(f"Resume configuration mismatch: {json.dumps(mismatches, ensure_ascii=False)}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_split_metadata(path: Path) -> dict:
    metadata_path = path / "_split_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
