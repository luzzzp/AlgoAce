from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.hf_model import extract_python_code
from algoace.reward import score_completion
from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with executable AlgoAce rewards.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reward-log", default="")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--reward-workers", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        import torch
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before GRPO.") from exc

    if args.num_generations < 2:
        raise SystemExit("--num-generations must be at least 2 for group-relative training.")
    if args.reward_workers < 1:
        raise SystemExit("--reward-workers must be at least 1.")

    set_seed(args.seed)
    bundles = {bundle.spec.id: bundle for bundle in load_problems(args.problems)}
    dataset = load_dataset("json", data_files=args.prompts, split="train")
    missing_ids = sorted(set(map(str, dataset["problem_id"])) - set(bundles))
    if missing_ids:
        raise SystemExit(f"GRPO prompts reference missing problems: {missing_ids[:5]}")
    reward_log = Path(args.reward_log) if args.reward_log else None
    if reward_log:
        reward_log.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume_from_checkpoint:
            reward_log.write_text("", encoding="utf-8")
    reward_call_index = 0

    def executable_reward(completions, problem_id, **kwargs):
        nonlocal reward_call_index
        reward_call_index += 1
        items = [
            (_completion_text(completion), str(item_id))
            for completion, item_id in zip(completions, problem_id)
        ]
        with ThreadPoolExecutor(max_workers=args.reward_workers) as pool:
            scored = list(
                pool.map(
                    lambda item: score_completion(item[0], bundles[item[1]]),
                    items,
                )
            )
        rewards = [item.total for item in scored]
        log_rows = []
        for completion_index, ((text, item_id), breakdown) in enumerate(zip(items, scored)):
            if reward_log:
                code = extract_python_code(text)
                log_rows.append(
                    {
                        "reward_call": reward_call_index,
                        "completion_index": completion_index,
                        "problem_id": item_id,
                        **asdict(breakdown),
                        "code_chars": len(code),
                        "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest() if code else "",
                    }
                )
        if reward_log and log_rows:
            with reward_log.open("a", encoding="utf-8") as handle:
                for row in log_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        return rewards

    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "save_total_limit": 5,
        "seed": args.seed,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "mask_truncated_completions": True,
        "model_init_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    }
    config = GRPOConfig(**_supported_kwargs(GRPOConfig, config_kwargs))
    if args.adapter:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[executable_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    print(
        json.dumps(
            {
                "stage": "grpo_preflight",
                "prompts": len(dataset),
                "num_generations": args.num_generations,
                "per_device_train_batch_size": args.num_generations,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "reward_workers": args.reward_workers,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_steps": args.max_steps,
                "reward_log": str(reward_log or ""),
            },
            indent=2,
        )
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


def _supported_kwargs(cls, values: dict) -> dict:
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in completion
        )
    return str(completion)


if __name__ == "__main__":
    main()
