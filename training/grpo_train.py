from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.reward import score_completion
from algoace.schema import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with executable AlgoAce rewards.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before GRPO.") from exc

    bundles = {bundle.spec.id: bundle for bundle in load_problems(args.problems)}
    dataset = load_dataset("json", data_files=args.prompts, split="train")

    def executable_reward(completions, problem_id, **kwargs):
        rewards = []
        for completion, item_id in zip(completions, problem_id):
            text = _completion_text(completion)
            rewards.append(score_completion(text, bundles[str(item_id)]).total)
        return rewards

    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "num_generations": args.num_generations,
        "logging_steps": 1,
        "save_steps": 50,
    }
    signature = inspect.signature(GRPOConfig)
    if "max_completion_length" in signature.parameters:
        config_kwargs["max_completion_length"] = args.max_completion_length
    config = GRPOConfig(**config_kwargs)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[executable_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(str(item.get("content") or "") if isinstance(item, dict) else str(item) for item in completion)
    return str(completion)


if __name__ == "__main__":
    main()

