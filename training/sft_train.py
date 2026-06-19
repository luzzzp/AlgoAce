from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.hf_model import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for AlgoAce code-only data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before SFT.") from exc

    from transformers import set_seed

    set_seed(args.seed)
    dataset = load_dataset("json", data_files=args.dataset, split="train").map(
        lambda row: {"text": _format(row)}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    config = _config(
        SFTConfig,
        args.output_dir,
        args.max_seq_length,
        args.epochs,
        args.learning_rate,
        args.seed,
    )
    trainer = SFTTrainer(
        **_trainer_kwargs(
            SFTTrainer,
            model,
            config,
            dataset,
            tokenizer,
            peft_config,
        )
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


def _format(row: dict[str, str]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{row['instruction']}\n\n{row['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['output']}<|im_end|>"
    )


def _config(cls, output_dir: str, max_length: int, epochs: float, learning_rate: float, seed: int):
    signature = inspect.signature(cls)
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "logging_steps": 10,
        "save_steps": 200,
        "seed": seed,
    }
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = max_length
    elif "max_length" in signature.parameters:
        kwargs["max_length"] = max_length
    if "dataset_text_field" in signature.parameters:
        kwargs["dataset_text_field"] = "text"
    return cls(**kwargs)


def _trainer_kwargs(cls, model, config, dataset, tokenizer, peft_config):
    signature = inspect.signature(cls.__init__)
    kwargs = {
        "model": model,
        "args": config,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    return kwargs


if __name__ == "__main__":
    main()
