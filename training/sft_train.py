from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algoace.hf_model import SYSTEM_PROMPT
from algoace.executor import has_syntax_warning
from algoace.hf_model import extract_python_code


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for AlgoAce code-only data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--min-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=100_000)
    parser.add_argument("--max-completion-chars", type=int, default=50_000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs for a short smoke run when greater than zero.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before SFT.") from exc

    if args.min_prompt_tokens < 1 or args.min_prompt_tokens >= args.max_seq_length:
        raise SystemExit("--min-prompt-tokens must be between 1 and max-seq-length - 1.")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset("json", data_files=args.dataset, split="train")
    initial_records = len(raw_dataset)
    dataset = raw_dataset.filter(
        lambda row: not _row_has_syntax_warning(row),
        desc="Drop Python targets with SyntaxWarning",
    )
    dropped_syntax_warnings = initial_records - len(dataset)
    before_character_filter = len(dataset)
    dataset = dataset.filter(
        lambda row: _within_char_limits(
            row,
            args.max_prompt_chars,
            args.max_completion_chars,
        ),
        desc="Drop pathological character-length outliers",
    )
    dropped_character_outliers = before_character_filter - len(dataset)
    max_completion_tokens = args.max_seq_length - args.min_prompt_tokens
    before_token_filter = len(dataset)
    dataset = dataset.filter(
        lambda row: _completion_token_count(row, tokenizer) <= max_completion_tokens,
        desc="Drop targets that cannot preserve the minimum prompt budget",
    )
    dropped_overlong_targets = before_token_filter - len(dataset)
    if len(dataset) == 0:
        raise SystemExit("No SFT records remain after target-length filtering.")
    dataset = dataset.map(
        lambda row: _tokenize_record(row, tokenizer, args.max_seq_length),
        remove_columns=dataset.column_names,
        desc="Tokenize prompts while masking non-code tokens",
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        seed=args.seed,
        bf16=bf16,
        fp16=bool(torch.cuda.is_available() and not bf16),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    print(
        json.dumps(
            {
                "stage": "sft_preflight",
                "input_records": initial_records,
                "training_records": len(dataset),
                "dropped_syntax_warnings": dropped_syntax_warnings,
                "dropped_character_outliers": dropped_character_outliers,
                "dropped_overlong_targets": dropped_overlong_targets,
                "max_seq_length": args.max_seq_length,
                "min_prompt_tokens": args.min_prompt_tokens,
                "max_prompt_chars": args.max_prompt_chars,
                "max_completion_chars": args.max_completion_chars,
                "max_steps": args.max_steps,
                "loss_scope": "assistant_code_only",
            },
            indent=2,
        )
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def _completion_token_count(row: dict[str, str], tokenizer) -> int:
    return len(_completion_ids(row, tokenizer))


def _within_char_limits(
    row: dict[str, str],
    max_prompt_chars: int,
    max_completion_chars: int,
) -> bool:
    prompt_ok = max_prompt_chars <= 0 or len(str(row.get("input") or "")) <= max_prompt_chars
    completion_ok = (
        max_completion_chars <= 0
        or len(str(row.get("output") or "")) <= max_completion_chars
    )
    return prompt_ok and completion_ok


def _row_has_syntax_warning(row: dict[str, str]) -> bool:
    code = extract_python_code(str(row.get("output") or ""))
    return bool(code and has_syntax_warning(code))


def _tokenize_record(
    row: dict[str, str],
    tokenizer,
    max_length: int,
) -> dict[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{row['instruction']}\n\n{row['input']}",
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return _pack_token_ids(prompt_ids, _completion_ids(row, tokenizer), max_length)


def _completion_ids(row: dict[str, str], tokenizer) -> list[int]:
    ids = tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = [*ids, tokenizer.eos_token_id]
    return ids


def _pack_token_ids(
    prompt_ids: list[int],
    completion_ids: list[int],
    max_length: int,
) -> dict[str, list[int]]:
    if not completion_ids:
        raise ValueError("Completion must contain at least one token.")
    if len(completion_ids) >= max_length:
        raise ValueError("Completion does not fit in max_length.")
    prompt_budget = max_length - len(completion_ids)
    if len(prompt_ids) > prompt_budget:
        head = max(1, int(prompt_budget * 0.75))
        tail = prompt_budget - head
        prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
    input_ids = [*prompt_ids, *completion_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + list(completion_ids),
    }


if __name__ == "__main__":
    main()
