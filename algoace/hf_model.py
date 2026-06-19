from __future__ import annotations

import re

from algoace.model_client import ModelResponse


SYSTEM_PROMPT = (
    "You are AlgoAce, a competitive programming code generator. "
    "Return exactly one complete Python 3 code block and no explanation. "
    "The program must follow the requested input/output contract and fit the resource limits."
)


class HuggingFaceCodeModel:
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        load_in_4bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-train.txt before loading a Hugging Face model.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = count > 1 or self.temperature > 0
        kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": count,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(self.temperature, 0.05)
        with self.torch.no_grad():
            output_ids = self.model.generate(**kwargs)
        responses = []
        prompt_length = inputs.input_ids.shape[-1]
        for item in output_ids:
            raw = self.tokenizer.decode(item[prompt_length:], skip_special_tokens=True)
            responses.append(ModelResponse(raw, extract_python_code(raw)))
        return responses


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python|py)\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""

