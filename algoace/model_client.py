from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    code: str


class ModelClient(Protocol):
    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        ...


class RuleBasedModel:
    """Small deterministic model used by unit tests."""

    def generate_codes(self, prompt: str, count: int = 1) -> list[ModelResponse]:
        wrong = "print(int(input().split()[0]))"
        correct = "a, b = map(int, input().split())\nprint(a + b)"
        code = correct if "Execution feedback" in prompt or "Internal verification failed" in prompt else wrong
        text = f"```python\n{code}\n```"
        return [ModelResponse(text, code) for _ in range(count)]

