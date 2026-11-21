"""Model-specific, opt-in GDK-02 runtime envelopes.

The served model ID is an operator-supplied binding, checked on every request.
All token counts come from the *same backend that performs inference*.
Checkpoint/quantization identity must be established by the operator's server.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_family: str
    hf_reference: str
    default_counter: str
    window_tokens: int = 12_288
    output_reserve_tokens: int = 2_048
    safety_reserve_tokens: int = 2_048

    def input_limit(self, served_context_tokens: int) -> int:
        if isinstance(served_context_tokens, bool) or not isinstance(served_context_tokens, int):
            raise ValueError("served context size must be an integer")
        if served_context_tokens <= self.output_reserve_tokens + self.safety_reserve_tokens:
            raise ValueError("served context is too small for reservations")
        return min(served_context_tokens, self.window_tokens) - (
            self.output_reserve_tokens + self.safety_reserve_tokens
        )


PROFILES: dict[str, ModelProfile] = {
    "inkling-2-mini": ModelProfile(
        name="inkling-2-mini",
        model_family="Thinking Machines Inkling 2 Mini",
        hf_reference="thinkingmachines/Inkling-2-Mini",
        default_counter="vllm_render",
    ),
    "bonsai-8b": ModelProfile(
        name="bonsai-8b",
        model_family="PrismML Bonsai 8B (1-bit GGUF)",
        hf_reference="prism-ml/Bonsai-8B-gguf",
        default_counter="llama_cpp_input_tokens",
    ),
    "bonsai-27b": ModelProfile(
        name="bonsai-27b",
        model_family="PrismML Bonsai 27B (1-bit GGUF)",
        hf_reference="prism-ml/Bonsai-27B-gguf",
        default_counter="llama_cpp_input_tokens",
    ),
}

COUNTERS = frozenset(("vllm_render", "llama_cpp_input_tokens"))


def get_profile(name: str) -> ModelProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown GDK model profile: {name}") from error
