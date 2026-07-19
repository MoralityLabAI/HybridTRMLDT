"""Context-aware parameter, FLOP, and activation accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RunContext:
    sequence_length: int
    microbatch_size: int
    precision_bytes: int
    gradient_checkpointing: bool
    attention_implementation: str

    def __post_init__(self) -> None:
        if min(self.sequence_length, self.microbatch_size, self.precision_bytes) <= 0:
            raise ValueError("run-context dimensions must be positive")


@dataclass(frozen=True)
class CostSpec:
    unique_parameters: int
    parameter_bytes: int
    attention_cost_model: str
    activation_cost_model: str
    embedding_parameters: int = 0
    core_parameters: int = 0

    def __post_init__(self) -> None:
        if min(
            self.unique_parameters,
            self.parameter_bytes,
            self.embedding_parameters,
            self.core_parameters,
        ) < 0:
            raise ValueError("cost quantities must be non-negative")


@dataclass(frozen=True)
class CostEstimate:
    unique_parameters: int
    embedding_parameters: int
    core_parameters: int
    applied_parameters_per_token: int
    gradient_applied_parameters_per_token: int
    expanded_visits: int
    estimated_flops: int
    estimated_activation_bytes: int
    estimated_parameter_bytes: int


def estimate_decoder_cost(
    *,
    cost: CostSpec,
    context: RunContext,
    module_parameters: int,
    expanded_visits: int,
    gradient_visible_visits: int,
    hidden_size: int,
    mlp_ratio: int = 4,
) -> CostEstimate:
    if min(module_parameters, expanded_visits, hidden_size) <= 0:
        raise ValueError("decoder cost inputs must be positive")
    if gradient_visible_visits < 0 or gradient_visible_visits > expanded_visits:
        raise ValueError("gradient-visible visits must be within the schedule")
    batch_tokens = context.microbatch_size * context.sequence_length
    linear_flops_per_visit = 2 * batch_tokens * (
        4 * hidden_size * hidden_size + 2 * mlp_ratio * hidden_size * hidden_size
    )
    attention_flops_per_visit = (
        4
        * context.microbatch_size
        * context.sequence_length**2
        * hidden_size
    )
    activation_factor = 4 if context.gradient_checkpointing else 10
    activation_bytes = (
        activation_factor
        * batch_tokens
        * hidden_size
        * context.precision_bytes
        * max(1, gradient_visible_visits)
    )
    return CostEstimate(
        unique_parameters=cost.unique_parameters,
        embedding_parameters=cost.embedding_parameters,
        core_parameters=cost.core_parameters,
        applied_parameters_per_token=module_parameters * expanded_visits,
        gradient_applied_parameters_per_token=(
            module_parameters * gradient_visible_visits
        ),
        expanded_visits=expanded_visits,
        estimated_flops=math.ceil(
            expanded_visits * (linear_flops_per_visit + attention_flops_per_visit)
        ),
        estimated_activation_bytes=activation_bytes,
        estimated_parameter_bytes=cost.parameter_bytes,
    )
