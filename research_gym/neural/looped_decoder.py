"""Iso-shape causal decoder whose expanded depth is schedule-driven."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("LoopedDecoderLM requires the optional 'neural' extra") from exc

from .schedule_executor import ScheduleExecutor


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        alpha: float,
        beta: float,
        normalization: str,
    ) -> None:
        super().__init__()
        if normalization not in {"pre", "post"}:
            raise ValueError("decoder block supports pre or post normalization")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalization = normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, bias=False, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

    def forward(self, state: Tensor, *, causal_mask: Tensor) -> Tensor:
        if self.normalization == "pre":
            normalized = self.norm1(state)
            attended = self.attention(
                normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
            )[0]
            state = self.alpha * state + self.beta * attended
            return self.alpha * state + self.beta * self.mlp(self.norm2(state))
        attended = self.attention(
            state, state, state, attn_mask=causal_mask, need_weights=False
        )[0]
        state = self.norm1(self.alpha * state + self.beta * attended)
        return self.norm2(self.alpha * state + self.beta * self.mlp(state))


@dataclass(frozen=True)
class LoopedDecoderOutput:
    logits: Tensor
    auxiliary_logits: tuple[Tensor, ...]
    carry: Tensor
    visit_outputs: tuple[Tensor, ...]


class LoopedDecoderLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        physical_modules: int,
        expanded_visits: int,
        tying: str,
        parameter_visits: frozenset[int],
        retained_state_edges: frozenset[int],
        supervision_points: tuple[int, ...],
        alpha: float,
        beta: float,
        normalization: str,
        carry_policy: str,
    ) -> None:
        super().__init__()
        if expanded_visits <= 0 or physical_modules <= 0:
            raise ValueError("decoder schedule dimensions must be positive")
        if tying == "fully_tied" and physical_modules != 1:
            raise ValueError("fully tied schedules require one physical module")
        if tying == "untied" and physical_modules != expanded_visits:
            raise ValueError("untied schedules require one module per visit")
        if carry_policy not in {"reset", "detached", "persistent"}:
            raise ValueError("unknown carry policy")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.expanded_visits = int(expanded_visits)
        self.parameter_visits = parameter_visits
        self.retained_state_edges = retained_state_edges
        self.supervision_points = supervision_points
        self.carry_policy = carry_policy
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            DecoderBlock(
                hidden_size,
                num_heads,
                alpha=alpha,
                beta=beta,
                normalization=normalization,
            )
            for _ in range(physical_modules)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.executor = ScheduleExecutor()
        if tying == "fully_tied":
            self.module_indices = tuple(0 for _ in range(expanded_visits))
        elif tying == "untied":
            self.module_indices = tuple(range(expanded_visits))
        else:
            self.module_indices = tuple(
                visit % physical_modules for visit in range(expanded_visits)
            )

    @classmethod
    def from_proposal(cls, proposal: Mapping[str, Any]) -> "LoopedDecoderLM":
        mutation = proposal["mutation"]
        model = proposal["model"]
        visits = int(mutation["expanded_visits"])
        gradient_visits = int(mutation["gradient_visible_visits"])
        edge_count = int(mutation["retained_state_edges"])
        parameter_positions = frozenset(range(visits - gradient_visits, visits))
        state_edges = frozenset(range(max(0, visits - 1 - edge_count), visits - 1))
        supervision_count = int(mutation["supervision_points"])
        supervision = (
            (visits // 2 - 1, visits - 1)
            if supervision_count == 2
            else (visits - 1,)
        )
        return cls(
            vocab_size=int(model["vocab_size"]),
            hidden_size=int(model["hidden_size"]),
            num_heads=int(model["num_heads"]),
            physical_modules=int(mutation["physical_modules"]),
            expanded_visits=visits,
            tying=str(mutation["tying"]),
            parameter_visits=parameter_positions,
            retained_state_edges=state_edges,
            supervision_points=supervision,
            alpha=float(mutation["alpha"]),
            beta=float(mutation["beta"]),
            normalization=str(mutation["normalization"]),
            carry_policy=str(mutation["carry"]),
        )

    def parameter_breakdown(self) -> dict[str, int]:
        embedding = self.embedding.weight.numel()
        core = sum(parameter.numel() for block in self.blocks for parameter in block.parameters())
        normalization = sum(parameter.numel() for parameter in self.input_norm.parameters())
        normalization += sum(parameter.numel() for parameter in self.final_norm.parameters())
        return {
            "unique_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "embedding_parameters": embedding,
            "core_parameters": core + normalization,
        }

    def forward(self, tokens: Tensor, *, carry: Tensor | None = None) -> LoopedDecoderOutput:
        state = self.input_norm(self.embedding(tokens))
        if carry is not None and self.carry_policy != "reset":
            state = state + (carry.detach() if self.carry_policy == "detached" else carry)
        sequence = tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        execution = self.executor(
            state,
            modules=self.blocks,
            module_indices=self.module_indices,
            parameter_visits=self.parameter_visits,
            retained_state_edges=self.retained_state_edges,
            module_kwargs={"causal_mask": causal_mask},
        )
        final_state = self.final_norm(execution.output)
        logits = F.linear(final_state, self.embedding.weight)
        auxiliary = tuple(
            F.linear(self.final_norm(execution.visit_outputs[position]), self.embedding.weight)
            for position in self.supervision_points
            if position != self.expanded_visits - 1
        )
        next_carry = execution.output.detach() if self.carry_policy == "detached" else execution.output
        return LoopedDecoderOutput(logits, auxiliary, next_carry, execution.visit_outputs)
