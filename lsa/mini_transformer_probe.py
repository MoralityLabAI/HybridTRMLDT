"""Small causal attention+MLP loops for visit-alignment measurements."""

from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional neural dependency
    raise ImportError("lsa.mini_transformer_probe requires the optional 'neural' extra") from exc


class MiniAttentionMlpResidualBlock(nn.Module):
    """Manual causal attention and MLP under one explicit outer residual."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        mlp_ratio: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or heads <= 0 or hidden_size % heads:
            raise ValueError("hidden_size must be positive and divisible by heads")
        if mlp_ratio <= 0 or alpha <= 0 or beta < 0:
            raise ValueError("invalid MLP or residual configuration")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.head_size = hidden_size // heads
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attention_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp_up = nn.Linear(hidden_size, mlp_ratio * hidden_size)
        self.mlp_down = nn.Linear(mlp_ratio * hidden_size, hidden_size)

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[-1] != self.hidden_size:
            raise ValueError("mini attention blocks require [batch, sequence, hidden] state")
        batch, length, _ = state.shape
        normalized = self.input_norm(state)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_size
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=state.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden_size)
        attended = self.attention_out(attended)
        branch_state = normalized + attended
        branch = attended + self.mlp_down(
            torch.nn.functional.gelu(self.mlp_up(self.mlp_norm(branch_state)))
        )
        return self.alpha * state + self.beta * branch


class MiniTransformerLoop(nn.Module):
    """Finite tied or visit-untied schedule of mini Transformer blocks."""

    def __init__(
        self,
        hidden_size: int,
        rounds: int,
        *,
        tied: bool,
        heads: int,
        mlp_ratio: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        self.hidden_size = int(hidden_size)
        self.rounds = int(rounds)
        self.tied = bool(tied)
        count = 1 if tied else rounds
        self.blocks = nn.ModuleList(
            MiniAttentionMlpResidualBlock(
                hidden_size,
                heads=heads,
                mlp_ratio=mlp_ratio,
                alpha=alpha,
                beta=beta,
            )
            for _ in range(count)
        )

    @property
    def alpha(self) -> float:
        return self.blocks[0].alpha

    @property
    def beta(self) -> float:
        return self.blocks[0].beta

    def block_at(self, visit: int) -> MiniAttentionMlpResidualBlock:
        if visit not in range(self.rounds):
            raise IndexError(visit)
        return self.blocks[0 if self.tied else visit]

    def forward(self, state: Tensor) -> Tensor:
        for visit in range(self.rounds):
            state = self.block_at(visit)(state)
        return state

    def states(self, state: Tensor) -> tuple[Tensor, ...]:
        result = [state]
        for visit in range(self.rounds):
            state = self.block_at(visit)(state)
            result.append(state)
        return tuple(result)

    def suffix(self, state: Tensor, after_visit: int) -> Tensor:
        for visit in range(after_visit + 1, self.rounds):
            state = self.block_at(visit)(state)
        return state
