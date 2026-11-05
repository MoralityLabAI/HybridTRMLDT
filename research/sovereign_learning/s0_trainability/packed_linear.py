"""Small-tensor PyTorch oracle for frozen rotated ternary projection + LoRA.

The trits are intentionally unpacked int8 buffers, not a packed CUDA
training kernel. No model training or GGUF loading is performed.
"""
from __future__ import annotations
import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("packed_linear requires the optional neural extra (torch)") from exc

from .ptq1_codec import GROUP, decode_matrix


def normalized_fwht(x: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    """Normalized Sylvester Walsh-Hadamard per last-axis block."""
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two >= 2")
    if x.shape[-1] % block_size:
        raise ValueError("input width must be divisible by block_size")
    y = x.reshape(*x.shape[:-1], -1, block_size)
    step = 1
    while step < block_size:
        z = y.reshape(*y.shape[:-1], -1, 2 * step)
        left, right = z[..., :step], z[..., step:]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*y.shape)
        step *= 2
    return (y / math.sqrt(block_size)).reshape_as(x)


class FrozenTernaryLinear(nn.Module):
    """Compute W_rot @ H @ (sign * x), dequantizing one g128 slice at a time.

    The real sign vector comes from GGUF metadata. All-ones signs are for
    synthetic checks only and must not stand in for real checkpoint signs.
    """

    def __init__(self, trits: torch.Tensor, scales: torch.Tensor, *,
                 signs: torch.Tensor | None = None,
                 rotation_block: int | None = None):
        super().__init__()
        if (trits.ndim != 2 or not trits.shape[0] or not trits.shape[1]
                or trits.shape[1] % GROUP):
            raise ValueError("trits must be [out, in], in divisible by 128")
        if not bool(torch.all((trits >= -1) & (trits <= 1))) or not bool(torch.all(trits == trits.round())):
            raise ValueError("trits must be -1, 0, +1")
        expected = (trits.shape[0], trits.shape[1] // GROUP)
        if tuple(scales.shape) != expected:
            raise ValueError(f"scales must be shape {expected}")
        if not bool(torch.all(torch.isfinite(scales))) or not bool(torch.all(scales >= 0)):
            raise ValueError("scales must be finite and nonnegative")
        if (signs is None) != (rotation_block is None):
            raise ValueError("signs and rotation_block must be specified together")
        if signs is not None:
            if tuple(signs.shape) != (trits.shape[1],):
                raise ValueError("one sign per input feature required")
            if not bool(torch.all(signs.abs() == 1)):
                raise ValueError("signs must be +/-1")
            if trits.shape[1] % rotation_block:
                raise ValueError("rotation_block must divide input width")
        self.register_buffer("trits", trits.to(torch.int8).clone())
        self.register_buffer("scales", scales.to(torch.float32).clone())
        self.register_buffer("signs", None if signs is None else signs.to(torch.float32).clone())
        self.rotation_block = rotation_block

    @classmethod
    def from_payload(cls, payload: bytes, *, rows: int, cols: int,
                     fmt: str = "ptq1", signs: torch.Tensor | None = None,
                     rotation_block: int | None = None):
        q, s = decode_matrix(payload, rows=rows, cols=cols, fmt=fmt)
        return cls(torch.tensor(q, dtype=torch.int8), torch.tensor(s, dtype=torch.float32),
                   signs=signs, rotation_block=rotation_block)

    @property
    def in_features(self) -> int:
        return self.trits.shape[1]

    @property
    def out_features(self) -> int:
        return self.trits.shape[0]

    def transformed_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.signs is None:
            return x
        return normalized_fwht(x * self.signs.to(dtype=x.dtype), block_size=self.rotation_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("input width disagrees with frozen projection")
        z = self.transformed_input(x)
        result = None
        for g in range(self.in_features // GROUP):
            lo, hi = g * GROUP, (g + 1) * GROUP
            w = self.trits[:, lo:hi].to(z.dtype) * self.scales[:, g:g+1].to(z.dtype)
            term = F.linear(z[..., lo:hi], w)
            result = term if result is None else result + term
        return result


class LoRAOverFrozen(nn.Module):
    """LoRA in original input basis; only A and B carry gradients."""

    def __init__(self, base: FrozenTernaryLinear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha):
            raise ValueError("invalid rank or alpha")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * (self.alpha / self.rank)
