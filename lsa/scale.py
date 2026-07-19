"""Iso-shape decoder scale ladder and exact parameter solving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleRung:
    name: str
    target_unique_parameters: int
    role: str


DEFAULT_SCALE_LADDER = (
    ScaleRung("S0", 5_000_000, "local_screening"),
    ScaleRung("S1", 12_000_000, "local_replication"),
    ScaleRung("S2", 30_000_000, "hypothesis_discrimination"),
    ScaleRung("S3", 72_000_000, "intermediate_scaling_fit"),
    ScaleRung("S4", 170_000_000, "pre_confirmatory"),
    ScaleRung("S5", 400_000_000, "confirmatory_only"),
)


@dataclass(frozen=True)
class DecoderShape:
    target_parameters: int
    hidden_size: int
    num_heads: int
    head_dim: int
    physical_blocks: int
    vocab_size: int
    embedding_parameters: int
    core_parameters: int
    normalization_parameters: int
    unique_parameters: int
    relative_error: float


def decoder_parameter_count(
    hidden_size: int,
    *,
    physical_blocks: int,
    vocab_size: int,
    affine_norm: bool = True,
) -> tuple[int, int, int, int]:
    if min(hidden_size, physical_blocks, vocab_size) <= 0:
        raise ValueError("decoder dimensions must be positive")
    embedding = vocab_size * hidden_size
    block_linear = physical_blocks * 12 * hidden_size * hidden_size
    block_norm = physical_blocks * (4 * hidden_size if affine_norm else 0)
    boundary_norm = 4 * hidden_size if affine_norm else 0
    normalization = block_norm + boundary_norm
    core = block_linear + normalization
    return embedding + core, embedding, core, normalization


def solve_decoder_width(
    target_parameters: int,
    *,
    physical_blocks: int = 2,
    vocab_size: int = 2048,
    head_dim: int = 32,
    tolerance: float = 0.03,
) -> DecoderShape:
    if target_parameters <= 0 or not 0 < tolerance < 1:
        raise ValueError("invalid target or tolerance")
    best: DecoderShape | None = None
    for hidden_size in range(head_dim, 8193, head_dim):
        total, embedding, core, normalization = decoder_parameter_count(
            hidden_size,
            physical_blocks=physical_blocks,
            vocab_size=vocab_size,
        )
        error = abs(total - target_parameters) / target_parameters
        shape = DecoderShape(
            target_parameters=target_parameters,
            hidden_size=hidden_size,
            num_heads=hidden_size // head_dim,
            head_dim=head_dim,
            physical_blocks=physical_blocks,
            vocab_size=vocab_size,
            embedding_parameters=embedding,
            core_parameters=core,
            normalization_parameters=normalization,
            unique_parameters=total,
            relative_error=error,
        )
        if best is None or error < best.relative_error:
            best = shape
        if total > target_parameters and error > (best.relative_error if best else 1):
            break
    assert best is not None
    if best.relative_error > tolerance:
        raise ValueError(
            f"no width solves {target_parameters} parameters within {tolerance:.1%}"
        )
    return best
