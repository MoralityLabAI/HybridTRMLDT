"""Bounded Stage-A training helpers for LoopedDecoderLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .looped_decoder import LoopedDecoderLM


@dataclass(frozen=True)
class ScreeningResult:
    proposal_id: str
    proposal_hash: str
    status: str
    stop_reason: str | None
    steps_completed: int
    state_visit_exposures: int
    initial_loss: float
    final_loss: float
    final_over_initial_loss: float
    max_gradient_norm: float
    mean_step_seconds: float
    peak_memory_bytes: int
    parameter_breakdown: Mapping[str, int]
    checkpoints: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_tokens(
    *, step: int, sequence_length: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    start = (step * 17 + 11) % vocab_size
    increments = torch.arange(sequence_length, device=device)
    return ((start + increments * 13 + (increments // 7) * 5) % vocab_size).unsqueeze(0)


def _loss(output, tokens: torch.Tensor) -> torch.Tensor:
    target = tokens[:, 1:]
    losses = [
        torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
            target.reshape(-1),
        )
    ]
    losses.extend(
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), target.reshape(-1)
        )
        for logits in output.auxiliary_logits
    )
    return torch.stack(losses).mean()


def screen_proposal(
    proposal: Mapping[str, Any],
    *,
    output_dir: Path,
    exposure_budget: int,
    checkpoints_pct: tuple[int, ...],
    seed: int,
) -> ScreeningResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoopedDecoderLM.from_proposal(proposal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    visits = int(proposal["mutation"]["expanded_visits"])
    sequence = int(proposal["mutation"]["sequence_length"])
    exposures_per_step = sequence * visits
    total_steps = max(1, math.ceil(exposure_budget / exposures_per_step))
    checkpoint_steps: dict[int, list[int]] = {}
    for percentage in checkpoints_pct:
        checkpoint_steps.setdefault(
            max(1, math.ceil(total_steps * percentage / 100)), []
        ).append(percentage)
    latest_checkpoint = output_dir / "checkpoints" / proposal["proposal_id"] / "latest.pt"
    latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    carry = None
    losses = []
    max_gradient = 0.0
    step_times = []
    checkpoint_records = []
    stop_reason = None
    peak_memory = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1, total_steps + 1):
            started = time.perf_counter()
            tokens = deterministic_tokens(
                step=step, sequence_length=sequence, vocab_size=model.vocab_size, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, carry=carry)
            carry = output.carry
            loss = _loss(output, tokens)
            if not torch.isfinite(loss):
                stop_reason = "nonfinite_loss"
                break
            loss.backward()
            gradient_sq = sum(
                float(parameter.grad.detach().float().square().sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient = math.sqrt(gradient_sq)
            max_gradient = max(max_gradient, gradient)
            if not math.isfinite(gradient):
                stop_reason = "nonfinite_gradient"
                break
            if gradient > 100.0:
                stop_reason = "gradient_norm_above_100"
                break
            optimizer.step()
            losses.append(float(loss.detach().item()))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
            step_times.append(time.perf_counter() - started)
            if step in checkpoint_steps:
                step_records = []
                for percentage in checkpoint_steps[step]:
                    checkpoint = {
                        "proposal_id": proposal["proposal_id"],
                        "proposal_hash": proposal["proposal_hash"],
                        "step": step,
                        "percentage": percentage,
                        "loss": losses[-1],
                        "gradient_norm": gradient,
                        "state_visit_exposures": step * exposures_per_step,
                    }
                    checkpoint_records.append(checkpoint)
                    step_records.append(checkpoint)
                torch.save(
                    {
                        "checkpoints": step_records,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    },
                    latest_checkpoint,
                )
        if not losses:
            initial_loss = final_loss = math.nan
            ratio = math.inf
        else:
            initial_loss = losses[0]
            final_loss = losses[-1]
            ratio = final_loss / initial_loss
            if stop_reason is None and ratio > 10.0:
                stop_reason = "final_over_initial_loss_above_10"
        status = "completed" if stop_reason is None else "stopped"
        return ScreeningResult(
            proposal_id=proposal["proposal_id"],
            proposal_hash=proposal["proposal_hash"],
            status=status,
            stop_reason=stop_reason,
            steps_completed=len(losses),
            state_visit_exposures=len(losses) * exposures_per_step,
            initial_loss=initial_loss,
            final_loss=final_loss,
            final_over_initial_loss=ratio,
            max_gradient_norm=max_gradient,
            mean_step_seconds=sum(step_times) / len(step_times) if step_times else math.nan,
            peak_memory_bytes=peak_memory,
            parameter_breakdown=model.parameter_breakdown(),
            checkpoints=tuple(checkpoint_records),
        )
    finally:
        del optimizer, model, carry
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
