"""Resumable, resource-aware training cells for LSPG architecture discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from lsa.canonical import digest
from lsa.topology import ScheduleTopology
from research_gym.neural.looped_decoder import LoopedDecoderLM

from .checkpoint import load_verified_checkpoint, save_paced_checkpoint
from .metrics import TaskMetrics, evaluate_predictions
from .planner import ArchitectureProposal
from .tasks import TaskBundle, TaskExample


STAGE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "A1": ("pointer_chase", "modular_recurrence", "rewrite_normalization"),
    "A2": ("pointer_chase", "modular_recurrence", "rewrite_normalization"),
    "B": ("sudoku", "routing"),
    "C": ("pointer_chase", "modular_recurrence", "rewrite_normalization", "sudoku", "routing"),
    "D": ("pointer_chase", "modular_recurrence", "rewrite_normalization", "sudoku", "routing"),
    "calibration": ("pointer_chase",),
    "smoke": ("pointer_chase", "sudoku"),
}


@dataclass(frozen=True)
class TrainingCellConfig:
    stage: str
    scale_rung: str
    seed: int
    token_visit_budget: int
    effective_batch_size: int
    microbatch_size: int
    learning_rate: float
    warmup_fraction: float = 0.05
    checkpoint_steps: int = 100
    checkpoint_seconds: int = 120
    checkpoint_write_bytes_per_second: int = 40 * 1024 * 1024
    maximum_gradient_norm: float = 100.0
    maximum_raw_gradient_norm: float = 1_000_000.0
    maximum_loss_ratio: float = 10.0
    evaluation_limit_per_family: int | None = 256
    allow_locked_evaluation: bool = False
    vram_fraction: float | None = None
    resource_only: bool = False
    measurement_warmup_steps: int = 0

    @property
    def families(self) -> tuple[str, ...]:
        if self.stage not in STAGE_FAMILIES:
            raise ValueError(f"unknown stage: {self.stage}")
        return STAGE_FAMILIES[self.stage]


@dataclass(frozen=True)
class TrainingCellResult:
    cell_id: str
    cell_hash: str
    proposal_id: str
    proposal_hash: str
    matched_control_id: str | None
    stage: str
    scale_rung: str
    seed: int
    status: str
    stop_reason: str | None
    integrity_passed: bool
    cleanup_passed: bool
    resumed: bool
    optimizer_steps: int
    target_optimizer_steps: int
    token_visit_exposures: int
    initial_loss: float | None
    final_loss: float | None
    final_over_initial_loss: float | None
    max_gradient_norm: float
    max_raw_gradient_norm: float | None
    raw_gradient_nonfinite: bool
    nonfinite_gradient_step: int | None
    gradient_clipped_steps: int
    mean_step_seconds: float | None
    measured_step_count: int
    measurement_warmup_steps: int
    resource_only: bool
    peak_memory_bytes: int
    unique_parameters: int
    estimated_flops: int
    train_visits: int
    macro_exact: float | None
    by_family: Mapping[str, float]
    depth_metrics: Mapping[str, Mapping[str, Any]]
    prediction_artifacts: Mapping[str, Mapping[str, Any]]
    checkpoints: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        handle.write("\n")


def _proposal_model_mapping(
    proposal: ArchitectureProposal, scale_rung: str
) -> dict[str, Any]:
    topology = ScheduleTopology.from_mapping(proposal.topology)
    return {
        "proposal_id": proposal.proposal_id,
        "proposal_hash": proposal.proposal_hash,
        "mutation": {
            "expanded_visits": topology.train_visits,
            "gradient_visible_visits": topology.train_visits,
            "retained_state_edges": max(0, topology.train_visits - 1),
            "physical_modules": topology.physical_modules,
            "tying": "explicit",
            "supervision_points": 1,
            "normalization": "pre",
            "carry": "reset",
            "alpha": 1.0,
            "beta": 0.25,
        },
        "model": dict(proposal.models[scale_rung]),
        "topology": dict(proposal.topology),
    }


def _example_index(seed: int, step: int, sample: int, family: str, size: int) -> int:
    return int(digest({"seed": seed, "step": step, "sample": sample, "family": family})[:16], 16) % size


def _microbatch(
    examples_by_family: Mapping[str, Sequence[TaskExample]],
    families: Sequence[str],
    *,
    seed: int,
    step: int,
    microstep: int,
    microbatch_size: int,
    effective_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected: list[TaskExample] = []
    base = step * effective_batch_size + microstep * microbatch_size
    for offset in range(microbatch_size):
        sample = base + offset
        family = families[sample % len(families)]
        values = examples_by_family[family]
        selected.append(values[_example_index(seed, step, sample, family, len(values))])
    tokens = torch.tensor([example.tokens for example in selected], dtype=torch.long, device=device)
    targets = torch.tensor([example.target_token for example in selected], dtype=torch.long, device=device)
    return tokens, targets


def _limited_examples(
    examples: Iterable[TaskExample], families: Sequence[str], limit: int | None
) -> tuple[TaskExample, ...]:
    grouped: dict[str, list[TaskExample]] = {family: [] for family in families}
    for example in sorted(examples, key=lambda value: value.example_id):
        if example.family in grouped and (limit is None or len(grouped[example.family]) < limit):
            grouped[example.family].append(example)
    return tuple(value for family in families for value in grouped[family])


def evaluate_model(
    model: LoopedDecoderLM,
    examples: Sequence[TaskExample],
    *,
    depth_visits: int,
    batch_size: int,
    device: torch.device,
) -> tuple[TaskMetrics, tuple[Mapping[str, Any], ...]]:
    predictions: dict[str, int] = {}
    model.eval()
    amp = device.type == "cuda"
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            tokens = torch.tensor([example.tokens for example in batch], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                output = model(tokens, depth_visits=depth_visits)
            values = output.logits[:, -1].argmax(dim=-1).detach().cpu().tolist()
            predictions.update(
                {example.example_id: int(value) for example, value in zip(batch, values)}
            )
    rows = tuple(
        {
            "example_id": example.example_id,
            "fingerprint": example.fingerprint,
            "family": example.family,
            "split": example.split,
            "difficulty": example.difficulty,
            "target_token": example.target_token,
            "prediction_token": predictions[example.example_id],
            "exact": int(predictions[example.example_id] == example.target_token),
            "group": {
                key: example.metadata[key]
                for key in ("puzzle_id", "env_id")
                if key in example.metadata
            },
        }
        for example in examples
    )
    return evaluate_predictions(examples, predictions), rows


def write_prediction_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    display_root: Path | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")
    display_path = path
    if display_root is not None:
        try:
            display_path = path.resolve().relative_to(display_root.resolve())
        except ValueError:
            pass
    return {
        "path": str(display_path).replace("\\", "/"),
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _scheduler_factor(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return max(1e-3, (step + 1) / warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def measured_step_times(
    step_times: Sequence[float], measurement_warmup_steps: int
) -> tuple[float, ...]:
    if measurement_warmup_steps < 0:
        raise ValueError("measurement warm-up steps must be non-negative")
    return tuple(float(value) for value in step_times[measurement_warmup_steps:])


def training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def run_training_cell(
    proposal: ArchitectureProposal,
    bundle: TaskBundle,
    config: TrainingCellConfig,
    *,
    output_dir: Path,
) -> TrainingCellResult:
    if config.effective_batch_size % config.microbatch_size:
        raise ValueError("microbatch must divide effective batch")
    if config.scale_rung not in proposal.models:
        raise ValueError("proposal does not materialize the requested scale")
    if config.resource_only and config.stage != "calibration":
        raise ValueError("resource-only cells must use the calibration stage")
    topology = ScheduleTopology.from_mapping(proposal.topology)
    exposures_per_step = (
        config.effective_batch_size * bundle.sequence_length * topology.train_visits
    )
    target_steps = max(1, math.ceil(config.token_visit_budget / exposures_per_step))
    cell_core = {
        "proposal_hash": proposal.proposal_hash,
        "task_bundle_hash": bundle.manifest()["bundle_hash"],
        "stage": config.stage,
        "scale_rung": config.scale_rung,
        "seed": config.seed,
        "token_visit_budget": config.token_visit_budget,
        "effective_batch_size": config.effective_batch_size,
        "microbatch_size": config.microbatch_size,
        "learning_rate": config.learning_rate,
        "resource_only": config.resource_only,
        "measurement_warmup_steps": config.measurement_warmup_steps,
    }
    cell_hash = digest(cell_core)
    cell_id = f"{proposal.proposal_id}-{config.stage}-{config.scale_rung}-s{config.seed}"
    cell_dir = output_dir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    event_path = cell_dir / "events.jsonl"
    result_path = cell_dir / "result.json"
    checkpoint_path = cell_dir / "checkpoints" / "latest.pt"
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = training_device()
    if device.type == "cuda" and config.vram_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(config.vram_fraction, device=device)
    model = LoopedDecoderLM.from_proposal(_proposal_model_mapping(proposal, config.scale_rung)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _scheduler_factor(step, target_steps, config.warmup_fraction),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_examples = bundle.split("train", config.families)
    examples_by_family = {
        family: tuple(example for example in train_examples if example.family == family)
        for family in config.families
    }
    if any(not values for values in examples_by_family.values()):
        raise ValueError("every selected task family needs training examples")
    state: dict[str, Any] = {
        "step": 0,
        "initial_loss": None,
        "losses": [],
        "max_gradient_norm": 0.0,
        "max_raw_gradient_norm": 0.0,
        "raw_gradient_nonfinite": False,
        "nonfinite_gradient_step": None,
        "gradient_clipped_steps": 0,
        "step_times": [],
        "checkpoints": [],
    }
    resumed = False
    if checkpoint_path.exists():
        saved = load_verified_checkpoint(checkpoint_path, map_location=str(device))
        if saved["cell_hash"] != cell_hash:
            raise ValueError("checkpoint belongs to another cell")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["training_state"]
        state.setdefault("max_raw_gradient_norm", 0.0)
        state.setdefault("raw_gradient_nonfinite", False)
        state.setdefault("nonfinite_gradient_step", None)
        state.setdefault("gradient_clipped_steps", 0)
        resumed = True
    _event(
        event_path,
        {
            "event": "resume" if resumed else "start",
            "cell_id": cell_id,
            "cell_hash": cell_hash,
            "step": state["step"],
            "target_steps": target_steps,
            "time": time.time(),
        },
    )
    stop_reason: str | None = None
    last_checkpoint_time = time.monotonic()
    cleanup_passed = False
    result: TrainingCellResult | None = None
    peak_memory = 0
    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        accumulation = config.effective_batch_size // config.microbatch_size
        for step in range(int(state["step"]), target_steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for microstep in range(accumulation):
                tokens, targets = _microbatch(
                    examples_by_family,
                    config.families,
                    seed=config.seed,
                    step=step,
                    microstep=microstep,
                    microbatch_size=config.microbatch_size,
                    effective_batch_size=config.effective_batch_size,
                    device=device,
                )
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    output = model(tokens)
                    loss = torch.nn.functional.cross_entropy(output.logits[:, -1], targets)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    stop_reason = "nonfinite_loss"
                    break
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.detach().item()) / accumulation
            if stop_reason:
                break
            scaler.unscale_(optimizer)
            raw_gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.maximum_gradient_norm
                ).item()
            )
            if not math.isfinite(raw_gradient):
                state["raw_gradient_nonfinite"] = True
                state["nonfinite_gradient_step"] = step
                stop_reason = "nonfinite_gradient"
                break
            gradient = min(raw_gradient, config.maximum_gradient_norm)
            state["max_gradient_norm"] = max(float(state["max_gradient_norm"]), gradient)
            state["max_raw_gradient_norm"] = max(
                float(state["max_raw_gradient_norm"]), raw_gradient
            )
            state["gradient_clipped_steps"] = int(state["gradient_clipped_steps"]) + int(
                raw_gradient > config.maximum_gradient_norm
            )
            if raw_gradient > config.maximum_raw_gradient_norm:
                stop_reason = "catastrophic_raw_gradient_norm"
                break
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state["losses"].append(accumulated_loss)
            state["initial_loss"] = state["initial_loss"] or accumulated_loss
            state["step"] = step + 1
            if amp:
                torch.cuda.synchronize(device)
                peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
            state["step_times"].append(time.perf_counter() - started)
            checkpoint_due = (
                state["step"] % config.checkpoint_steps == 0
                or time.monotonic() - last_checkpoint_time >= config.checkpoint_seconds
                or state["step"] == target_steps
            )
            if checkpoint_due:
                checkpoint_receipt = save_paced_checkpoint(
                    {
                        "cell_hash": cell_hash,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "training_state": state,
                    },
                    checkpoint_path,
                    bytes_per_second=config.checkpoint_write_bytes_per_second,
                )
                checkpoint_record = {
                    "step": state["step"],
                    "token_visit_exposures": state["step"] * exposures_per_step,
                    **checkpoint_receipt,
                }
                state["checkpoints"].append(checkpoint_record)
                _event(event_path, {"event": "checkpoint", "cell_id": cell_id, **checkpoint_record})
                last_checkpoint_time = time.monotonic()
        losses = [float(value) for value in state["losses"]]
        initial_loss = float(state["initial_loss"]) if state["initial_loss"] is not None else None
        final_loss = losses[-1] if losses else None
        loss_ratio = final_loss / initial_loss if initial_loss and final_loss is not None else None
        if stop_reason is None and loss_ratio is not None and loss_ratio > config.maximum_loss_ratio:
            stop_reason = "final_over_initial_loss_above_10"
        status = "completed" if stop_reason is None and state["step"] == target_steps else "stopped"
        if stop_reason is not None:
            _event(
                event_path,
                {
                    "event": "stop",
                    "cell_id": cell_id,
                    "step": int(state["step"]),
                    "stop_reason": stop_reason,
                    "raw_gradient_nonfinite": bool(state["raw_gradient_nonfinite"]),
                },
            )
        depth_metrics: dict[str, Mapping[str, Any]] = {}
        prediction_artifacts: dict[str, Mapping[str, Any]] = {}
        macro_exact: float | None = None
        by_family: Mapping[str, float] = {}
        if status == "completed" and not config.resource_only:
            split = "evaluation" if config.stage == "D" else "calibration"
            if split == "evaluation" and not config.allow_locked_evaluation:
                raise ValueError("locked evaluation requires explicit Stage-D authorization")
            evaluation = _limited_examples(
                bundle.split(split, config.families),
                config.families,
                config.evaluation_limit_per_family,
            )
            for depth in topology.inference_visits:
                metrics, prediction_rows = evaluate_model(
                    model,
                    evaluation,
                    depth_visits=depth,
                    batch_size=config.microbatch_size,
                    device=device,
                )
                depth_metrics[str(depth)] = asdict(metrics)
                prediction_artifacts[str(depth)] = write_prediction_artifact(
                    cell_dir / "predictions" / f"depth_{depth}.jsonl",
                    prediction_rows,
                    display_root=Path.cwd(),
                )
            primary = depth_metrics[str(topology.train_visits)]
            macro_exact = float(primary["macro_exact"])
            by_family = dict(primary["by_family"])
        resource = next(
            value for value in proposal.resources if value.scale_rung == config.scale_rung
        )
        measured_times = measured_step_times(
            state["step_times"], config.measurement_warmup_steps
        )
        result = TrainingCellResult(
            cell_id=cell_id,
            cell_hash=cell_hash,
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.proposal_hash,
            matched_control_id=proposal.matched_control_id,
            stage=config.stage,
            scale_rung=config.scale_rung,
            seed=config.seed,
            status=status,
            stop_reason=stop_reason,
            integrity_passed=True,
            cleanup_passed=False,
            resumed=resumed,
            optimizer_steps=int(state["step"]),
            target_optimizer_steps=target_steps,
            token_visit_exposures=int(state["step"]) * exposures_per_step,
            initial_loss=None if config.resource_only else initial_loss,
            final_loss=None if config.resource_only else final_loss,
            final_over_initial_loss=None if config.resource_only else loss_ratio,
            max_gradient_norm=float(state["max_gradient_norm"]),
            max_raw_gradient_norm=(
                float(state["max_raw_gradient_norm"])
                if float(state["max_raw_gradient_norm"]) > 0.0
                else None
            ),
            raw_gradient_nonfinite=bool(state["raw_gradient_nonfinite"]),
            nonfinite_gradient_step=(
                int(state["nonfinite_gradient_step"])
                if state["nonfinite_gradient_step"] is not None
                else None
            ),
            gradient_clipped_steps=int(state["gradient_clipped_steps"]),
            mean_step_seconds=(
                sum(measured_times) / len(measured_times)
                if measured_times
                else None
            ),
            measured_step_count=len(measured_times),
            measurement_warmup_steps=config.measurement_warmup_steps,
            resource_only=config.resource_only,
            peak_memory_bytes=peak_memory,
            unique_parameters=model.parameter_breakdown()["unique_parameters"],
            estimated_flops=resource.estimated_flops_per_example,
            train_visits=topology.train_visits,
            macro_exact=macro_exact,
            by_family=by_family,
            depth_metrics=depth_metrics,
            prediction_artifacts=prediction_artifacts,
            checkpoints=tuple(state["checkpoints"]),
        )
    finally:
        del optimizer, scheduler, scaler, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        cleanup_passed = True
    assert result is not None
    result = TrainingCellResult(**{**result.to_dict(), "cleanup_passed": cleanup_passed})
    payload = result.to_dict()
    _strict_json(result_path, payload)
    _event(event_path, {"event": "finish", "cell_id": cell_id, "status": result.status})
    return result


def result_receipt(result_path: Path, *, display_root: Path | None = None) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    display_path = result_path
    if display_root is not None:
        try:
            display_path = result_path.resolve().relative_to(display_root.resolve())
        except ValueError:
            pass
    return {
        "cell_id": result["cell_id"],
        "cell_hash": result["cell_hash"],
        "result_path": str(display_path).replace("\\", "/"),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "status": result["status"],
        "integrity_passed": result["integrity_passed"],
        "cleanup_passed": result["cleanup_passed"],
    }
