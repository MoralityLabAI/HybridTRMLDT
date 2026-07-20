"""Resource-only calibration and frozen campaign-profile selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsa.canonical import digest


STAGE_CELL_COUNTS = {
    "A1": 18,
    "A2": 36,
    "B": 24,
    "C": 24,
    "D": 24,
}


@dataclass(frozen=True)
class CalibrationMeasurement:
    scale_rung: str
    mean_step_seconds: float
    measured_step_count: int
    peak_memory_bytes: int
    peak_ram_mb: float
    peak_vram_mb: float
    peak_io_mb_s: float
    result_sha256: str
    resource_receipt_sha256: str


def calibration_token_visit_budget(
    *, effective_batch_size: int, sequence_length: int, visits: int = 8, steps: int = 50
) -> int:
    return effective_batch_size * sequence_length * visits * steps


def latest_completed_resource_receipt(paths: Sequence[Path]) -> Path:
    completed: list[tuple[int, Path]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            completed.append((int(value["attempt"]), path))
    if not completed:
        raise ValueError("no completed calibration resource receipt")
    return max(completed, key=lambda value: value[0])[1]


def load_calibration_measurement(
    result_path: Path,
    resource_receipt_path: Path,
    *,
    expected_warmup_steps: int = 20,
    expected_measured_steps: int = 30,
) -> CalibrationMeasurement:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resource = json.loads(resource_receipt_path.read_text(encoding="utf-8"))
    if result["status"] != "completed" or resource["status"] != "completed":
        raise ValueError("calibration cell did not complete")
    if not result["integrity_passed"] or not result["cleanup_passed"]:
        raise ValueError("calibration result integrity or cleanup failed")
    if not resource["cleanup_passed"] or resource["lingering_owned_process"]:
        raise ValueError("calibration wrapper cleanup failed")
    if not result["resource_only"]:
        raise ValueError("calibration exposed a task-outcome cell")
    if result.get("precision") != "fp32":
        raise ValueError("v1.2 calibration requires FP32 precision")
    if any(result[field] is not None for field in ("macro_exact", "initial_loss", "final_loss")):
        raise ValueError("resource-only calibration leaked task outcomes")
    if result["by_family"] or result["depth_metrics"] or result["prediction_artifacts"]:
        raise ValueError("resource-only calibration leaked evaluation metrics")
    if int(result["measurement_warmup_steps"]) != expected_warmup_steps:
        raise ValueError("calibration warm-up count mismatch")
    if int(result["measured_step_count"]) != expected_measured_steps:
        raise ValueError("calibration measurement count mismatch")
    mean_seconds = float(result["mean_step_seconds"])
    if not math.isfinite(mean_seconds) or mean_seconds <= 0.0:
        raise ValueError("calibration step time must be finite and positive")
    return CalibrationMeasurement(
        scale_rung=str(result["scale_rung"]),
        mean_step_seconds=mean_seconds,
        measured_step_count=int(result["measured_step_count"]),
        peak_memory_bytes=int(result["peak_memory_bytes"]),
        peak_ram_mb=float(resource["peak_ram_mb"]),
        peak_vram_mb=float(resource["peak_vram_mb"]),
        peak_io_mb_s=float(resource["peak_io_mb_s"]),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        resource_receipt_sha256=hashlib.sha256(resource_receipt_path.read_bytes()).hexdigest(),
    )


def profile_forecast(
    policy: Mapping[str, Any],
    measurements: Mapping[str, CalibrationMeasurement],
    *,
    effective_batch_size: int,
    sequence_length: int,
    profile_name: str,
    safety_factor: float = 1.2,
) -> dict[str, Any]:
    stage_rows: dict[str, Any] = {}
    total_seconds = 0.0
    for stage, stage_spec in policy["funnel"].items():
        scale = str(stage_spec["scale"])
        budget = int(policy["profiles"][profile_name][stage])
        # L=6 maximizes optimizer steps at a fixed token-visit budget. The K=2,
        # L=8 calibration step is retained as the conservative per-step price.
        steps = max(1, math.ceil(budget / (effective_batch_size * sequence_length * 6)))
        cells = STAGE_CELL_COUNTS[stage]
        raw_seconds = cells * steps * measurements[scale].mean_step_seconds
        guarded_seconds = raw_seconds * safety_factor
        total_seconds += guarded_seconds
        stage_rows[stage] = {
            "scale_rung": scale,
            "token_visit_budget": budget,
            "conservative_optimizer_steps": steps,
            "maximum_cells": cells,
            "measured_step_seconds": measurements[scale].mean_step_seconds,
            "safety_factor": safety_factor,
            "forecast_gpu_seconds": guarded_seconds,
        }
    return {
        "profile": profile_name,
        "stage_forecasts": stage_rows,
        "mandatory_path_gpu_seconds": total_seconds,
        "mandatory_path_gpu_hours": total_seconds / 3600.0,
    }


def select_budget_profile(
    policy: Mapping[str, Any],
    measurements: Sequence[CalibrationMeasurement],
    *,
    effective_batch_size: int,
    sequence_length: int,
    safety_factor: float = 1.2,
) -> dict[str, Any]:
    by_scale = {value.scale_rung: value for value in measurements}
    if set(by_scale) != {"S0", "S1", "S2"}:
        raise ValueError("calibration requires exactly S0, S1, and S2")
    forecasts = {
        name: profile_forecast(
            policy,
            by_scale,
            effective_batch_size=effective_batch_size,
            sequence_length=sequence_length,
            profile_name=name,
            safety_factor=safety_factor,
        )
        for name in ("full", "medium", "minimum")
    }
    limit = float(policy["profile_selection"]["mandatory_path_gpu_seconds"])
    selected = next(
        (
            name
            for name in ("full", "medium", "minimum")
            if forecasts[name]["mandatory_path_gpu_seconds"] <= limit
        ),
        None,
    )
    status = "selected" if selected is not None else "construction_failure"
    payload = {
        "schema_version": 1,
        "status": status,
        "selected_profile": selected,
        "selection_rule": "largest_registered_profile_within_24_gpu_hours",
        "safety_factor": safety_factor,
        "task_outcomes_observed": False,
        "measurements": [asdict(by_scale[scale]) for scale in ("S0", "S1", "S2")],
        "forecasts": forecasts,
        "mandatory_path_limit_gpu_seconds": limit,
        "reserve_gpu_seconds": int(policy["profile_selection"]["reserve_gpu_seconds"]),
    }
    payload["selection_hash"] = digest(payload)
    return payload
