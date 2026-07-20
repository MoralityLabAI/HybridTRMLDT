"""Registered LSA v0.1 visit-alignment and stability campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn

from lsa.kappa_probe import ToyLoop, estimate_kappa, geometric_mean
from lsa.mini_transformer_probe import MiniTransformerLoop
from research_gym.analysis.lsa_v0_1 import (
    bootstrap_gamma,
    classify_gamma_trajectory,
    empirical_boundary,
    fit_records,
    score_r16_holdout,
    stable_training,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_algebra_v0_1.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_algebra_v0_1"
REGISTRATION = ROOT / "configs" / "loop_schedule_algebra_v0_1_registration.json"
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_v0_1_receipt.json"
CHECKPOINT_PATTERN = re.compile(
    r"^(?P<cell>gamma_(?P<regime>tied|untied)_R(?P<rounds>\d+)_S(?P<seed>\d+))_step_(?P<step>\d+)\.pt$"
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in sorted(records, key=lambda row: row["record_id"]):
            handle.write(_canonical_bytes(record))


def _append_event(path: Path, event: dict[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(payload))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    actual_hash = _sha256(path)
    if actual_hash != registration["config_sha256"]:
        raise RuntimeError(
            f"registered config hash mismatch: expected {registration['config_sha256']}, got {actual_hash}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("protocol id does not match registration")
    return config, actual_hash


def _verify_source_blobs(config: dict[str, Any]) -> dict[str, str]:
    source = config["source_v0"]
    checks = {
        "receipt": (source["receipt_path"], source["receipt_sha256"]),
        "gamma_records": (source["gamma_records_path"], source["gamma_records_sha256"]),
        "gamma_result": (source["gamma_result_path"], source["gamma_result_sha256"]),
    }
    verified: dict[str, str] = {}
    for name, (path, expected) in checks.items():
        actual = hashlib.sha256(_git_blob(source["git_commit"], path)).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source v0 {name} hash mismatch: expected {expected}, got {actual}")
        verified[name] = actual
    return verified


def _device(config: dict[str, Any]) -> torch.device:
    if config["resources"]["device"] != "cuda_if_available_else_cpu":
        raise ValueError("unsupported device policy")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_vram_fraction(config: dict[str, Any], requested: float | None) -> None:
    if not torch.cuda.is_available():
        return
    registered = float(config["resources"]["torch_vram_fraction"])
    fraction = registered if requested is None else float(requested)
    if fraction <= 0.0 or fraction > registered:
        raise ValueError("VRAM fraction must be positive and no greater than the registered cap")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def measurement_seed(seed: int, exposure: int | None = None) -> int:
    """Keep the stochastic probe direction fixed across training checkpoints."""

    del exposure
    return int(seed) + 50_000


def signed_permutation_batch(
    seed: int,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    *,
    stream: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + stream)
    inputs = torch.randn(batch_size, hidden_size, generator=generator)
    permutation = torch.randperm(hidden_size, generator=torch.Generator().manual_seed(seed + 17))
    signs = torch.where(
        torch.rand(hidden_size, generator=torch.Generator().manual_seed(seed + 29)) > 0.5,
        1.0,
        -1.0,
    )
    targets = torch.tanh(inputs[:, permutation] * signs)
    return inputs.to(device), targets.to(device)


def prefix_context_batch(
    seed: int,
    batch_size: int,
    hidden_size: int,
    sequence_length: int,
    device: torch.device,
    *,
    stream: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed * 2017 + stream)
    inputs = torch.randn(batch_size, sequence_length, hidden_size, generator=generator)
    permutation = torch.randperm(hidden_size, generator=torch.Generator().manual_seed(seed + 41))
    signs = torch.where(
        torch.rand(hidden_size, generator=torch.Generator().manual_seed(seed + 53)) > 0.5,
        1.0,
        -1.0,
    )
    transformed = inputs[..., permutation] * signs
    denominator = torch.arange(1, sequence_length + 1, dtype=inputs.dtype).view(1, -1, 1)
    prefix_mean = transformed.cumsum(dim=1) / denominator
    targets = torch.tanh(0.5 * transformed + 0.5 * prefix_mean)
    return inputs.to(device), targets.to(device)


def _family_spec(config: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "primary_mlp":
        return config["primary_mlp"]
    if family == "external_validity":
        return config["external_validity"]
    raise ValueError(f"unsupported family: {family}")


def _build_model(
    config: dict[str, Any],
    family: str,
    *,
    rounds: int,
    tied: bool,
    alpha: float,
    beta: float,
) -> nn.Module:
    spec = _family_spec(config, family)
    if family == "primary_mlp":
        return ToyLoop(spec["hidden_size"], rounds, tied=tied, alpha=alpha, beta=beta)
    return MiniTransformerLoop(
        spec["hidden_size"],
        rounds,
        tied=tied,
        heads=spec["attention_heads"],
        mlp_ratio=spec["mlp_ratio"],
        alpha=alpha,
        beta=beta,
    )


def _task_batch(
    config: dict[str, Any],
    family: str,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
    stream: int,
) -> tuple[Tensor, Tensor]:
    spec = _family_spec(config, family)
    if family == "primary_mlp":
        return signed_permutation_batch(seed, batch_size, spec["hidden_size"], device, stream=stream)
    return prefix_context_batch(
        seed,
        batch_size,
        spec["hidden_size"],
        spec["sequence_length"],
        device,
        stream=stream,
    )


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output_dir: Path,
    cell_id: str,
    step: int,
    exposure: int,
) -> dict[str, Any]:
    path = output_dir / "checkpoints" / f"{cell_id}_E{exposure}_step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cell_id": cell_id,
            "step": step,
            "exposure": exposure,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _train_cell(
    config: dict[str, Any],
    *,
    family: str,
    seed: int,
    rounds: int,
    tied: bool,
    alpha: float,
    beta: float,
    learning_rate: float,
    output_dir: Path,
    cell_id: str,
    event_path: Path,
    measurement_exposures: Iterable[int] = (),
    checkpoint_exposures: Iterable[int] = (),
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    shared = config["shared_training"]
    spec = _family_spec(config, family)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config)
    model = _build_model(
        config, family, rounds=rounds, tied=tied, alpha=alpha, beta=beta
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=shared["weight_decay"]
    )
    batch_size = int(spec["batch_size"])
    exposures_per_step = batch_size * rounds
    progress_target = int(shared["progress_target"])
    if progress_target % exposures_per_step:
        raise RuntimeError(f"{cell_id} cannot land exactly on the registered exposure target")
    measurement_targets = {int(value) for value in measurement_exposures}
    checkpoint_targets = {int(value) for value in checkpoint_exposures}
    all_targets = measurement_targets | checkpoint_targets
    if any(value < 0 or value > progress_target or value % exposures_per_step for value in all_targets):
        raise RuntimeError(f"{cell_id} has an unreachable registered checkpoint exposure")
    steps = progress_target // exposures_per_step
    initial_loss: float | None = None
    final_loss = math.nan
    max_gradient_norm = 0.0
    nonfinite = False
    measurements: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    def observe(step: int, exposure: int) -> None:
        if exposure in checkpoint_targets:
            checkpoint = _save_checkpoint(
                model,
                optimizer,
                output_dir=output_dir,
                cell_id=cell_id,
                step=step,
                exposure=exposure,
            )
            checkpoints.append(checkpoint)
            _append_event(
                event_path,
                {"event": "checkpoint", "cell_id": cell_id, "exposure": exposure, **checkpoint},
            )
        if exposure in measurement_targets:
            inputs, targets = _task_batch(
                config,
                family,
                seed=seed,
                batch_size=int(shared["measurement_batch_size"]),
                device=device,
                stream=999_001,
            )
            estimate = estimate_kappa(
                model,
                inputs,
                targets,
                power_iterations=int(shared["power_iterations"]),
                seed=measurement_seed(seed, exposure),
            )
            measurements.append(
                {
                    "exposure": exposure,
                    "kappa": estimate.kappa,
                    "estimate": estimate.to_dict(),
                }
            )
            del inputs, targets

    _append_event(
        event_path,
        {
            "event": "cell_start",
            "cell_id": cell_id,
            "family": family,
            "rounds": rounds,
            "tied": tied,
            "steps": steps,
            "device": str(device),
        },
    )
    observe(0, 0)
    completed_steps = 0
    for step in range(1, steps + 1):
        inputs, targets = _task_batch(
            config,
            family,
            seed=seed,
            batch_size=batch_size,
            device=device,
            stream=step - 1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        if not torch.isfinite(loss):
            nonfinite = True
            break
        loss.backward()
        grad_sq = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = math.sqrt(grad_sq)
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        if not math.isfinite(gradient_norm):
            nonfinite = True
            break
        optimizer.step()
        final_loss = float(loss.detach().item())
        completed_steps = step
        exposure = step * exposures_per_step
        if exposure in all_targets:
            observe(step, exposure)
        del inputs, targets, loss
    training = {
        "steps": completed_steps,
        "state_visit_exposures": completed_steps * exposures_per_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_gradient_norm": max_gradient_norm,
        "nonfinite": nonfinite,
        "checkpoints": checkpoints,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate,
        "device": str(device),
    }
    _append_event(event_path, {"event": "cell_complete", "cell_id": cell_id, **training})
    return model, measurements, training


def _phase_result(
    config: dict[str, Any],
    config_hash: str,
    output_dir: Path,
    phase: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    records_path = output_dir / f"{phase}_records.jsonl"
    _write_records(records_path, records)
    result = {
        "protocol_id": config["protocol_id"],
        "phase": phase,
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(records),
        "summary": summary,
    }
    _write_json(output_dir / f"{phase}_result.json", result)
    return result


def run_source_replay(
    config: dict[str, Any], config_hash: str, output_dir: Path, checkpoint_dir: Path
) -> dict[str, Any]:
    verified_blobs = _verify_source_blobs(config)
    source = config["source_v0"]
    paths = sorted(checkpoint_dir.glob(source["checkpoint_glob"]), key=lambda path: path.name)
    if len(paths) != source["expected_checkpoint_count"]:
        raise RuntimeError(
            f"expected {source['expected_checkpoint_count']} source checkpoints, found {len(paths)}"
        )
    inventory = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]
    inventory_path = output_dir / "source_checkpoint_manifest.json"
    _write_json(
        inventory_path,
        {
            "protocol_id": config["protocol_id"],
            "source_directory": str(checkpoint_dir),
            "checkpoint_count": len(inventory),
            "checkpoints": inventory,
        },
    )
    manifest_hash_before_load = _sha256(inventory_path)
    source_rows = [
        json.loads(line)
        for line in _git_blob(source["git_commit"], source["gamma_records_path"]).splitlines()
    ]
    expected_terminal = {
        row["record_id"]: float(row["kappa"])
        for row in source_rows
        if row["kind"] == "gamma"
    }
    parsed = []
    terminal_step: dict[str, int] = defaultdict(int)
    for path in paths:
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            raise RuntimeError(f"unexpected checkpoint filename: {path.name}")
        values = match.groupdict()
        terminal_step[values["cell"]] = max(terminal_step[values["cell"]], int(values["step"]))
        parsed.append((path, values))
    records: list[dict[str, Any]] = []
    device = _device(config)
    primary = config["primary_mlp"]
    shared = config["shared_training"]
    for path, values in parsed:
        rounds = int(values["rounds"])
        seed = int(values["seed"])
        step = int(values["step"])
        regime = values["regime"]
        model = ToyLoop(
            primary["hidden_size"],
            rounds,
            tied=regime == "tied",
            alpha=shared["alpha"],
            beta=shared["beta"],
        ).to(device)
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["cell_id"] != values["cell"] or int(payload["step"]) != step:
            raise RuntimeError(f"checkpoint identity mismatch: {path.name}")
        model.load_state_dict(payload["state_dict"])
        inputs, targets = signed_permutation_batch(
            seed,
            shared["measurement_batch_size"],
            primary["hidden_size"],
            device,
            stream=999_001,
        )
        estimate = estimate_kappa(
            model,
            inputs,
            targets,
            power_iterations=shared["power_iterations"],
            seed=measurement_seed(seed, step * primary["batch_size"] * rounds),
        )
        is_terminal = step == terminal_step[values["cell"]]
        expected = expected_terminal[values["cell"]] if is_terminal else None
        records.append(
            {
                "record_id": f"replay_{values['cell']}_step_{step}",
                "kind": "source_checkpoint_replay",
                "regime": regime,
                "rounds": rounds,
                "seed": seed,
                "step": step,
                "exposure": step * primary["batch_size"] * rounds,
                "checkpoint_sha256": _sha256(path),
                "kappa": estimate.kappa,
                "terminal": is_terminal,
                "sealed_terminal_kappa": expected,
                "terminal_absolute_error": None if expected is None else abs(estimate.kappa - expected),
            }
        )
        del model, payload, inputs, targets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    terminal_errors = [
        row["terminal_absolute_error"] for row in records if row["terminal_absolute_error"] is not None
    ]
    max_error = max(terminal_errors)
    summary = {
        "source_blobs": verified_blobs,
        "checkpoint_manifest_path": str(inventory_path.relative_to(ROOT)),
        "checkpoint_manifest_sha256_before_load": manifest_hash_before_load,
        "checkpoint_manifest_sha256_after_load": _sha256(inventory_path),
        "checkpoint_count": len(inventory),
        "terminal_cells": len(terminal_errors),
        "maximum_terminal_absolute_error": max_error,
        "replay_tolerance": 1e-5,
        "replay_passed": max_error <= 1e-5 and _sha256(inventory_path) == manifest_hash_before_load,
        "trajectory_role": source["replay_role"],
        "trajectory_limit": source["replay_limit"],
    }
    return _phase_result(config, config_hash, output_dir, "source_replay", records, summary)


def _require_phase(output_dir: Path, phase: str, config_hash: str) -> dict[str, Any]:
    path = output_dir / f"{phase}_result.json"
    if not path.exists():
        raise RuntimeError(f"required phase is missing: {phase}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash:
        raise RuntimeError(f"required phase has wrong config hash: {phase}")
    records_path = ROOT / result["records_path"]
    if _sha256(records_path) != result["records_sha256"]:
        raise RuntimeError(f"required phase records failed rehash: {phase}")
    return result


def run_primary(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    replay = _require_phase(output_dir, "source_replay", config_hash)
    if not replay["summary"]["replay_passed"]:
        raise RuntimeError("source replay did not pass")
    shared = config["shared_training"]
    primary = config["primary_mlp"]
    checkpoints = shared["checkpoint_exposures"]
    event_path = output_dir / "primary_events.jsonl"
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for regime in shared["regimes"]:
        tied = regime == "tied"
        for rounds in primary["rounds"]:
            for seed in shared["seeds"]:
                cell_id = f"primary_{regime}_R{rounds}_S{seed}"
                measure = checkpoints if tied else [shared["progress_target"]]
                save = checkpoints if tied else [shared["progress_target"]]
                model, measurements, training = _train_cell(
                    config,
                    family="primary_mlp",
                    seed=seed,
                    rounds=rounds,
                    tied=tied,
                    alpha=shared["alpha"],
                    beta=shared["beta"],
                    learning_rate=shared["learning_rate"],
                    output_dir=output_dir,
                    cell_id=cell_id,
                    event_path=event_path,
                    measurement_exposures=measure,
                    checkpoint_exposures=save,
                )
                training_cells.append({"cell_id": cell_id, **training})
                for measurement in measurements:
                    records.append(
                        {
                            "record_id": f"{cell_id}_E{measurement['exposure']}",
                            "kind": "primary_gamma",
                            "regime": regime,
                            "rounds": rounds,
                            "seed": seed,
                            **measurement,
                        }
                    )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    rounds = config["gamma_trajectory"]["primary_rounds"]
    trajectory_fits = []
    trajectory_intervals = []
    for exposure in checkpoints:
        fit = fit_records(records, rounds=rounds, regime="tied", exposure=exposure)
        trajectory_fits.append(fit)
        trajectory_intervals.append(
            {
                "exposure": exposure,
                **bootstrap_gamma(
                    records,
                    rounds=rounds,
                    regime="tied",
                    exposure=exposure,
                    seeds=shared["seeds"],
                    samples=config["gamma_trajectory"]["bootstrap_samples"],
                    bootstrap_seed=config["gamma_trajectory"]["bootstrap_seed"],
                    interval_mass=config["gamma_trajectory"]["interval_mass"],
                ),
            }
        )
    terminal = shared["progress_target"]
    tied_final = trajectory_fits[-1]
    untied_final = fit_records(records, rounds=rounds, regime="untied", exposure=terminal)
    r16_values = [
        row["kappa"]
        for row in records
        if row["regime"] == "tied" and row["rounds"] == 16 and row["exposure"] == terminal
    ]
    source_result = json.loads(
        _git_blob(config["source_v0"]["git_commit"], config["source_v0"]["gamma_result_path"])
    )
    source_means = source_result["summary"]["fits"]["tied"]["geometric_mean_kappa"]
    replication = {
        str(round_count): {
            "v0": source_means[str(round_count)],
            "v0_1": tied_final["geometric_mean_kappa"][str(round_count)],
            "ratio": tied_final["geometric_mean_kappa"][str(round_count)]
            / source_means[str(round_count)],
        }
        for round_count in config["r16_holdout"]["fit_rounds"]
    }
    summary = {
        "trajectory_fits": trajectory_fits,
        "trajectory_bootstrap_intervals": trajectory_intervals,
        "trajectory_classification": classify_gamma_trajectory(config, trajectory_fits),
        "r16_holdout": score_r16_holdout(config, geometric_mean(r16_values)),
        "terminal_four_point_tied_fit": tied_final,
        "terminal_four_point_untied_fit": untied_final,
        "terminal_gamma_contrast": tied_final["gamma"] - untied_final["gamma"],
        "gamma_one_envelope_below": trajectory_intervals[-1]["upper"] < 1.0,
        "v0_terminal_replication": replication,
        "training_cells": training_cells,
    }
    return _phase_result(config, config_hash, output_dir, "primary", records, summary)


def run_external(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    _require_phase(output_dir, "primary", config_hash)
    shared = config["shared_training"]
    external = config["external_validity"]
    terminal = shared["progress_target"]
    event_path = output_dir / "external_events.jsonl"
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for regime in external["regimes"]:
        tied = regime == "tied"
        for rounds in external["rounds"]:
            for seed in shared["seeds"]:
                cell_id = f"external_{regime}_R{rounds}_S{seed}"
                model, measurements, training = _train_cell(
                    config,
                    family="external_validity",
                    seed=seed,
                    rounds=rounds,
                    tied=tied,
                    alpha=shared["alpha"],
                    beta=shared["beta"],
                    learning_rate=shared["learning_rate"],
                    output_dir=output_dir,
                    cell_id=cell_id,
                    event_path=event_path,
                    measurement_exposures=[terminal],
                    checkpoint_exposures=[terminal],
                )
                training_cells.append({"cell_id": cell_id, **training})
                measurement = measurements[0]
                records.append(
                    {
                        "record_id": cell_id,
                        "kind": "external_gamma",
                        "regime": regime,
                        "rounds": rounds,
                        "seed": seed,
                        **measurement,
                    }
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    tied_fit = fit_records(
        records, rounds=external["rounds"], regime="tied", exposure=terminal
    )
    untied_fit = fit_records(
        records, rounds=external["rounds"], regime="untied", exposure=terminal
    )
    contrast = tied_fit["gamma"] - untied_fit["gamma"]
    minimum_r2 = config["stop_branches"]["minimum_power_law_r_squared"]
    summary = {
        "tied_fit": tied_fit,
        "untied_fit": untied_fit,
        "gamma_contrast": contrast,
        "replication_supported": (
            tied_fit["r_squared"] >= minimum_r2
            and untied_fit["r_squared"] >= minimum_r2
            and contrast >= 0.1
        ),
        "training_cells": training_cells,
    }
    return _phase_result(config, config_hash, output_dir, "external", records, summary)


def run_ladder(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    _require_phase(output_dir, "external", config_hash)
    shared = config["shared_training"]
    ladder = config["learning_rate_ladder"]
    event_path = output_dir / "ladder_events.jsonl"
    records: list[dict[str, Any]] = []
    for learning_rate in ladder["learning_rates"]:
        for rounds in ladder["heldout_rounds"]:
            for p in ladder["p_grid"]:
                beta = rounds ** (-2.0 * p)
                for seed in shared["seeds"]:
                    cell_id = f"ladder_LR{learning_rate:g}_R{rounds}_P{p:.2f}_S{seed}"
                    model, _, training = _train_cell(
                        config,
                        family="primary_mlp",
                        seed=seed,
                        rounds=rounds,
                        tied=True,
                        alpha=1.0,
                        beta=beta,
                        learning_rate=learning_rate,
                        output_dir=output_dir,
                        cell_id=cell_id,
                        event_path=event_path,
                    )
                    records.append(
                        {
                            "record_id": cell_id,
                            "kind": "learning_rate_ladder",
                            "learning_rate": learning_rate,
                            "rounds": rounds,
                            "p": p,
                            "alpha": 1.0,
                            "beta": beta,
                            "seed": seed,
                            "stable": stable_training(training),
                            "training": training,
                        }
                    )
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    outcomes = []
    ordering = []
    for learning_rate in ladder["learning_rates"]:
        by_round: dict[int, dict[str, Any]] = {}
        for rounds in ladder["heldout_rounds"]:
            cell_rows = [
                row
                for row in records
                if row["learning_rate"] == learning_rate and row["rounds"] == rounds
            ]
            boundary = empirical_boundary(cell_rows, ladder["p_grid"])
            by_round[rounds] = boundary
            outcomes.append({"learning_rate": learning_rate, "rounds": rounds, **boundary})
        first, second = ladder["heldout_rounds"]
        if all(by_round[r]["censoring"] == "exact_grid" for r in (first, second)):
            supported = by_round[second]["machine_boundary"] >= by_round[first]["machine_boundary"]
            status = "supported" if supported else "violated"
        else:
            supported = None
            status = "nonidentifying_censoring"
        ordering.append(
            {"learning_rate": learning_rate, "status": status, "supported": supported}
        )
    summary = {
        "outcomes": outcomes,
        "ordering": ordering,
        "all_cells_stable": all(row["stable"] for row in records),
        "stable_cells": sum(row["stable"] for row in records),
        "total_cells": len(records),
    }
    return _phase_result(config, config_hash, output_dir, "ladder", records, summary)


def finalize(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    phases = {
        phase: _require_phase(output_dir, phase, config_hash)
        for phase in ("source_replay", "primary", "external", "ladder")
    }
    resource_receipts = {}
    for phase in phases:
        path = output_dir / f"{phase}.resource_receipt.json"
        if not path.exists():
            raise RuntimeError(f"missing resource receipt: {path.name}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["status"] != "completed" or not receipt["cleanup_passed"]:
            raise RuntimeError(f"resource receipt did not pass: {phase}")
        resource_receipts[phase] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "status": receipt["status"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_ram_mb": receipt["peak_ram_mb"],
            "peak_io_mb_s": receipt["peak_io_mb_s"],
            "peak_vram_mb": receipt["peak_vram_mb"],
            "cleanup_passed": receipt["cleanup_passed"],
        }
    primary = phases["primary"]["summary"]
    external = phases["external"]["summary"]
    ladder = phases["ladder"]["summary"]
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "claim_scope": config["claim_scope"],
        "config": {
            "path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "sha256": config_hash,
            "registration_path": str(REGISTRATION.relative_to(ROOT)),
        },
        "execution_git_head": _git_head(),
        "phases": {
            name: {
                "result_path": str((output_dir / f"{name}_result.json").relative_to(ROOT)),
                "result_sha256": _sha256(output_dir / f"{name}_result.json"),
                "records_path": result["records_path"],
                "records_sha256": result["records_sha256"],
                "record_count": result["record_count"],
            }
            for name, result in phases.items()
        },
        "primary_findings": {
            "trajectory_classification": primary["trajectory_classification"],
            "terminal_tied_gamma": primary["terminal_four_point_tied_fit"]["gamma"],
            "terminal_tied_gamma_interval": primary["trajectory_bootstrap_intervals"][-1],
            "terminal_tied_r_squared": primary["terminal_four_point_tied_fit"]["r_squared"],
            "terminal_untied_gamma": primary["terminal_four_point_untied_fit"]["gamma"],
            "r16_holdout": primary["r16_holdout"],
            "below_gamma_one_envelope": primary["gamma_one_envelope_below"],
        },
        "external_validity": {
            "replication_supported": external["replication_supported"],
            "tied_gamma": external["tied_fit"]["gamma"],
            "untied_gamma": external["untied_fit"]["gamma"],
            "gamma_contrast": external["gamma_contrast"],
        },
        "learning_rate_ladder": {
            "all_cells_stable": ladder["all_cells_stable"],
            "ordering": ladder["ordering"],
            "outcomes": ladder["outcomes"],
        },
        "resources": resource_receipts,
        "nonclaim": "No task-performance, sample-efficiency, general Transformer, or general stability claim.",
    }
    _write_json(FINAL_RECEIPT, receipt)
    _write_json(output_dir / "result_receipt.json", receipt)
    return receipt


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("validate", "source_replay", "primary", "external", "ladder", "finalize"),
        required=True,
    )
    parser.add_argument("--source-checkpoint-dir", type=Path)
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config, config_hash = _load_registered_config(args.config.resolve())
    _set_vram_fraction(config, args.vram_fraction)
    output = args.output.resolve()
    try:
        if args.phase == "validate":
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "config_sha256": config_hash,
                        "source_blobs": _verify_source_blobs(config),
                    },
                    sort_keys=True,
                )
            )
        elif args.phase == "source_replay":
            if args.source_checkpoint_dir is None:
                raise ValueError("source_replay requires --source-checkpoint-dir")
            print(
                json.dumps(
                    run_source_replay(config, config_hash, output, args.source_checkpoint_dir.resolve()),
                    sort_keys=True,
                )
            )
        elif args.phase == "primary":
            print(json.dumps(run_primary(config, config_hash, output), sort_keys=True))
        elif args.phase == "external":
            print(json.dumps(run_external(config, config_hash, output), sort_keys=True))
        elif args.phase == "ladder":
            print(json.dumps(run_ladder(config, config_hash, output), sort_keys=True))
        else:
            print(json.dumps(finalize(config, config_hash, output), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
