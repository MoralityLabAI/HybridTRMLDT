"""Registered toy campaign for Loop Schedule Algebra v0."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable

import torch

from lsa.invariants import predicted_minimal_exponent
from lsa.kappa_probe import (
    ToyLoop,
    estimate_kappa,
    fit_power_law,
    geometric_mean,
    spread_ratio,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_algebra_v0.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_algebra_v0"
REGISTRATION = ROOT / "configs" / "loop_schedule_algebra_v0_registration.json"


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


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(payload))


def _load_registered_config(path: Path) -> tuple[dict[str, Any], str]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    actual_hash = _sha256(path)
    if actual_hash != registration["config_sha256"]:
        raise RuntimeError(
            f"registered config hash mismatch: expected {registration['config_sha256']}, got {actual_hash}"
        )
    return json.loads(path.read_text(encoding="utf-8")), actual_hash


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _device(config: dict[str, Any]) -> torch.device:
    requested = config["resources"]["device"]
    if requested != "cuda_if_available_else_cpu":
        raise ValueError(f"unsupported registered device policy: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _task_batch(
    seed: int,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    *,
    stream: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + stream)
    inputs = torch.randn(batch_size, hidden_size, generator=cpu_generator)
    permutation = torch.randperm(hidden_size, generator=torch.Generator().manual_seed(seed + 17))
    signs = torch.where(
        torch.rand(hidden_size, generator=torch.Generator().manual_seed(seed + 29)) > 0.5,
        1.0,
        -1.0,
    )
    targets = torch.tanh(inputs[:, permutation] * signs)
    return inputs.to(device), targets.to(device)


def _train_cell(
    config: dict[str, Any],
    *,
    seed: int,
    rounds: int,
    tied: bool,
    alpha: float,
    beta: float,
    output_dir: Path,
    cell_id: str,
    event_path: Path,
) -> tuple[ToyLoop, dict[str, Any]]:
    toy = config["toy"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config)
    model = ToyLoop(
        toy["hidden_size"], rounds, tied=tied, alpha=alpha, beta=beta
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=toy["learning_rate"],
        weight_decay=toy["weight_decay"],
    )
    exposures_per_step = toy["batch_size"] * rounds
    steps = math.ceil(toy["progress_target"] / exposures_per_step)
    checkpoint_every = int(toy["checkpoint_every_steps"])
    initial_loss: float | None = None
    final_loss = math.nan
    max_gradient_norm = 0.0
    nonfinite = False
    checkpoint_paths: list[str] = []
    _append_event(
        event_path,
        {"event": "cell_start", "cell_id": cell_id, "steps": steps, "device": str(device)},
    )
    for step in range(steps):
        inputs, targets = _task_batch(
            seed,
            toy["batch_size"],
            toy["hidden_size"],
            device,
            stream=step,
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
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            checkpoint = output_dir / "checkpoints" / f"{cell_id}_step_{step + 1}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "cell_id": cell_id,
                    "step": step + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint,
            )
            checkpoint_paths.append(str(checkpoint.relative_to(ROOT)))
            _append_event(
                event_path,
                {"event": "checkpoint", "cell_id": cell_id, "step": step + 1, "path": checkpoint_paths[-1]},
            )
    completed_steps = step + 1
    metrics = {
        "steps": completed_steps,
        "state_visit_exposures": completed_steps * exposures_per_step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "max_gradient_norm": max_gradient_norm,
        "nonfinite": nonfinite,
        "checkpoints": checkpoint_paths,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    _append_event(event_path, {"event": "cell_complete", "cell_id": cell_id, **metrics})
    return model, metrics


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda row: row["record_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in ordered:
            handle.write(_canonical_bytes(record))


def _gamma_summary(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    probe = config["kappa_probe"]
    stop = config["stop_branches"]
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        if record["kind"] == "gamma":
            by_cell[(record["regime"], record["rounds"])].append(record["kappa"])
    spreads = {
        f"{regime}_R{rounds}": spread_ratio(values)
        for (regime, rounds), values in sorted(by_cell.items())
    }
    unstable = [key for key, value in spreads.items() if value > stop["max_across_seed_kappa_spread"]]
    summary: dict[str, Any] = {"spreads": spreads, "unstable_cells": unstable, "fits": {}}
    if unstable:
        summary["status"] = "estimator_unstable"
        return summary
    poor_fits: list[str] = []
    for regime in probe["regimes"]:
        kappa_means = [
            geometric_mean(by_cell[(regime, rounds)]) for rounds in probe["rounds"]
        ]
        fit = fit_power_law(probe["rounds"], kappa_means)
        summary["fits"][regime] = {
            **fit.to_dict(),
            "geometric_mean_kappa": dict(zip(map(str, probe["rounds"]), kappa_means)),
        }
        if fit.r_squared < stop["minimum_power_law_r_squared"]:
            poor_fits.append(regime)
    summary["poor_fits"] = poor_fits
    summary["status"] = "poor_power_law" if poor_fits else "fit_complete"
    p1 = config["p1"]
    summary["p1"] = {
        "tied_confirmed": p1["tied_gamma_interval"][0]
        <= summary["fits"]["tied"]["gamma"]
        <= p1["tied_gamma_interval"][1],
        "untied_confirmed": p1["untied_gamma_interval"][0]
        <= summary["fits"]["untied"]["gamma"]
        <= p1["untied_gamma_interval"][1],
    }
    p3 = {
        record["mask"]: record["kappa"]
        for record in records
        if record["kind"] == "mask" and record["seed"] == probe["seeds"][0]
    }
    summary["p3_first_seed"] = {
        "kappa_by_mask": p3,
        "confirmed": bool(p3)
        and p3["one_step"] <= p3["last_k"] <= p3["full"],
    }
    return summary


def run_gamma(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "events.jsonl"
    records: list[dict[str, Any]] = []
    probe = config["kappa_probe"]
    for regime in probe["regimes"]:
        tied = regime == "tied"
        for rounds in probe["rounds"]:
            for seed in probe["seeds"]:
                cell_id = f"gamma_{regime}_R{rounds}_S{seed}"
                model, training = _train_cell(
                    config,
                    seed=seed,
                    rounds=rounds,
                    tied=tied,
                    alpha=probe["alpha"],
                    beta=probe["beta"],
                    output_dir=output_dir,
                    cell_id=cell_id,
                    event_path=event_path,
                )
                inputs, targets = _task_batch(
                    seed,
                    config["toy"]["measurement_batch_size"],
                    config["toy"]["hidden_size"],
                    _device(config),
                    stream=999_001,
                )
                estimate = estimate_kappa(
                    model,
                    inputs,
                    targets,
                    power_iterations=probe["power_iterations"],
                    seed=seed + 50_000,
                )
                records.append(
                    {
                        "record_id": cell_id,
                        "kind": "gamma",
                        "regime": regime,
                        "rounds": rounds,
                        "seed": seed,
                        "kappa": estimate.kappa,
                        "estimate": estimate.to_dict(),
                        "training": training,
                    }
                )
                if tied and rounds == config["p3"]["rounds"]:
                    for mask, visible_count in config["p3"]["masks"].items():
                        masked = estimate_kappa(
                            model,
                            inputs,
                            targets,
                            visible_visits=range(rounds - visible_count, rounds),
                            power_iterations=probe["power_iterations"],
                            seed=seed + 60_000,
                        )
                        records.append(
                            {
                                "record_id": f"mask_{mask}_R{rounds}_S{seed}",
                                "kind": "mask",
                                "mask": mask,
                                "rounds": rounds,
                                "visible_rounds": visible_count,
                                "seed": seed,
                                "kappa": masked.kappa,
                                "estimate": masked.to_dict(),
                            }
                        )
                del model, inputs, targets
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    records_path = output_dir / "gamma_records.jsonl"
    _write_records(records_path, records)
    summary = _gamma_summary(config, records)
    result = {
        "protocol_id": config["protocol_id"],
        "phase": "gamma",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(records),
        "summary": summary,
    }
    _write_json(output_dir / "gamma_result.json", result)
    return result


def seal_predictions(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    gamma_path = output_dir / "gamma_result.json"
    gamma = json.loads(gamma_path.read_text(encoding="utf-8"))
    if gamma["config_sha256"] != config_hash or gamma["summary"]["status"] != "fit_complete":
        raise RuntimeError("gamma phase did not clear registered stop branches")
    tied_gamma = gamma["summary"]["fits"]["tied"]["gamma"]
    prediction = predicted_minimal_exponent({"tied": tied_gamma})
    payload = {
        "protocol_id": config["protocol_id"],
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "gamma_result_sha256": _sha256(gamma_path),
        "gamma_binding": tied_gamma,
        "predicted_p_star": prediction,
        "heldout": [
            {"rounds": rounds, "predicted_boundary": prediction}
            for rounds in config["p2"]["heldout_rounds"]
        ],
        "outcomes_observed": False,
    }
    prediction_path = output_dir / "p2_predictions.json"
    _write_json(prediction_path, payload)
    return payload


def _is_stable(config: dict[str, Any], training: dict[str, Any]) -> bool:
    initial = training["initial_loss"]
    final = training["final_loss"]
    return bool(
        not training["nonfinite"]
        and math.isfinite(final)
        and training["max_gradient_norm"] <= 100.0
        and initial is not None
        and final / initial <= 10.0
    )


def _empirical_boundary(config: dict[str, Any], rows: list[dict[str, Any]]) -> float | None:
    grid = config["p2"]["p_grid"]
    by_p: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        by_p[float(row["p"])].append(bool(row["stable"]))
    stable = {p: sum(by_p[p]) >= 2 for p in grid}
    for index, p in enumerate(grid):
        if all(stable[later] for later in grid[index:]):
            return p
    return None


def run_boundary(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    prediction_path = output_dir / "p2_predictions.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction["config_sha256"] != config_hash or prediction["outcomes_observed"]:
        raise RuntimeError("invalid or post-outcome prediction file")
    event_path = output_dir / "boundary_events.jsonl"
    records: list[dict[str, Any]] = []
    for rounds in config["p2"]["heldout_rounds"]:
        for p in config["p2"]["p_grid"]:
            beta = rounds ** (-2.0 * p)
            for seed in config["kappa_probe"]["seeds"]:
                cell_id = f"boundary_R{rounds}_P{p:.2f}_S{seed}"
                model, training = _train_cell(
                    config,
                    seed=seed,
                    rounds=rounds,
                    tied=True,
                    alpha=1.0,
                    beta=beta,
                    output_dir=output_dir,
                    cell_id=cell_id,
                    event_path=event_path,
                )
                records.append(
                    {
                        "record_id": cell_id,
                        "kind": "boundary",
                        "rounds": rounds,
                        "p": p,
                        "alpha": 1.0,
                        "beta": beta,
                        "seed": seed,
                        "stable": _is_stable(config, training),
                        "training": training,
                    }
                )
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    records_path = output_dir / "boundary_records.jsonl"
    _write_records(records_path, records)
    outcomes = []
    tolerance = config["p2"]["confirmation_tolerance"]
    for heldout in prediction["heldout"]:
        cell_rows = [row for row in records if row["rounds"] == heldout["rounds"]]
        observed = _empirical_boundary(config, cell_rows)
        confirmed = observed is not None and abs(observed - heldout["predicted_boundary"]) <= tolerance + 1e-12
        outcomes.append({**heldout, "empirical_boundary": observed, "confirmed": confirmed})
    result = {
        "protocol_id": config["protocol_id"],
        "phase": "boundary",
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "predictions_sha256": _sha256(prediction_path),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(records),
        "outcomes": outcomes,
        "p2_confirmed": all(outcome["confirmed"] for outcome in outcomes),
        "headline": "boundary_transfer" if all(outcome["confirmed"] for outcome in outcomes) else "boundary_nontransfer",
    }
    _write_json(output_dir / "boundary_result.json", result)
    return result


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
    parser.add_argument("--phase", choices=("validate", "gamma", "seal", "boundary"), required=True)
    args = parser.parse_args()
    config, config_hash = _load_registered_config(args.config.resolve())
    try:
        if args.phase == "validate":
            print(json.dumps({"status": "valid", "config_sha256": config_hash}, sort_keys=True))
        elif args.phase == "gamma":
            print(json.dumps(run_gamma(config, config_hash, args.output.resolve()), sort_keys=True))
        elif args.phase == "seal":
            print(json.dumps(seal_predictions(config, config_hash, args.output.resolve()), sort_keys=True))
        else:
            print(json.dumps(run_boundary(config, config_hash, args.output.resolve()), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
