"""Sealed R32-fit/R64-holdout extension of the LSA v0.1 campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import torch

from lsa.kappa_probe import geometric_mean, spread_ratio
from research_gym.analysis.lsa_saturation import (
    bootstrap_predictions,
    choose_holdout_model,
    fit_registered_models,
    geometric_means_from_records,
    legacy_power_law,
    local_effective_gammas,
    predict_curve,
    score_saturation_support,
)
from research_gym.analysis.lsa_v0_1 import bootstrap_gamma, fit_records
from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
    ROOT,
    _append_event,
    _cleanup,
    _git_head,
    _load_registered_config as load_parent_config,
    _set_vram_fraction,
    _sha256,
    _train_cell,
    _write_json,
    _write_records,
)


DEFAULT_CONFIG = ROOT / "configs" / "loop_schedule_algebra_saturation_addendum_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "loop_schedule_algebra_saturation_addendum_v1"
REGISTRATION = (
    ROOT / "configs" / "loop_schedule_algebra_saturation_addendum_v1_registration.json"
)
FINAL_RECEIPT = ROOT / "data" / "benchmarks" / "lsa_saturation_addendum_v1_receipt.json"


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise RuntimeError(f"registered path escapes repository: {value}")
    return path


def load_registered_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    config_hash = _sha256(path)
    if config_hash != registration["config_sha256"]:
        raise RuntimeError(
            f"registered addendum hash mismatch: expected {registration['config_sha256']}, "
            f"got {config_hash}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != registration["protocol_id"]:
        raise RuntimeError("addendum protocol id does not match registration")
    parent_path = _root_path(config["parent_v0_1"]["config_path"])
    parent, parent_hash = load_parent_config(parent_path)
    if parent_hash != config["parent_v0_1"]["config_sha256"]:
        raise RuntimeError("parent config hash does not match addendum")
    for name in ("receipt", "primary_records"):
        source_path = _root_path(config["parent_v0_1"][f"{name}_path"])
        expected = config["parent_v0_1"][f"{name}_sha256"]
        if _sha256(source_path) != expected:
            raise RuntimeError(f"parent {name} hash does not match addendum")
    return config, config_hash, parent


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _phase_result(
    config: dict[str, Any],
    config_hash: str,
    output_dir: Path,
    phase: str,
    records: Iterable[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    rows = list(records)
    records_path = output_dir / f"{phase}_records.jsonl"
    _write_records(records_path, rows)
    result = {
        "protocol_id": config["protocol_id"],
        "phase": phase,
        "config_sha256": config_hash,
        "git_head": _git_head(),
        "records_path": str(records_path.relative_to(ROOT)),
        "records_sha256": _sha256(records_path),
        "record_count": len(rows),
        "summary": summary,
    }
    _write_json(output_dir / f"{phase}_result.json", result)
    return result


def _verify_parent_means(config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    parent = config["parent_v0_1"]
    rounds = [int(value) for value in parent["source_rounds"]]
    means, _ = geometric_means_from_records(
        records,
        rounds=rounds,
        regime="tied",
        exposure=int(parent["source_exposure"]),
    )
    expected = parent["source_terminal_tied_kappa"]
    for round_count, actual in zip(rounds, means):
        if not math.isclose(actual, float(expected[str(round_count)]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"sealed parent kappa mismatch at R={round_count}")


def validate(config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    parent_records_path = _root_path(config["parent_v0_1"]["primary_records_path"])
    records = _read_records(parent_records_path)
    _verify_parent_means(config, records)
    return {
        "status": "valid",
        "config_sha256": config_hash,
        "parent_primary_records_sha256": _sha256(parent_records_path),
        "r32_or_r64_records_present": False,
    }


def _new_round_records(
    config: dict[str, Any],
    parent: dict[str, Any],
    output_dir: Path,
    *,
    rounds: int,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = config["frozen_training"]
    shared = parent["shared_training"]
    event_path = output_dir / f"{phase}_events.jsonl"
    records: list[dict[str, Any]] = []
    training_cells: list[dict[str, Any]] = []
    for regime in frozen["regimes"]:
        tied = regime == "tied"
        for seed in frozen["seeds"]:
            cell_id = f"saturation_{regime}_R{rounds}_S{seed}"
            model, measurements, training = _train_cell(
                parent,
                family="primary_mlp",
                seed=int(seed),
                rounds=rounds,
                tied=tied,
                alpha=float(frozen["alpha"]),
                beta=float(frozen["beta"]),
                learning_rate=float(frozen["learning_rate"]),
                output_dir=output_dir,
                cell_id=cell_id,
                event_path=event_path,
                measurement_exposures=[int(frozen["measurement_exposure"])],
                checkpoint_exposures=[int(frozen["checkpoint_exposure"])],
            )
            if training["nonfinite"]:
                raise RuntimeError(f"nonfinite training in {cell_id}")
            if training["state_visit_exposures"] != frozen["progress_target"]:
                raise RuntimeError(f"exposure mismatch in {cell_id}")
            measurement = measurements[0]
            records.append(
                {
                    "record_id": f"{cell_id}_E{measurement['exposure']}",
                    "kind": "saturation_gamma",
                    "regime": regime,
                    "rounds": rounds,
                    "seed": seed,
                    **measurement,
                }
            )
            training_cells.append({"cell_id": cell_id, **training})
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records, training_cells


def run_fit_r32(
    config: dict[str, Any], config_hash: str, parent: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    if (output_dir / "holdout_r64_records.jsonl").exists():
        raise RuntimeError("R64 records already exist; fit phase cannot be rerun")
    parent_records = _read_records(_root_path(config["parent_v0_1"]["primary_records_path"]))
    _verify_parent_means(config, parent_records)
    new_records, training_cells = _new_round_records(
        config,
        parent,
        output_dir,
        rounds=int(config["rounds"]["new_fit_round"]),
        phase="fit_r32",
    )
    combined = parent_records + new_records
    rounds = [int(value) for value in config["rounds"]["fit"]]
    exposure = int(config["frozen_training"]["measurement_exposure"])
    kappas, values = geometric_means_from_records(
        combined, rounds=rounds, regime="tied", exposure=exposure
    )
    maximum_spread = max(spread_ratio(values[round_count]) for round_count in rounds)
    if maximum_spread > float(config["stop_branches"]["maximum_across_seed_kappa_spread"]):
        raise RuntimeError("registered estimator-spread stop fired")
    fits = fit_registered_models(rounds, kappas, config)
    holdout_round = int(config["rounds"]["untouched_holdout_round"])
    predictions = {name: predict_curve(fit, holdout_round) for name, fit in fits.items()}
    intervals = bootstrap_predictions(
        values,
        rounds=rounds,
        prediction_round=holdout_round,
        config=config,
    )
    prediction_path = _root_path(config["prediction_seal"]["path"])
    prediction = {
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "generated_git_head": _git_head(),
        "fit_rounds": rounds,
        "holdout_round": holdout_round,
        "parent_records_sha256": config["parent_v0_1"]["primary_records_sha256"],
        "r32_records_path": str((output_dir / "fit_r32_records.jsonl").relative_to(ROOT)),
        "geometric_mean_kappa": dict(zip(map(str, rounds), kappas)),
        "candidate_fits": {name: fit.to_dict() for name, fit in fits.items()},
        "r64_predictions": predictions,
        "r64_prediction_intervals": intervals,
        "local_effective_gamma": local_effective_gammas(rounds, kappas),
        "legacy_power_law_diagnostic": legacy_power_law(rounds, kappas),
        "holdout_observed": False,
    }
    # The prediction references the records path; write records first, then seal prediction bytes.
    _write_records(output_dir / "fit_r32_records.jsonl", new_records)
    prediction["r32_records_sha256"] = _sha256(output_dir / "fit_r32_records.jsonl")
    _write_json(prediction_path, prediction)
    summary = {
        "training_cells": training_cells,
        "geometric_mean_kappa": prediction["geometric_mean_kappa"],
        "candidate_fits": prediction["candidate_fits"],
        "r64_predictions": predictions,
        "r64_prediction_intervals": intervals,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": _sha256(prediction_path),
        "maximum_seed_spread_ratio": maximum_spread,
    }
    return _phase_result(config, config_hash, output_dir, "fit_r32", new_records, summary)


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def verify_prediction_seal(config: dict[str, Any], config_hash: str) -> dict[str, str]:
    prediction_path = _root_path(config["prediction_seal"]["path"])
    relative = prediction_path.relative_to(ROOT).as_posix()
    if not prediction_path.exists():
        raise RuntimeError("registered R64 prediction artifact is missing")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction["config_sha256"] != config_hash or prediction["holdout_observed"]:
        raise RuntimeError("R64 prediction artifact has invalid provenance state")
    _git("ls-files", "--error-unmatch", "--", relative)
    if _git("status", "--porcelain", "--", relative):
        raise RuntimeError("R64 prediction artifact is not clean")
    commit = _git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError("R64 prediction artifact has no commit")
    committed_bytes = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    committed_hash = hashlib.sha256(committed_bytes).hexdigest()
    if committed_hash != _sha256(prediction_path):
        raise RuntimeError("committed R64 prediction bytes differ from working tree")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT)
    branch = _git("branch", "--show-current")
    remote_line = _git("ls-remote", "--heads", "origin", f"refs/heads/{branch}")
    if not remote_line:
        raise RuntimeError("tracked remote branch is missing")
    remote_head = remote_line.split()[0]
    if remote_head != _git("rev-parse", "HEAD"):
        raise RuntimeError("local HEAD must equal remote branch HEAD before R64")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, remote_head], cwd=ROOT)
    return {
        "path": relative,
        "sha256": _sha256(prediction_path),
        "commit": commit,
        "branch": branch,
        "remote_head": remote_head,
    }


def _require_phase(output_dir: Path, phase: str, config_hash: str) -> dict[str, Any]:
    result_path = output_dir / f"{phase}_result.json"
    if not result_path.exists():
        raise RuntimeError(f"required phase is missing: {phase}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["config_sha256"] != config_hash:
        raise RuntimeError(f"required phase config mismatch: {phase}")
    records_path = _root_path(result["records_path"])
    if _sha256(records_path) != result["records_sha256"]:
        raise RuntimeError(f"required phase records failed rehash: {phase}")
    return result


def run_holdout_r64(
    config: dict[str, Any], config_hash: str, parent: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    seal = verify_prediction_seal(config, config_hash)
    if (output_dir / "holdout_r64_records.jsonl").exists():
        raise RuntimeError("R64 holdout records already exist")
    fit_result = _require_phase(output_dir, "fit_r32", config_hash)
    prediction = json.loads(_root_path(config["prediction_seal"]["path"]).read_text())
    if fit_result["summary"]["prediction_sha256"] != seal["sha256"]:
        raise RuntimeError("fit result does not bind the committed prediction")
    new_records, training_cells = _new_round_records(
        config,
        parent,
        output_dir,
        rounds=int(config["rounds"]["untouched_holdout_round"]),
        phase="holdout_r64",
    )
    parent_records = _read_records(_root_path(config["parent_v0_1"]["primary_records_path"]))
    r32_records = _read_records(output_dir / "fit_r32_records.jsonl")
    combined = parent_records + r32_records + new_records
    rounds = [2, 4, 8, 16, 32, 64]
    exposure = int(config["frozen_training"]["measurement_exposure"])
    tied_kappas, _ = geometric_means_from_records(
        combined, rounds=rounds, regime="tied", exposure=exposure
    )
    untied_kappas, _ = geometric_means_from_records(
        combined, rounds=rounds, regime="untied", exposure=exposure
    )
    tied_by_round = dict(zip(rounds, tied_kappas))
    predictions = {
        name: float(value) for name, value in prediction["r64_predictions"].items()
    }
    decision = choose_holdout_model(tied_by_round[64], predictions, config)
    fit_rounds = [int(value) for value in config["rounds"]["fit"]]
    refits = fit_registered_models(fit_rounds, tied_kappas[:-1], config)
    saturation = score_saturation_support(
        winner=decision["winner"],
        saturating_fit=refits["saturating_exponential"],
        kappas=tied_by_round,
        config=config,
    )
    flat = config["untied_flatness_control"]
    point_untied = fit_records(
        combined, rounds=rounds, regime="untied", exposure=exposure
    )
    interval_untied = bootstrap_gamma(
        combined,
        rounds=rounds,
        regime="untied",
        exposure=exposure,
        seeds=[int(value) for value in config["frozen_training"]["seeds"]],
        samples=int(flat["bootstrap_samples"]),
        bootstrap_seed=int(flat["bootstrap_seed"]),
        interval_mass=float(flat["interval_mass"]),
    )
    flatness_passed = (
        float(interval_untied["lower"]) <= 0.0 <= float(interval_untied["upper"])
        and abs(float(point_untied["gamma"])) <= 0.1
    )
    if saturation["supported"]:
        classification = "saturation_supported"
    elif decision["winner"] == "logarithmic":
        classification = "logarithmic_growth_supported"
    else:
        classification = "form_unresolved"
    summary = {
        "prediction_seal": seal,
        "training_cells": training_cells,
        "tied_geometric_mean_kappa": dict(zip(map(str, rounds), tied_kappas)),
        "untied_geometric_mean_kappa": dict(zip(map(str, rounds), untied_kappas)),
        "sealed_predictions": predictions,
        "holdout_decision": decision,
        "saturation_support": saturation,
        "classification": classification,
        "local_effective_gamma": local_effective_gammas(rounds, tied_kappas),
        "legacy_power_law_diagnostic": legacy_power_law(rounds, tied_kappas),
        "untied_flatness_control": {
            "point_fit": point_untied,
            "bootstrap_interval": interval_untied,
            "passed": flatness_passed,
        },
    }
    return _phase_result(config, config_hash, output_dir, "holdout_r64", new_records, summary)


def _resource_receipt(output_dir: Path, phase: str) -> dict[str, Any]:
    path = output_dir / f"{phase}.resource_receipt.json"
    if not path.exists():
        raise RuntimeError(f"missing resource receipt: {phase}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["status"] != "completed" or not value["cleanup_passed"]:
        raise RuntimeError(f"resource or cleanup failure: {phase}")
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), **value}


def finalize(config: dict[str, Any], config_hash: str, output_dir: Path) -> dict[str, Any]:
    fit = _require_phase(output_dir, "fit_r32", config_hash)
    holdout = _require_phase(output_dir, "holdout_r64", config_hash)
    resources = {
        phase: _resource_receipt(output_dir, phase) for phase in ("fit_r32", "holdout_r64")
    }
    receipt = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "config": {
            "path": str(DEFAULT_CONFIG.relative_to(ROOT)),
            "registration_path": str(REGISTRATION.relative_to(ROOT)),
            "sha256": config_hash,
        },
        "execution_git_head": _git_head(),
        "prediction_seal": holdout["summary"]["prediction_seal"],
        "phases": {
            "fit_r32": {
                "result_path": str((output_dir / "fit_r32_result.json").relative_to(ROOT)),
                "result_sha256": _sha256(output_dir / "fit_r32_result.json"),
                "records_path": fit["records_path"],
                "records_sha256": fit["records_sha256"],
                "record_count": fit["record_count"],
            },
            "holdout_r64": {
                "result_path": str((output_dir / "holdout_r64_result.json").relative_to(ROOT)),
                "result_sha256": _sha256(output_dir / "holdout_r64_result.json"),
                "records_path": holdout["records_path"],
                "records_sha256": holdout["records_sha256"],
                "record_count": holdout["record_count"],
            },
        },
        "classification": holdout["summary"]["classification"],
        "holdout_decision": holdout["summary"]["holdout_decision"],
        "saturation_support": holdout["summary"]["saturation_support"],
        "tied_geometric_mean_kappa": holdout["summary"]["tied_geometric_mean_kappa"],
        "untied_flatness_control": holdout["summary"]["untied_flatness_control"],
        "resources": resources,
        "claim_scope": config["claim_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(output_dir / "result_receipt.json", receipt)
    _write_json(FINAL_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("validate", "fit_r32", "holdout_r64", "finalize"),
        required=True,
    )
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config, config_hash, parent = load_registered_config(args.config.resolve())
    _set_vram_fraction(parent, args.vram_fraction)
    output = args.output.resolve()
    try:
        if args.phase == "validate":
            print(json.dumps(validate(config, config_hash), sort_keys=True))
        elif args.phase == "fit_r32":
            print(json.dumps(run_fit_r32(config, config_hash, parent, output), sort_keys=True))
        elif args.phase == "holdout_r64":
            print(
                json.dumps(
                    run_holdout_r64(config, config_hash, parent, output), sort_keys=True
                )
            )
        else:
            print(json.dumps(finalize(config, config_hash, output), sort_keys=True))
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
