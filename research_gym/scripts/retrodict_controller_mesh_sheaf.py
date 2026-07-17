from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from typing import Mapping

from research_gym.analysis.controller_mesh_sheaf import (
    analyze_policy_sheaf,
    canonical_sha256,
    effective_policy_index,
    gate_margin_diagnostic,
    kernel_migration,
    markdown_report,
    matched_null_association,
    prediction_result,
    probe_sha256,
    summarize_policy_outcomes,
    validate_frozen_config,
)
from research_gym.benchmarks.gaming_vs_improvement_bench import (
    GamingBenchmarkConfig,
    _fit_probe,
    _model_sha256,
    _undertrained_base,
    build_region_heldout_examples,
)
from research_gym.scripts.bench_gaming_vs_improvement import _write_json


DEFAULT_CONFIG = Path("configs/controller_mesh_sheaf_retrodiction_v1.json")
DEFAULT_OUTPUT = Path("data/benchmarks/controller_mesh_sheaf_retrodiction_v1.json")
DEFAULT_REPORT = Path("reports/controller_mesh_sheaf_retrodiction_v1.md")
DEFAULT_EXPERIMENT = Path("experiments/controller_mesh_sheaf_retrodiction_v1")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, expected: str) -> str:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {path}: expected={expected} actual={actual}")
    return actual


def _parse_arm(arm_id: str) -> tuple[str, str, str, int]:
    evidence, rejection, adaptation, seed_token = arm_id.split("__")
    if not seed_token.startswith("s"):
        raise ValueError(f"invalid arm seed: {arm_id}")
    return evidence, rejection, adaptation, int(seed_token[1:])


def _load_registered_records(
    path: Path,
    *,
    expert_rounds: int,
    kernel_arm: str,
    kernel_rounds: set[int],
) -> tuple[dict[str, list[dict[str, object]]], dict[int, dict[int, list[dict[str, object]]]], int]:
    final_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    kernel_records: dict[int, dict[int, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if not bool(row["integrity_ok"]):
                raise ValueError(f"integrity failure in {row['receipt_sha256']}")
            arm_id = str(row["arm_id"])
            evidence, rejection, adaptation, seed = _parse_arm(arm_id)
            final_round = 0 if adaptation == "frozen" else expert_rounds
            if int(row["round"]) == final_round:
                final_records[arm_id].append(row)
            if (
                "__".join((evidence, rejection, adaptation)) == kernel_arm
                and int(row["round"]) in kernel_rounds
            ):
                kernel_records[seed][int(row["round"])].append(row)
    return dict(final_records), kernel_records, count


def _topology_contrast(
    policy_rows: list[dict[str, object]],
    *,
    evidence_source: str,
) -> dict[str, float]:
    selected = [
        row
        for row in policy_rows
        if row["evidence_source"] == evidence_source
        and row["rejection_action"] == "state_conditioned_fallback"
        and row["adaptation"] == "expert_iterated"
    ]
    if not selected:
        raise ValueError(f"missing topology contrast for {evidence_source}")
    return {
        "spectral_gap": mean(float(row["spectrum"]["spectral_gap"]) for row in selected),
        "low_band_rank": mean(float(row["spectrum"]["low_band_rank"]) for row in selected),
        "slow_mode_rank": mean(float(row["spectrum"]["slow_mode_rank"]) for row in selected),
        "utility_delta_vs_proposal": mean(
            float(row["outcomes"]["utility_delta_vs_proposal"]) for row in selected
        ),
        "action_change_rate": mean(
            float(row["outcomes"]["action_change_rate"]) for row in selected
        ),
    }


def _kernel_rows(
    registration: Mapping[str, object],
    benchmark_registration: Mapping[str, object],
    kernel_records: Mapping[int, Mapping[int, list[dict[str, object]]]],
) -> list[dict[str, object]]:
    kernel_config = registration["kernel_migration"]
    benchmark_config = GamingBenchmarkConfig.from_registration(benchmark_registration)
    round0 = int(kernel_config["round0"])
    final_round = int(kernel_config["final_round"])
    headline = json.loads(
        Path(registration["source"]["result"]).read_text(encoding="utf-8")
    )["headline_exposed_probe"]["trajectories"]
    evasion_by_seed = {int(row["seed"]): bool(row["evasion_pattern"]) for row in headline}
    output = []
    for seed in benchmark_config.seeds:
        regions = build_region_heldout_examples(benchmark_config, seed=seed)
        base_model, _ = _undertrained_base(benchmark_config, regions, seed=seed)
        gate_examples = list(regions["probe_calibration"])[::2]
        exposed = _fit_probe(
            base_model,
            gate_examples,
            benchmark_config,
            seed=seed + 2000,
        )
        initial = kernel_records[seed][round0]
        final = kernel_records[seed][final_round]
        recorded_hashes = {str(row["model_sha256"]) for row in initial}
        replayed_model_hash = _model_sha256(base_model)
        if recorded_hashes != {replayed_model_hash}:
            raise ValueError(
                f"replayed base model hash does not match seed {seed}: "
                f"records={sorted(recorded_hashes)} replay={replayed_model_hash}"
            )
        row = kernel_migration(
            initial,
            final,
            exposed.probe.weight,
            property_rank=int(kernel_config["property_rank"]),
            null_replicates=int(kernel_config["matched_random_directions"]),
            null_seed=int(kernel_config["base_seed"]) + seed,
        )
        row["posthoc_gate_margin"] = gate_margin_diagnostic(
            initial,
            final,
            exposed.probe.weight,
            bias=exposed.probe.bias,
            threshold=exposed.probe.threshold,
        )
        row.update(
            {
                "seed": seed,
                "evasion_label_from_source": evasion_by_seed[seed],
                "round0_model_sha256": replayed_model_hash,
                "exposed_probe_sha256": probe_sha256(
                    exposed.probe.weight,
                    bias=exposed.probe.bias,
                    threshold=exposed.probe.threshold,
                ),
                "exposed_probe_calibration_auroc": exposed.auroc,
                "round0_records": len(initial),
                "final_records": len(final),
            }
        )
        output.append(row)
    return output


def run_retrodiction(registration: Mapping[str, object]) -> dict[str, object]:
    config_sha256 = validate_frozen_config(registration)
    source = registration["source"]
    result_path = Path(source["result"])
    records_path = Path(source["records"])
    result_sha256 = _verify_source(result_path, str(source["result_sha256"]))
    records_sha256 = _verify_source(records_path, str(source["records_sha256"]))
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    benchmark_registration = json.loads(
        Path(source["benchmark_config"]).read_text(encoding="utf-8")
    )
    benchmark_config = GamingBenchmarkConfig.from_registration(benchmark_registration)
    kernel_config = registration["kernel_migration"]
    final_records, kernel_records, record_count = _load_registered_records(
        records_path,
        expert_rounds=benchmark_config.expert_iteration_rounds,
        kernel_arm=str(kernel_config["arm"]),
        kernel_rounds={int(kernel_config["round0"]), int(kernel_config["final_round"])},
    )
    if record_count != int(source["expected_record_count"]):
        raise ValueError(f"record count mismatch: expected={source['expected_record_count']} actual={record_count}")
    if len(final_records) != int(source["expected_policy_instances"]):
        raise ValueError(
            f"policy count mismatch: expected={source['expected_policy_instances']} "
            f"actual={len(final_records)}"
        )
    arm_to_group, class_summaries = effective_policy_index(source_result)
    if len(class_summaries) != int(source["expected_behavioral_classes"]):
        raise ValueError(
            f"behavioral class mismatch: expected={source['expected_behavioral_classes']} "
            f"actual={len(class_summaries)}"
        )

    construction = registration["construction"]
    null_config = registration["matched_null"]
    policy_rows = []
    policy_nulls: dict[str, dict[str, list[float]]] = {}
    for arm_id, rows in sorted(final_records.items()):
        evidence, rejection, adaptation, seed = _parse_arm(arm_id)
        stable_arm_seed = int(sha256(arm_id.encode("utf-8")).hexdigest()[:8], 16)
        spectrum, null_values = analyze_policy_sheaf(
            rows,
            evidence_source=evidence,
            energy_fraction=float(construction["centered_energy_fraction"]),
            max_rank=int(construction["max_centered_rank_per_stalk"]),
            singular_tolerance=float(construction["singular_tolerance"]),
            zero_tolerance=float(construction["zero_tolerance"]),
            low_band_max=float(construction["low_band_max"]),
            slow_band_max=float(construction["slow_band_max"]),
            null_replicates=int(null_config["replicates"]),
            null_seed=int(null_config["base_seed"]) + stable_arm_seed,
        )
        policy_rows.append(
            {
                "arm_id": arm_id,
                "effective_policy_id": arm_to_group[arm_id],
                "seed": seed,
                "evidence_source": evidence,
                "rejection_action": rejection,
                "adaptation": adaptation,
                "episodes": len(rows),
                "spectrum": spectrum,
                "outcomes": summarize_policy_outcomes(rows),
            }
        )
        policy_nulls[arm_id] = null_values

    associations = []
    ordered_arms = [str(row["arm_id"]) for row in policy_rows]
    for feature in registration["retrodiction"]["features"]:
        observed = [float(row["spectrum"][feature]) for row in policy_rows]
        null = [policy_nulls[arm][feature] for arm in ordered_arms]
        for outcome in registration["retrodiction"]["outcomes"]:
            values = [float(row["outcomes"][outcome]) for row in policy_rows]
            associations.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    **matched_null_association(observed, null, values),
                }
            )

    kernels = _kernel_rows(registration, benchmark_registration, kernel_records)
    result: dict[str, object] = {
        "schema": "controller_mesh_sheaf_retrodiction_v1",
        "study_id": registration["study_id"],
        "config_sha256": config_sha256,
        "source_receipts": {
            "result_sha256": result_sha256,
            "records_sha256": records_sha256,
            "record_count": record_count,
            "integrity_failures": 0,
        },
        "policy_instance_count": len(policy_rows),
        "behavioral_class_count": len(class_summaries),
        "topology_incompatible_behavioral_class_count": sum(
            not bool(row["topology_compatible"]) for row in class_summaries
        ),
        "behavioral_classes": class_summaries,
        "policy_sheaves": policy_rows,
        "associations": associations,
        "registered_contrasts": {
            evidence: _topology_contrast(policy_rows, evidence_source=evidence)
            for evidence in ("exact_mechanics", "exposed_frozen_probe", "hidden_rotating_probe")
        },
        "kernel_migration": {
            "seeds": kernels,
            "seed_29_prediction": prediction_result(kernels),
        },
        "matched_null": registration["matched_null"],
        "claim_boundary": registration["claim_boundary"],
    }
    result["analysis_receipt_sha256"] = canonical_sha256(result)
    return result


def experiment_notes(result: Mapping[str, object]) -> str:
    prediction = result["kernel_migration"]["seed_29_prediction"]
    return "\n".join(
        [
            "# Controller-Mesh Sheaf Retrodiction Notes",
            "",
            f"- protocol: `{result['study_id']}`",
            f"- config SHA-256: `{result['config_sha256']}`",
            f"- source records SHA-256: `{result['source_receipts']['records_sha256']}`",
            f"- analysis receipt SHA-256: `{result['analysis_receipt_sha256']}`",
            f"- policy instances / behavioral classes: `{result['policy_instance_count']} / {result['behavioral_class_count']}`",
            f"- strict seed-29 prediction passed: `{prediction['passed']}`",
            "- no utility or oracle label enters the primary sheaf construction",
            "- replayed round-0 model hashes bind the kernel audit to the original exposed probes",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrodict controller outcomes with registered empirical sheaf spectra."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    registration = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_retrodiction(registration)
    _write_json(result, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(result), encoding="utf-8")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result, args.experiment_dir / "results.json")
    (args.experiment_dir / "training_notes.md").write_text(
        experiment_notes(result), encoding="utf-8"
    )
    print(markdown_report(result))


if __name__ == "__main__":
    main()
