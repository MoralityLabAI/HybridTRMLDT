"""Deterministic candidate matrix for RSITopology-aware HRM review."""

from __future__ import annotations

from dataclasses import replace

from research_gym.core.training_review import (
    ControlRiskMeasurement,
    DamageReceipt,
    EdgeRiskReceipt,
    GeometryReceipt,
    LoopRiskReceipt,
    ResourceReceipt,
    ReviewUse,
    TrainingCandidate,
    TrainingMechanism,
    TrainingReviewPolicy,
    TrainingRunStatus,
    UtilityReceipt,
    result_to_dict,
    review_training_candidate,
)


def _hash(label: str) -> str:
    return (label.encode("utf-8").hex() * 64)[:64]


def clean_candidate(
    candidate_id: str = "clean-signed",
    mechanism: TrainingMechanism = TrainingMechanism.SIGNED_GLOBAL_CONTROL,
) -> TrainingCandidate:
    edge = EdgeRiskReceipt("checkpoint-0-to-1", 0.99, 0.005)
    measurement = ControlRiskMeasurement(
        measurement_id="context-a/checkpoint-0-1",
        edges=(edge,),
        loop=LoopRiskReceipt(4.0, 1.0, False, True, True),
        simultaneous_coverage=0.99,
    )
    return TrainingCandidate(
        candidate_id=candidate_id,
        mechanism=mechanism,
        model_sha256=_hash(f"model-{candidate_id}"),
        dataset_sha256=_hash("dataset"),
        operator_sha256=_hash("operator"),
        geometry=GeometryReceipt(
            band_id="mid-band-0",
            inferred_rank=3,
            occupancy_margin=0.18,
            band_stable=True,
            jacobians_registered=True,
            target_blind_seal=_hash("geometry-seal"),
            control_measurements=(measurement,),
        ),
        utility=UtilityReceipt(
            grouped_held_out_gain=0.12,
            standardized_uplift=0.8,
            realized_kl=0.02,
            registered_kl_budget=0.03,
            matched_rank_haar_win=True,
            replicate_count=5,
            revealed_after_geometry_seal=True,
        ),
        damage=DamageReceipt(held_out=True, suite_passed=True, maximum_regression=-0.01),
        resources=ResourceReceipt(
            run_id=f"run-{candidate_id}",
            status=TrainingRunStatus.COMPLETED,
            ram_cap_mb=2048,
            cpu_cap_pct=50.0,
            io_cap_mb_s=50.0,
            checkpoint_interval="100 steps or 300 seconds",
            chunk_strategy="bounded minibatches",
            hard_caps_enforced=True,
            checkpoint_present=True,
            timeout_respected=True,
            event_log_present=True,
            summary_present=True,
            resource_metrics_present=True,
            abort_status_recorded=True,
            cleanup_recorded=True,
            cleanup_passed=True,
        ),
    )


def candidate_matrix() -> dict[str, TrainingCandidate]:
    clean = clean_candidate()
    high_holonomy_measurement = replace(
        clean.geometry.control_measurements[0],
        loop=LoopRiskReceipt(42.0, 2.0, False, True, True),
    )
    reversal_measurement = replace(
        clean.geometry.control_measurements[0],
        loop=LoopRiskReceipt(None, 0.0, True, True, True),
    )
    unmeasured = replace(
        clean.geometry.control_measurements[0],
        loop=LoopRiskReceipt(4.0, 1.0, False, False, True),
    )
    long_path = replace(
        clean.geometry.control_measurements[0],
        measurement_id="six-edge-checkpoint-path",
        edges=tuple(EdgeRiskReceipt(f"edge-{index}", 0.95, 0.01) for index in range(6)),
        loop=LoopRiskReceipt(18.0, 2.0, False, True, True),
    )
    return {
        "clean_signed": clean,
        "high_holonomy": replace(
            clean,
            candidate_id="high-holonomy",
            geometry=replace(clean.geometry, control_measurements=(high_holonomy_measurement,)),
        ),
        "orientation_reversal": replace(
            clean,
            candidate_id="orientation-reversal",
            geometry=replace(clean.geometry, control_measurements=(reversal_measurement,)),
        ),
        "unmeasured_loop": replace(
            clean,
            candidate_id="unmeasured-loop",
            geometry=replace(clean.geometry, control_measurements=(unmeasured,)),
        ),
        "long_high_retention_path": replace(
            clean,
            candidate_id="long-high-retention-path",
            geometry=replace(clean.geometry, control_measurements=(long_path,)),
        ),
        "utility_failure": replace(
            clean,
            candidate_id="utility-failure",
            utility=replace(clean.utility, grouped_held_out_gain=-0.02),
        ),
        "ordinary_high_holonomy": replace(
            clean,
            candidate_id="ordinary-high-holonomy",
            mechanism=TrainingMechanism.ORDINARY_OPTIMIZER,
            geometry=replace(clean.geometry, control_measurements=(high_holonomy_measurement,)),
        ),
        "bundle_high_holonomy": replace(
            clean,
            candidate_id="bundle-high-holonomy",
            mechanism=TrainingMechanism.BUNDLE_ALLOCATION,
            geometry=replace(clean.geometry, control_measurements=(high_holonomy_measurement,)),
        ),
        "resource_incomplete": replace(
            clean,
            candidate_id="resource-incomplete",
            resources=replace(clean.resources, checkpoint_present=False, cleanup_recorded=False),
        ),
        "aborted_run": replace(
            clean,
            candidate_id="aborted-run",
            resources=replace(clean.resources, status=TrainingRunStatus.ABORTED),
        ),
    }


def run_training_review_benchmark(policy: TrainingReviewPolicy | None = None) -> dict[str, object]:
    policy = policy or TrainingReviewPolicy()
    rows = []
    for scenario, candidate in candidate_matrix().items():
        use = ReviewUse.MODEL_PROMOTION
        if scenario in {"high_holonomy", "orientation_reversal", "unmeasured_loop", "long_high_retention_path"}:
            use = ReviewUse.SIGNED_CONTROL
        result = review_training_candidate(candidate, use, policy)
        rows.append({"scenario": scenario, **result_to_dict(result)})
    return {
        "schema_version": "1.0.0",
        "benchmark": "rsi_topology_hrm_training_review",
        "policy": {
            "signed_error_budget": policy.signed_error_budget,
            "delta": policy.delta,
            "minimum_occupancy_margin": policy.minimum_occupancy_margin,
            "minimum_worst_direction_retention": policy.minimum_worst_direction_retention,
            "maximum_realized_kl": policy.maximum_realized_kl,
            "minimum_replicates": policy.minimum_replicates,
            "resource_preset": {"ram_cap_mb": 2048, "cpu_cap_pct": 50.0, "io_cap_mb_s": 50.0},
        },
        "results": rows,
        "claim_boundary": (
            "Deterministic contract exercise over synthetic sealed receipts; no neural model was trained, "
            "promoted, or edited by this benchmark."
        ),
    }


def summary_markdown(payload: dict[str, object]) -> str:
    rows = payload["results"]
    assert isinstance(rows, list)
    lines = [
        "# RSITopology-Aware HRM Training Review",
        "",
        "| Scenario | Mechanism | Request | Identity | Route | Bound |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        lines.append(
            f"| {row['scenario']} | {row['mechanism']} | {row['requested_use']} | "
            f"{row['identity_level']} | {row['route']} | "
            f"{float(row['worst_signed_control_loss_upper_bound']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The clean signed candidate is authorized. High measured holonomy and orientation reversal are routed",
            "to sectioning rather than global signed control. An unmeasured loop routes to audit. Strong local edge",
            "retention does not rescue a long path whose conservative bound exceeds the error budget.",
            "",
            "Ordinary optimizer promotion remains a behavioral decision even when signed internal coordinates are not",
            "globally identifiable. Bundle allocation needs lineage but not globally flat signed coordinates. Geometry",
            "alone never authorizes promotion: held-out utility, matched controls, damage, provenance, and resource",
            "receipts remain separate gates.",
            "",
            f"Claim boundary: {payload['claim_boundary']}",
            "",
        ]
    )
    return "\n".join(lines)
