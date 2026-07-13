"""RSITopology-derived authorization gates for HRM training review.

The reviewer consumes sealed receipts. It does not inspect outcomes while
discovering geometry, mutate model weights, or upgrade receipt provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Iterable


class IdentityLevel(str, Enum):
    ENGINEERING_EVIDENCE = "engineering_evidence"
    LINEAGE_CERTIFIED = "lineage_certified"
    HOLONOMY_CLEAN = "holonomy_clean"


class TrainingMechanism(str, Enum):
    ORDINARY_OPTIMIZER = "ordinary_optimizer"
    BUNDLE_ALLOCATION = "bundle_allocation"
    SIGNED_GLOBAL_CONTROL = "signed_global_control"
    SIGNED_SECTIONED_CONTROL = "signed_sectioned_control"


class ReviewUse(str, Enum):
    ENGINEERING_REVIEW = "engineering_review"
    BUNDLE_ENERGY_ALLOCATION = "bundle_energy_allocation"
    SIGNED_CONTROL = "signed_control"
    MODEL_PROMOTION = "model_promotion"


class ReviewRoute(str, Enum):
    AUTHORIZE = "authorize"
    SECTION = "section"
    AUDIT = "audit"
    REJECT = "reject"


class TrainingRunStatus(str, Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class EdgeRiskReceipt:
    edge_id: str
    minimum_edge_worst_direction_retention: float
    retention_uncertainty: float = 0.0


@dataclass(frozen=True)
class LoopRiskReceipt:
    maximum_canonical_angle_degrees: float | None
    angle_uncertainty_degrees: float
    det_h_flag: bool
    measured: bool = True
    matched_noise_null_passed: bool = True


@dataclass(frozen=True)
class ControlRiskMeasurement:
    measurement_id: str
    edges: tuple[EdgeRiskReceipt, ...]
    loop: LoopRiskReceipt
    simultaneous_coverage: float


@dataclass(frozen=True)
class ControlRiskBound:
    measurement_id: str
    lineage_contraction_point: float
    lineage_contraction_bound: float
    holonomy_displacement_point: float
    holonomy_displacement_bound: float
    point_risk_score: float
    signed_control_loss_upper_bound: float
    uncertainty_margin: float
    audit_gap: bool
    orientation_reversal: bool
    noise_null_passed: bool
    simultaneous_coverage: float


@dataclass(frozen=True)
class GeometryReceipt:
    band_id: str
    inferred_rank: int
    occupancy_margin: float
    band_stable: bool
    jacobians_registered: bool
    target_blind_seal: str
    control_measurements: tuple[ControlRiskMeasurement, ...]
    section_count: int = 0
    all_sections_holonomy_clean: bool = False


@dataclass(frozen=True)
class UtilityReceipt:
    grouped_held_out_gain: float
    standardized_uplift: float
    realized_kl: float
    registered_kl_budget: float
    matched_rank_haar_win: bool
    replicate_count: int
    revealed_after_geometry_seal: bool


@dataclass(frozen=True)
class DamageReceipt:
    held_out: bool
    suite_passed: bool
    maximum_regression: float


@dataclass(frozen=True)
class ResourceReceipt:
    run_id: str
    status: TrainingRunStatus
    ram_cap_mb: int
    cpu_cap_pct: float
    io_cap_mb_s: float
    checkpoint_interval: str
    chunk_strategy: str
    hard_caps_enforced: bool
    checkpoint_present: bool
    timeout_respected: bool
    event_log_present: bool
    summary_present: bool
    resource_metrics_present: bool
    abort_status_recorded: bool
    cleanup_recorded: bool
    cleanup_passed: bool


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    mechanism: TrainingMechanism
    model_sha256: str
    dataset_sha256: str
    operator_sha256: str
    geometry: GeometryReceipt
    utility: UtilityReceipt
    damage: DamageReceipt
    resources: ResourceReceipt


@dataclass(frozen=True)
class TrainingReviewPolicy:
    signed_error_budget: float = 0.5
    delta: float = 0.05
    minimum_occupancy_margin: float = 0.05
    minimum_worst_direction_retention: float = 0.8
    minimum_grouped_gain: float = 0.0
    minimum_standardized_uplift: float = 0.0
    maximum_realized_kl: float = 0.05
    minimum_replicates: int = 3
    maximum_damage_regression: float = 0.0


@dataclass(frozen=True)
class TrainingReviewResult:
    candidate_id: str
    requested_use: ReviewUse
    mechanism: TrainingMechanism
    route: ReviewRoute
    authorized: bool
    identity_level: IdentityLevel
    required_identity_level: IdentityLevel
    worst_signed_control_loss_upper_bound: float
    risk_budget_margin: float
    information_coefficient: float | None
    failures: tuple[str, ...]
    next_modules: tuple[str, ...]
    receipt_sha256: str
    claim_boundary: str


_IDENTITY_ORDER = {
    IdentityLevel.ENGINEERING_EVIDENCE: 0,
    IdentityLevel.LINEAGE_CERTIFIED: 1,
    IdentityLevel.HOLONOMY_CLEAN: 2,
}


def control_risk_bound(measurement: ControlRiskMeasurement) -> ControlRiskBound:
    """Compute the conservative RSITopology signed-coordinate error bound."""
    if not measurement.edges:
        raise ValueError("at least one edge receipt is required")
    if not 0.0 <= measurement.simultaneous_coverage <= 1.0:
        raise ValueError("simultaneous_coverage must lie in [0,1]")

    lineage_point = 0.0
    lineage_bound = 0.0
    for edge in measurement.edges:
        value = float(edge.minimum_edge_worst_direction_retention)
        uncertainty = float(edge.retention_uncertainty)
        if not math.isfinite(value) or not math.isfinite(uncertainty):
            raise ValueError(f"non-finite retention receipt: {edge.edge_id}")
        if not 0.0 <= value <= 1.0 or uncertainty < 0.0:
            raise ValueError(f"invalid retention receipt: {edge.edge_id}")
        lineage_point += 1.0 - math.sqrt(value)
        lineage_bound += 1.0 - math.sqrt(max(0.0, value - uncertainty))

    loop = measurement.loop
    if not math.isfinite(loop.angle_uncertainty_degrees) or loop.angle_uncertainty_degrees < 0.0:
        raise ValueError("angle uncertainty must be nonnegative")
    if loop.det_h_flag:
        holonomy_point = 2.0
        holonomy_bound = 2.0
    else:
        if loop.maximum_canonical_angle_degrees is None:
            raise ValueError("orientation-preserving receipt requires a canonical angle")
        angle = float(loop.maximum_canonical_angle_degrees)
        if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
            raise ValueError("canonical angle must lie in [0,180]")
        holonomy_point = 2.0 * math.sin(math.radians(angle) / 2.0)
        upper_angle = min(180.0, angle + loop.angle_uncertainty_degrees)
        holonomy_bound = 2.0 * math.sin(math.radians(upper_angle) / 2.0)

    point = min(2.0, lineage_point + holonomy_point)
    conservative = min(2.0, lineage_bound + holonomy_bound)
    audit_gap = not loop.measured
    if audit_gap:
        conservative = 2.0
    return ControlRiskBound(
        measurement_id=measurement.measurement_id,
        lineage_contraction_point=lineage_point,
        lineage_contraction_bound=lineage_bound,
        holonomy_displacement_point=holonomy_point,
        holonomy_displacement_bound=holonomy_bound,
        point_risk_score=point,
        signed_control_loss_upper_bound=conservative,
        uncertainty_margin=max(0.0, conservative - point),
        audit_gap=audit_gap,
        orientation_reversal=loop.det_h_flag,
        noise_null_passed=loop.matched_noise_null_passed,
        simultaneous_coverage=measurement.simultaneous_coverage,
    )


def information_coefficient(standardized_uplift: float, realized_kl: float) -> float | None:
    """Return the descriptive uplift/KL ratio; it is never an authorization gate."""
    if realized_kl <= 0.0:
        return None
    return standardized_uplift / math.sqrt(2.0 * realized_kl)


def review_training_candidate(
    candidate: TrainingCandidate,
    requested_use: ReviewUse,
    policy: TrainingReviewPolicy | None = None,
) -> TrainingReviewResult:
    """Route a sealed training candidate through typed HRM review modules."""
    policy = policy or TrainingReviewPolicy()
    _validate_policy(policy)
    bounds = tuple(control_risk_bound(item) for item in candidate.geometry.control_measurements)
    worst_bound = max((item.signed_control_loss_upper_bound for item in bounds), default=2.0)
    identity_level, identity_failures = _identity_level(candidate, bounds, policy)
    required_level = _required_identity_level(candidate.mechanism, requested_use)

    failures = list(identity_failures)
    audit_modules: list[str] = []
    hard_rejections: list[str] = []

    if not _is_sha256(candidate.model_sha256):
        hard_rejections.append("model_hash_invalid")
    if not _is_sha256(candidate.dataset_sha256):
        hard_rejections.append("dataset_hash_invalid")
    if not _is_sha256(candidate.operator_sha256):
        hard_rejections.append("operator_hash_invalid")

    if requested_use != ReviewUse.ENGINEERING_REVIEW:
        _review_utility(candidate.utility, policy, failures, hard_rejections, audit_modules)
    if requested_use in {ReviewUse.SIGNED_CONTROL, ReviewUse.MODEL_PROMOTION}:
        _review_damage(candidate.damage, policy, failures, hard_rejections, audit_modules)
    if requested_use == ReviewUse.MODEL_PROMOTION:
        _review_resources(candidate.resources, failures, hard_rejections, audit_modules)

    if requested_use in {ReviewUse.BUNDLE_ENERGY_ALLOCATION, ReviewUse.MODEL_PROMOTION} and (
        candidate.mechanism == TrainingMechanism.BUNDLE_ALLOCATION
        or requested_use == ReviewUse.BUNDLE_ENERGY_ALLOCATION
    ):
        if candidate.utility.realized_kl > candidate.utility.registered_kl_budget:
            hard_rejections.append("registered_kl_budget_exceeded")
        if candidate.utility.realized_kl > policy.maximum_realized_kl:
            hard_rejections.append("policy_kl_budget_exceeded")

    identity_sufficient = _IDENTITY_ORDER[identity_level] >= _IDENTITY_ORDER[required_level]
    if not identity_sufficient:
        failures.append(f"requires_{required_level.value}")

    signed_request = requested_use == ReviewUse.SIGNED_CONTROL or (
        requested_use == ReviewUse.MODEL_PROMOTION
        and candidate.mechanism == TrainingMechanism.SIGNED_GLOBAL_CONTROL
    )
    sectioned_promotion = requested_use == ReviewUse.MODEL_PROMOTION and (
        candidate.mechanism == TrainingMechanism.SIGNED_SECTIONED_CONTROL
    )

    if hard_rejections:
        route = ReviewRoute.REJECT
        next_modules = ("retain_base",)
    elif sectioned_promotion and (
        candidate.geometry.section_count < 1 or not candidate.geometry.all_sections_holonomy_clean
    ):
        route = ReviewRoute.SECTION
        failures.append("section_certificates_incomplete")
        next_modules = ("topology_sectioner", "patch_attestor", "damage_suite")
    elif signed_request and not identity_sufficient and _has_measured_topology(bounds):
        route = ReviewRoute.SECTION
        next_modules = ("topology_sectioner", "patch_attestor", "utility_evaluator")
    elif not identity_sufficient or audit_modules:
        route = ReviewRoute.AUDIT
        next_modules = tuple(dict.fromkeys(audit_modules or _identity_audit_modules(failures)))
    else:
        route = ReviewRoute.AUTHORIZE
        next_modules = ("authorization_receipt_signer", "promotion_registry")

    all_failures = tuple(dict.fromkeys(failures + hard_rejections))
    authorized = route == ReviewRoute.AUTHORIZE
    payload = {
        "candidate_id": candidate.candidate_id,
        "requested_use": requested_use.value,
        "mechanism": candidate.mechanism.value,
        "route": route.value,
        "authorized": authorized,
        "identity_level": identity_level.value,
        "required_identity_level": required_level.value,
        "worst_signed_control_loss_upper_bound": worst_bound,
        "risk_budget_margin": policy.signed_error_budget - worst_bound,
        "failures": all_failures,
        "model_sha256": candidate.model_sha256,
        "dataset_sha256": candidate.dataset_sha256,
        "operator_sha256": candidate.operator_sha256,
        "run_id": candidate.resources.run_id,
    }
    receipt_hash = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return TrainingReviewResult(
        candidate_id=candidate.candidate_id,
        requested_use=requested_use,
        mechanism=candidate.mechanism,
        route=route,
        authorized=authorized,
        identity_level=identity_level,
        required_identity_level=required_level,
        worst_signed_control_loss_upper_bound=worst_bound,
        risk_budget_margin=policy.signed_error_budget - worst_bound,
        information_coefficient=information_coefficient(
            candidate.utility.standardized_uplift,
            candidate.utility.realized_kl,
        ),
        failures=all_failures,
        next_modules=next_modules,
        receipt_sha256=receipt_hash,
        claim_boundary=(
            "The topology gate bounds false authorization for registered signed coordinates only. "
            "Model promotion additionally requires grouped held-out utility, damage, provenance, and resource receipts."
        ),
    )


def result_to_dict(result: TrainingReviewResult) -> dict[str, object]:
    payload = asdict(result)
    payload["requested_use"] = result.requested_use.value
    payload["mechanism"] = result.mechanism.value
    payload["route"] = result.route.value
    payload["identity_level"] = result.identity_level.value
    payload["required_identity_level"] = result.required_identity_level.value
    return payload


def _identity_level(
    candidate: TrainingCandidate,
    bounds: tuple[ControlRiskBound, ...],
    policy: TrainingReviewPolicy,
) -> tuple[IdentityLevel, tuple[str, ...]]:
    geometry = candidate.geometry
    failures: list[str] = []
    if geometry.inferred_rank < 1:
        failures.append("bundle_rank_not_inferred")
    if geometry.occupancy_margin < policy.minimum_occupancy_margin:
        failures.append("occupancy_margin_below_policy")
    if not geometry.band_stable:
        failures.append("spectral_band_unstable")
    if not geometry.jacobians_registered:
        failures.append("jacobians_not_registered")
    if not _is_sha256(geometry.target_blind_seal):
        failures.append("target_blind_geometry_not_sealed")
    if not bounds:
        failures.append("lineage_measurements_missing")

    lineage_failures = {
        "bundle_rank_not_inferred",
        "occupancy_margin_below_policy",
        "spectral_band_unstable",
        "jacobians_not_registered",
        "target_blind_geometry_not_sealed",
        "lineage_measurements_missing",
    }
    for bound, measurement in zip(bounds, geometry.control_measurements):
        for edge in measurement.edges:
            conservative_retention = max(
                0.0,
                edge.minimum_edge_worst_direction_retention - edge.retention_uncertainty,
            )
            if conservative_retention < policy.minimum_worst_direction_retention:
                failures.append(f"edge_retention_below_policy:{edge.edge_id}")
                lineage_failures.add(f"edge_retention_below_policy:{edge.edge_id}")
        if bound.audit_gap:
            failures.append(f"loop_unmeasured:{bound.measurement_id}")
        if bound.orientation_reversal:
            failures.append(f"orientation_reversal:{bound.measurement_id}")
        if not bound.noise_null_passed:
            failures.append(f"matched_noise_null_failed:{bound.measurement_id}")
        if bound.simultaneous_coverage < 1.0 - policy.delta:
            failures.append(f"simultaneous_coverage_below_policy:{bound.measurement_id}")
        if bound.signed_control_loss_upper_bound > policy.signed_error_budget:
            failures.append(f"signed_error_budget_exceeded:{bound.measurement_id}")

    if lineage_failures & set(failures):
        return IdentityLevel.ENGINEERING_EVIDENCE, tuple(failures)

    holonomy_failures = (
        "loop_unmeasured:",
        "orientation_reversal:",
        "matched_noise_null_failed:",
        "simultaneous_coverage_below_policy:",
        "signed_error_budget_exceeded:",
    )
    if any(item.startswith(holonomy_failures) for item in failures):
        return IdentityLevel.LINEAGE_CERTIFIED, tuple(failures)
    return IdentityLevel.HOLONOMY_CLEAN, tuple(failures)


def _required_identity_level(mechanism: TrainingMechanism, use: ReviewUse) -> IdentityLevel:
    if use == ReviewUse.SIGNED_CONTROL:
        return IdentityLevel.HOLONOMY_CLEAN
    if use == ReviewUse.BUNDLE_ENERGY_ALLOCATION:
        return IdentityLevel.LINEAGE_CERTIFIED
    if use == ReviewUse.MODEL_PROMOTION:
        if mechanism == TrainingMechanism.BUNDLE_ALLOCATION:
            return IdentityLevel.LINEAGE_CERTIFIED
        if mechanism == TrainingMechanism.SIGNED_GLOBAL_CONTROL:
            return IdentityLevel.HOLONOMY_CLEAN
        if mechanism == TrainingMechanism.SIGNED_SECTIONED_CONTROL:
            return IdentityLevel.LINEAGE_CERTIFIED
    return IdentityLevel.ENGINEERING_EVIDENCE


def _review_utility(
    receipt: UtilityReceipt,
    policy: TrainingReviewPolicy,
    failures: list[str],
    hard_rejections: list[str],
    audit_modules: list[str],
) -> None:
    numeric_values = (
        receipt.grouped_held_out_gain,
        receipt.standardized_uplift,
        receipt.realized_kl,
        receipt.registered_kl_budget,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        hard_rejections.append("utility_receipt_non_finite")
        return
    if receipt.realized_kl < 0.0 or receipt.registered_kl_budget < 0.0:
        hard_rejections.append("kl_receipt_negative")
    if not receipt.revealed_after_geometry_seal:
        hard_rejections.append("prereveal_geometry_seal_violated")
    if receipt.grouped_held_out_gain <= policy.minimum_grouped_gain:
        hard_rejections.append("grouped_held_out_gain_not_positive")
    if receipt.standardized_uplift <= policy.minimum_standardized_uplift:
        hard_rejections.append("standardized_uplift_not_positive")
    if not receipt.matched_rank_haar_win:
        hard_rejections.append("matched_rank_haar_control_not_beaten")
    if receipt.replicate_count < policy.minimum_replicates:
        failures.append("insufficient_replicates")
        audit_modules.append("utility_evaluator")


def _review_damage(
    receipt: DamageReceipt,
    policy: TrainingReviewPolicy,
    failures: list[str],
    hard_rejections: list[str],
    audit_modules: list[str],
) -> None:
    if not math.isfinite(receipt.maximum_regression):
        hard_rejections.append("damage_receipt_non_finite")
        return
    if not receipt.held_out:
        failures.append("held_out_damage_suite_missing")
        audit_modules.append("damage_suite")
    if not receipt.suite_passed or receipt.maximum_regression > policy.maximum_damage_regression:
        hard_rejections.append("damage_budget_failed")


def _review_resources(
    receipt: ResourceReceipt,
    failures: list[str],
    hard_rejections: list[str],
    audit_modules: list[str],
) -> None:
    if not receipt.run_id:
        hard_rejections.append("run_id_missing")
    if receipt.status == TrainingRunStatus.ABORTED:
        hard_rejections.append("training_run_aborted")
    if not receipt.timeout_respected:
        hard_rejections.append("training_timeout_exceeded")
    if not receipt.hard_caps_enforced:
        hard_rejections.append("hard_resource_caps_not_enforced")
    if receipt.cleanup_recorded and not receipt.cleanup_passed:
        hard_rejections.append("owned_process_cleanup_failed")
    cap_values_finite = math.isfinite(receipt.cpu_cap_pct) and math.isfinite(receipt.io_cap_mb_s)
    if (
        receipt.ram_cap_mb < 1
        or not cap_values_finite
        or not 0.0 < receipt.cpu_cap_pct <= 100.0
        or receipt.io_cap_mb_s <= 0.0
    ):
        hard_rejections.append("resource_caps_invalid")
    for condition, failure in (
        (bool(receipt.checkpoint_interval), "checkpoint_interval_missing"),
        (bool(receipt.chunk_strategy), "chunk_strategy_missing"),
        (receipt.checkpoint_present, "checkpoint_receipt_missing"),
        (receipt.event_log_present, "training_event_log_missing"),
        (receipt.summary_present, "training_summary_missing"),
        (receipt.resource_metrics_present, "resource_metrics_missing"),
        (receipt.abort_status_recorded, "abort_status_missing"),
        (receipt.cleanup_recorded, "cleanup_receipt_missing"),
    ):
        if not condition:
            failures.append(failure)
            audit_modules.append("resource_receipt_auditor")


def _has_measured_topology(bounds: Iterable[ControlRiskBound]) -> bool:
    values = tuple(bounds)
    return bool(values) and all(not item.audit_gap for item in values)


def _identity_audit_modules(failures: Iterable[str]) -> tuple[str, ...]:
    modules = []
    for failure in failures:
        if failure.startswith(("bundle_", "occupancy_", "spectral_", "jacobians_", "target_blind_")):
            modules.append("spectral_geometry_auditor")
        elif failure.startswith("edge_") or failure.startswith("lineage_"):
            modules.append("lineage_auditor")
        elif failure.startswith(("loop_", "orientation_", "matched_noise_", "simultaneous_", "signed_")):
            modules.append("holonomy_auditor")
    return tuple(dict.fromkeys(modules or ("receipt_auditor",)))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _validate_policy(policy: TrainingReviewPolicy) -> None:
    if not 0.0 <= policy.signed_error_budget <= 2.0:
        raise ValueError("signed_error_budget must lie in [0,2]")
    if not 0.0 <= policy.delta < 1.0:
        raise ValueError("delta must lie in [0,1)")
    if not 0.0 <= policy.minimum_worst_direction_retention <= 1.0:
        raise ValueError("minimum retention must lie in [0,1]")
    if policy.minimum_replicates < 1:
        raise ValueError("minimum_replicates must be positive")
