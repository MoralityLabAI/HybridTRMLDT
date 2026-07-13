from dataclasses import replace
import math

from research_gym.benchmarks.training_review_bench import candidate_matrix, clean_candidate
from research_gym.core.training_review import (
    IdentityLevel,
    ReviewRoute,
    ReviewUse,
    TrainingMechanism,
    control_risk_bound,
    review_training_candidate,
)


def test_clean_signed_candidate_is_holonomy_clean_and_authorized():
    candidate = clean_candidate()
    result = review_training_candidate(candidate, ReviewUse.MODEL_PROMOTION)

    assert result.authorized
    assert result.route == ReviewRoute.AUTHORIZE
    assert result.identity_level == IdentityLevel.HOLONOMY_CLEAN
    assert result.worst_signed_control_loss_upper_bound < 0.5
    assert len(result.receipt_sha256) == 64


def test_risk_bound_matches_lineage_plus_holonomy_theorem():
    measurement = clean_candidate().geometry.control_measurements[0]
    bound = control_risk_bound(measurement)
    expected_lineage = 1.0 - math.sqrt(0.99 - 0.005)
    expected_holonomy = 2.0 * math.sin(math.radians(5.0) / 2.0)

    assert math.isclose(bound.lineage_contraction_bound, expected_lineage)
    assert math.isclose(bound.holonomy_displacement_bound, expected_holonomy)
    assert math.isclose(bound.signed_control_loss_upper_bound, expected_lineage + expected_holonomy)


def test_high_holonomy_and_orientation_reversal_route_to_sectioning():
    candidates = candidate_matrix()
    for name in ("high_holonomy", "orientation_reversal"):
        result = review_training_candidate(candidates[name], ReviewUse.SIGNED_CONTROL)
        assert not result.authorized
        assert result.route == ReviewRoute.SECTION
        assert result.identity_level == IdentityLevel.LINEAGE_CERTIFIED


def test_unmeasured_loop_fails_closed_to_audit():
    candidate = candidate_matrix()["unmeasured_loop"]
    result = review_training_candidate(candidate, ReviewUse.SIGNED_CONTROL)

    assert result.route == ReviewRoute.AUDIT
    assert result.worst_signed_control_loss_upper_bound == 2.0
    assert "holonomy_auditor" in result.next_modules


def test_long_path_can_fail_despite_strong_local_edge_retention():
    candidate = candidate_matrix()["long_high_retention_path"]
    result = review_training_candidate(candidate, ReviewUse.SIGNED_CONTROL)

    assert all(edge.minimum_edge_worst_direction_retention >= 0.95 for edge in candidate.geometry.control_measurements[0].edges)
    assert result.worst_signed_control_loss_upper_bound > 0.5
    assert result.route == ReviewRoute.SECTION


def test_geometry_cannot_rescue_failed_held_out_utility():
    candidate = candidate_matrix()["utility_failure"]
    result = review_training_candidate(candidate, ReviewUse.MODEL_PROMOTION)

    assert result.identity_level == IdentityLevel.HOLONOMY_CLEAN
    assert result.route == ReviewRoute.REJECT
    assert "grouped_held_out_gain_not_positive" in result.failures


def test_high_holonomy_has_different_effect_by_training_mechanism():
    candidates = candidate_matrix()
    ordinary = review_training_candidate(candidates["ordinary_high_holonomy"], ReviewUse.MODEL_PROMOTION)
    bundle = review_training_candidate(candidates["bundle_high_holonomy"], ReviewUse.MODEL_PROMOTION)

    assert ordinary.route == ReviewRoute.AUTHORIZE
    assert bundle.route == ReviewRoute.AUTHORIZE
    assert bundle.required_identity_level == IdentityLevel.LINEAGE_CERTIFIED

    signed_candidate = replace(
        candidates["ordinary_high_holonomy"],
        mechanism=TrainingMechanism.SIGNED_GLOBAL_CONTROL,
    )
    signed = review_training_candidate(signed_candidate, ReviewUse.MODEL_PROMOTION)
    assert signed.route == ReviewRoute.SECTION


def test_incomplete_resource_receipts_route_promotion_to_audit():
    candidate = candidate_matrix()["resource_incomplete"]
    result = review_training_candidate(candidate, ReviewUse.MODEL_PROMOTION)

    assert result.route == ReviewRoute.AUDIT
    assert "resource_receipt_auditor" in result.next_modules


def test_aborted_training_is_retained_but_not_promoted():
    candidate = candidate_matrix()["aborted_run"]
    result = review_training_candidate(candidate, ReviewUse.MODEL_PROMOTION)

    assert result.route == ReviewRoute.REJECT
    assert "training_run_aborted" in result.failures


def test_sectioned_signed_promotion_requires_at_least_one_clean_patch():
    candidate = replace(
        clean_candidate(),
        mechanism=TrainingMechanism.SIGNED_SECTIONED_CONTROL,
        geometry=replace(
            clean_candidate().geometry,
            section_count=0,
            all_sections_holonomy_clean=True,
        ),
    )
    result = review_training_candidate(candidate, ReviewUse.MODEL_PROMOTION)

    assert result.route == ReviewRoute.SECTION
    assert "section_certificates_incomplete" in result.failures
