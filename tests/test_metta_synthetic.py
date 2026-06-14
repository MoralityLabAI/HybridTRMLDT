from pathlib import Path

from research_gym.core.metta_frames import DeductionFrame, ExecutionFrame, RepairFrame, RoutingFrame
from research_gym.envs.metta_synthetic import default_frames, frames_from_file


def by_rule(frames):
    return {getattr(frame, "rule_name", ""): frame for frame in frames if hasattr(frame, "rule_name")}


def test_default_synthetic_metta_includes_distinct_mechanics():
    frames = default_frames()
    rules = by_rule(frames)

    assert "arith_add_heat" in rules
    assert "compare_heat_gate" in rules
    assert "precondition_fail_defuse" in rules
    assert "postcondition_update_scene" in rules
    assert "type_check_action" in rules
    assert "deduction_closure" in rules
    assert "bottom_conflict" in rules


def test_precondition_failure_is_failed_execution_without_state_change():
    frame = by_rule(default_frames())["precondition_fail_defuse"]

    assert isinstance(frame, ExecutionFrame)
    assert not frame.passed
    assert frame.before == frame.after
    assert "precondition_failed" in frame.invariants


def test_type_check_eliminates_wrong_type_candidates():
    frame = by_rule(default_frames())["type_check_action"]

    assert isinstance(frame, DeductionFrame)
    assert frame.after["action"] == ["befriend", "defuse"]
    assert frame.eliminated["action"] == ["42"]
    assert not frame.conflict


def test_bottom_conflict_has_empty_candidate_set():
    frame = by_rule(default_frames())["bottom_conflict"]

    assert isinstance(frame, DeductionFrame)
    assert frame.conflict
    assert frame.after["choice"] == []


def test_default_synthetic_metta_includes_routing_and_malformed_rule_repair():
    frames = default_frames()
    routes = [frame for frame in frames if isinstance(frame, RoutingFrame)]
    repairs = [frame for frame in frames if isinstance(frame, RepairFrame)]

    assert {frame.chosen_skill for frame in routes} >= {"arith_add_heat", "precondition_fail_defuse"}
    assert any(frame.diagnostic == "malformed_rule" and frame.passed_after_repair for frame in repairs)


def test_example_file_matches_expanded_directive_coverage():
    frames = frames_from_file(Path("examples/metta_rules/toy_skills.metta"))
    rules = by_rule(frames)

    assert "arith_add_heat" in rules
    assert "bottom_conflict" in rules
    assert any(isinstance(frame, RepairFrame) and frame.diagnostic == "malformed_rule" for frame in frames)
