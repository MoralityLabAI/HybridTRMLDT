from research_gym.core.frames import Frame
from research_gym.core.metta_frames import DeductionFrame, ExecutionFrame, RepairFrame, RoutingFrame
from research_gym.core.typed_soundness import SoundnessType
from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv, StoryState


REQUIRED_KEYS = {
    "id",
    "family",
    "source",
    "input_state",
    "operation",
    "output_state",
    "soundness_type",
    "label",
    "metadata",
}


def test_reachability_frame_exports_common_schema():
    env = CoupledStoryworldEnv()
    specialized = env.label_frame(StoryState(1, 3, 0, 4), horizon=2, frame_id="reach-1")
    common = specialized.to_frame()
    restored = Frame.from_jsonable(common.to_jsonable())

    assert set(common.to_jsonable()) == REQUIRED_KEYS
    assert common.family == "reachability"
    assert common.soundness_type in set(SoundnessType)
    assert restored == common


def test_metta_frame_types_export_common_schema():
    frames = [
        ExecutionFrame(
            frame_id="exec-1",
            source="unit",
            rule_name="step",
            before={"x": 1},
            after={"x": 2},
            operation="step",
        ),
        DeductionFrame(
            frame_id="deduce-1",
            source="unit",
            rule_name="prune",
            before={"slot": ["a", "b"]},
            after={"slot": ["a"]},
            soundness=SoundnessType.ENV_SOUND_DEAD,
        ),
        RepairFrame(
            frame_id="repair-1",
            source="unit",
            diagnostic="unknown_symbol",
            buggy_code="(bad)",
            repaired_code="(good)",
            passed_after_repair=True,
        ),
        RoutingFrame(
            frame_id="route-1",
            source="unit",
            task_signature={"task": "choose"},
            candidate_skills=["a", "b"],
            chosen_skill="a",
        ),
    ]

    common = [frame.to_frame() for frame in frames]

    assert {frame.family for frame in common} == {"execution", "deduction", "repair", "routing"}
    assert all(set(frame.to_jsonable()) == REQUIRED_KEYS for frame in common)
    assert common[1].soundness_type == SoundnessType.ENV_SOUND_DEAD
