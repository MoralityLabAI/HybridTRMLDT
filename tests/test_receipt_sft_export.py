import json

from research_gym.core.frames import Frame
from research_gym.core.typed_soundness import SoundnessType
from research_gym.scripts.export_receipt_sft import (
    export,
    group_digest,
    public_input,
    proposal_target,
    sft_record,
    split_for,
)


def sample_frame(*, frame_id="case-a", family="reachability", label="dead"):
    return Frame(
        id=frame_id,
        family=family,
        source="unit",
        input_state={"slot": ["a", "b"], "observable": 4},
        operation={"rule": "prune"},
        output_state={"slot": ["b"]},
        soundness_type=SoundnessType.ENV_SOUND_DEAD,
        label=label,
        metadata={
            "witness_actions": ["secret_ground_truth"],
            "counterfactual": {"target": "secret_ground_truth"},
        },
    )


def test_only_public_fields_go_to_model_input():
    row = sft_record(sample_frame())
    value = json.loads(row["input"])
    assert set(value) == {"family", "input_state", "operation"}
    for forbidden in ["output_state", "soundness_type", "label", "metadata"]:
        assert forbidden not in value
    assert "secret_ground_truth" not in row["input"]
    assert row["metadata"]["authority"] == "supervised_prediction_only"
    assert json.loads(row["output"]) == {
        "label": "dead",
        "predicted_soundness_type": "env_sound_dead",
        "proposed_output_state": {"slot": ["b"]},
    }


def test_group_split_ignores_labels_and_ids():
    frame_a = sample_frame(frame_id="case-a")
    frame_b = sample_frame(frame_id="case-b", label="different_target")
    assert group_digest(frame_a) == group_digest(frame_b)
    assert split_for(frame_a) == split_for(frame_b)
    assert public_input(frame_a) == public_input(frame_b)
    assert proposal_target(frame_a) != proposal_target(frame_b)


def test_family_holdout_forces_eval():
    frame = sample_frame(family="unseen_reasoning")
    assert split_for(frame, holdout_families=["unseen_reasoning"]) == "eval"


def test_export_manifest_and_hashes(tmp_path):
    frames = [
        sample_frame(frame_id="case-a", family="unseen_reasoning"),
        sample_frame(frame_id="case-b", family="unseen_reasoning"),
    ]
    result = export(frames, tmp_path, holdout_families=["unseen_reasoning"])
    assert result["counts"] == {"train": 0, "calibration": 0, "eval": 2}
    assert result["unique_groups"] == 1
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == 2
    assert result["file_sha256"]["eval.jsonl"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == result
