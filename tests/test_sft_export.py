import json

from research_gym.core.frames import Frame, read_frame_jsonl, write_frame_jsonl
from research_gym.core.typed_soundness import SoundnessType
from research_gym.scripts.export_sft_dataset import (
    INSTRUCTION,
    decision_for_frame,
    read_common_frames,
    sft_record_for_frame,
    write_sft_dataset,
)


def sample_frame() -> Frame:
    return Frame(
        id="frame-1",
        family="deduction",
        source="unit",
        input_state={"slot": ["a", "b"]},
        operation={"rule": "prune"},
        output_state={"slot": ["a"]},
        soundness_type=SoundnessType.ENV_SOUND_DEAD,
        label="refinement",
    )


def test_decision_for_frame_preserves_typed_target():
    decision = decision_for_frame(sample_frame())

    assert decision == {
        "family": "deduction",
        "soundness_type": "env_sound_dead",
        "label": "refinement",
        "output_state": {"slot": ["a"]},
    }


def test_sft_record_contains_json_strings_and_metadata():
    record = sft_record_for_frame(sample_frame())

    assert record["instruction"] == INSTRUCTION
    assert json.loads(record["input"])["id"] == "frame-1"
    assert json.loads(record["output"])["soundness_type"] == "env_sound_dead"
    assert record["metadata"]["family"] == "deduction"


def test_write_sft_dataset(tmp_path):
    out = tmp_path / "sft" / "frames.jsonl"

    write_sft_dataset([sample_frame()], out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["metadata"]["id"] == "frame-1"


def test_read_common_frames_from_file_and_directory(tmp_path):
    frame_path = tmp_path / "frames.jsonl"
    write_frame_jsonl(frame_path, [sample_frame()])

    assert read_common_frames(frame_path) == read_frame_jsonl(frame_path)
    assert read_common_frames(tmp_path) == read_frame_jsonl(frame_path)
