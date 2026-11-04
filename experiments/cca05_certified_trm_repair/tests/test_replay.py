"""Benchmark reproducibility: rerun every arm from the committed proposer tapes."""

import json
from pathlib import Path

import pytest

import run_cca05 as R

OUT = Path(__file__).resolve().parents[1] / "outputs"


@pytest.mark.parametrize("tag", list(R.TAGS))
def test_tape_replay_reproduces_committed_receipts(tag):
    if not (OUT / "tapes" / f"{tag}.json.gz").exists():
        pytest.skip("tapes not generated yet")
    result = R.run_replay(None, [tag])[tag]
    assert result == {"receipt_digest_equal": True, "single_step_equal": True, "trajectory_equal": True}


def test_environment_digest_matches_committed_outputs():
    path = OUT / "environment.json"
    if not path.exists():
        pytest.skip("environment report not generated yet")
    assert json.loads(path.read_text(encoding="utf-8"))["env_digest"] == R.M.ENV_DIGEST
