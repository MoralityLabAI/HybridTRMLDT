from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from research_gym.integrity import canonical_file_sha256
from research_gym.scripts import materialize_rlm_snapshot_mesh_v0_1 as materializer


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "data/benchmarks/rlm_snapshot_mesh_materialization_v0_1_receipt.json"


def _canonical_sha256(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return sha256(payload).hexdigest()


def test_materialized_corpus_replays_bound_hashes_and_counts() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    tasks_path = ROOT / receipt["tasks_path"]
    proposals_path = ROOT / receipt["proposals_path"]
    checkpoint_path = ROOT / receipt["checkpoint_path"]

    assert receipt["generator_seed"] == materializer.GENERATOR_SEED
    assert receipt["task_suffix"] == materializer.TASK_SUFFIX
    assert receipt["provider_outcomes_present"] is False
    assert canonical_file_sha256(tasks_path) == receipt["tasks_sha256"]
    assert canonical_file_sha256(proposals_path) == receipt["proposals_sha256"]
    assert canonical_file_sha256(checkpoint_path) == receipt["checkpoint_sha256"]
    assert canonical_file_sha256(ROOT / receipt["source_path"]) == receipt["source_sha256"]

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    proposals = [json.loads(line) for line in proposals_path.read_text(encoding="utf-8").splitlines()]
    assert len(tasks["tasks"]) == receipt["task_count"] == 152
    assert len(proposals) == receipt["proposal_count"] == 152


def test_materialized_corpus_has_isolated_ids_and_consistent_splits() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    payload = json.loads((ROOT / receipt["tasks_path"]).read_text(encoding="utf-8"))
    proposals = [
        json.loads(line)
        for line in (ROOT / receipt["proposals_path"]).read_text(encoding="utf-8").splitlines()
    ]
    task_ids = [row["task_id"] for row in payload["tasks"]]
    group_ids = [row["group_id"] for row in payload["tasks"]]

    assert len(task_ids) == len(set(task_ids))
    assert all(task_id.endswith(materializer.TASK_SUFFIX) for task_id in task_ids)
    assert all(group_id.endswith(materializer.TASK_SUFFIX) for group_id in group_ids)
    assert {row["task_id"] for row in proposals} == set(task_ids)
    assert len({row["record_id"] for row in proposals}) == len(proposals)
    for split in ("train", "calibration", "eval"):
        expected = sorted(row["group_id"] for row in payload["tasks"] if row["split"] == split)
        assert payload["split_group_sha256"][split] == _canonical_sha256(expected)


def test_materializer_refuses_to_overwrite_sealed_outputs() -> None:
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        materializer.materialize(
            materializer.DEFAULT_TASKS,
            materializer.DEFAULT_PROPOSALS,
            materializer.DEFAULT_RECEIPT,
        )
