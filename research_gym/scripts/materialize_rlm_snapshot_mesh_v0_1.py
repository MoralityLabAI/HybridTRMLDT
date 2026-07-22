"""Materialize the fresh task/proposal corpus for snapshot-first mesh v0.1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    ACTION_VOCAB,
    LongContextControlTask,
    materialize_task_suite,
    rank_actions,
)
from research_gym.integrity import canonical_file_sha256
from research_gym.neural.control_trm import ControlTRMProposer


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SEED = 194117
TASK_SUFFIX = "__snapshot_v0_1"
CHECKPOINT = ROOT / "experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/checkpoints/seed-211-step-200.pt"
DEFAULT_TASKS = ROOT / "data/benchmarks/rlm_snapshot_mesh_tasks_v0_1.json"
DEFAULT_PROPOSALS = ROOT / "data/benchmarks/rlm_snapshot_mesh_proposals_v0_1.jsonl"
DEFAULT_RECEIPT = ROOT / "data/benchmarks/rlm_snapshot_mesh_materialization_v0_1_receipt.json"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _renamed_payload() -> dict[str, Any]:
    payload = materialize_task_suite(seed=GENERATOR_SEED)
    payload["suite_id"] = "rlm_snapshot_first_mesh_control_v0_1"
    payload["parent_generator"] = "rlm_trm_ldt_long_context_control_v1"
    for row in payload["tasks"]:
        row["task_id"] = str(row["task_id"]) + TASK_SUFFIX
        row["group_id"] = str(row["group_id"]) + TASK_SUFFIX
    payload["split_group_sha256"] = {
        split: _canonical_sha256(
            sorted(row["group_id"] for row in payload["tasks"] if row["split"] == split)
        )
        for split in ("train", "calibration", "eval")
    }
    return payload


def _proposal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if tuple(checkpoint["action_vocab"]) != ACTION_VOCAB:
        raise RuntimeError("ControlTRM checkpoint action vocabulary changed")
    model = ControlTRMProposer(
        feature_dim=int(checkpoint["feature_dim"]),
        action_count=len(ACTION_VOCAB),
        latent_dim=int(checkpoint["latent_dim"]),
        recurrence_steps=int(checkpoint["recurrence_steps"]),
    ).cpu()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = []
    for raw in payload["tasks"]:
        task = LongContextControlTask.from_jsonable(raw)
        features = torch.tensor(task.public_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(features).action_logits[0]
        scores = {
            action: float(logits[ACTION_VOCAB.index(action)].item())
            for action in task.candidates
        }
        ranked = rank_actions(scores, task.candidates)
        probabilities = torch.softmax(
            torch.tensor([scores[action] for action in ranked]), dim=0
        )
        rows.append(
            {
                "record_id": f"seed-211__{task.task_id}",
                "seed": 211,
                "task_id": task.task_id,
                "ranked_actions": ranked,
                "scores": {
                    action: round(scores[action], 8) for action in task.candidates
                },
                "confidence": round(float(probabilities[0].item()), 8),
                "claimed_provenance": "model_sound",
            }
        )
    return sorted(rows, key=lambda row: row["record_id"])


def materialize(tasks_path: Path, proposals_path: Path, receipt_path: Path) -> dict[str, Any]:
    for path in (tasks_path, proposals_path, receipt_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite snapshot-mesh corpus artifact: {path}")
    payload = _renamed_payload()
    proposals = _proposal_rows(payload)
    _write_json(tasks_path, payload)
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    with proposals_path.open("wb") as handle:
        for row in proposals:
            handle.write(_canonical_bytes(row))
    receipt = {
        "status": "materialized",
        "corpus_id": payload["suite_id"],
        "generator_seed": GENERATOR_SEED,
        "task_suffix": TASK_SUFFIX,
        "task_count": len(payload["tasks"]),
        "proposal_count": len(proposals),
        "checkpoint_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": canonical_file_sha256(CHECKPOINT),
        "tasks_path": tasks_path.relative_to(ROOT).as_posix(),
        "tasks_sha256": canonical_file_sha256(tasks_path),
        "proposals_path": proposals_path.relative_to(ROOT).as_posix(),
        "proposals_sha256": canonical_file_sha256(proposals_path),
        "source_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "source_sha256": canonical_file_sha256(__file__),
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "provider_outcomes_present": False,
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    value = materialize(args.tasks.resolve(), args.proposals.resolve(), args.receipt.resolve())
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
