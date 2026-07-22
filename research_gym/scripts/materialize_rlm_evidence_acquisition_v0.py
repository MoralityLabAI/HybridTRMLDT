"""Materialize the fresh corpus for the RLM evidence-acquisition mesh."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    ACTION_VOCAB,
    LongContextControlTask,
    canonical_sha256,
    materialize_task_suite,
    rank_actions,
)
from research_gym.integrity import canonical_file_sha256
from research_gym.neural.control_trm import ControlTRMProposer


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SEED = 394117
TASK_SUFFIX = "__acquisition_v0"
CHECKPOINT = ROOT / "experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/checkpoints/seed-211-step-200.pt"
DEFAULT_TASKS = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_tasks.json"
DEFAULT_PROPOSALS = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_proposals.jsonl"
DEFAULT_TRUTH = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_truth.jsonl"
DEFAULT_RECEIPT = ROOT / "data/benchmarks/rlm_evidence_acquisition_mesh_v0_materialization_receipt.json"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _write_jsonl(path: Path, values: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical_bytes(value))


def _rotate_scores(scores: Mapping[str, float], candidates: tuple[str, ...]) -> dict[str, float]:
    values = [float(scores[action]) for action in candidates]
    rotated = values[1:] + values[:1]
    return {action: round(rotated[index], 8) for index, action in enumerate(candidates)}


def _stratum(task_id: str) -> str:
    index = int(task_id.split("__")[-1])
    return ("proxy_stale", "trained_stale", "both_current")[index % 3]


def _renamed_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = materialize_task_suite(seed=GENERATOR_SEED)
    payload["suite_id"] = "rlm_evidence_acquisition_mesh_v0"
    payload["parent_generator"] = "rlm_trm_ldt_long_context_control_v1"
    truth_rows = []
    for row in payload["tasks"]:
        original_task_id = str(row["task_id"])
        stratum = _stratum(original_task_id)
        original_proxy = dict(row["proxy_scores"])
        if stratum == "proxy_stale":
            row["proxy_scores"] = _rotate_scores(original_proxy, tuple(row["candidates"]))
        row["task_id"] = original_task_id + TASK_SUFFIX
        row["group_id"] = str(row["group_id"]) + TASK_SUFFIX
        truth_rows.append(
            {
                "task_id": row["task_id"],
                "stratum": stratum,
                "proxy_current": stratum != "proxy_stale",
                "trained_current": stratum != "trained_stale",
                "original_proxy_scores_sha256": canonical_sha256(original_proxy),
                "deployed_proxy_scores_sha256": canonical_sha256(row["proxy_scores"]),
            }
        )
    payload["split_group_sha256"] = {
        split: sha256(
            _canonical_bytes(sorted(row["group_id"] for row in payload["tasks"] if row["split"] == split))
        ).hexdigest()
        for split in ("train", "calibration", "eval")
    }
    return payload, truth_rows


def _proposal_rows(payload: Mapping[str, Any], truth_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth = {str(row["task_id"]): row for row in truth_rows}
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
        original_scores = {
            action: float(logits[ACTION_VOCAB.index(action)].item())
            for action in task.candidates
        }
        scores = dict(original_scores)
        if not truth[task.task_id]["trained_current"]:
            scores = _rotate_scores(scores, task.candidates)
        ranked = rank_actions(scores, task.candidates)
        probabilities = torch.softmax(torch.tensor([scores[action] for action in ranked]), dim=0)
        truth[task.task_id]["original_trained_scores_sha256"] = canonical_sha256(original_scores)
        truth[task.task_id]["deployed_trained_scores_sha256"] = canonical_sha256(scores)
        receipt_ref = canonical_sha256(
            {
                "task_id": task.task_id,
                "checkpoint_sha256": canonical_file_sha256(CHECKPOINT),
                "scores_sha256": truth[task.task_id]["deployed_trained_scores_sha256"],
            }
        )[:20]
        rows.append(
            {
                "record_id": f"seed-211__{task.task_id}",
                "seed": 211,
                "task_id": task.task_id,
                "ranked_actions": ranked,
                "scores": {action: round(scores[action], 8) for action in task.candidates},
                "confidence": round(float(probabilities[0].item()), 8),
                "claimed_provenance": "model_sound",
                "proposal_receipt_ref": receipt_ref,
            }
        )
    return sorted(rows, key=lambda row: row["record_id"])


def materialize(
    tasks_path: Path,
    proposals_path: Path,
    truth_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    for path in (tasks_path, proposals_path, truth_path, receipt_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite evidence-acquisition artifact: {path}")
    payload, truth_rows = _renamed_payload()
    proposals = _proposal_rows(payload, truth_rows)
    truth_rows = sorted(truth_rows, key=lambda row: row["task_id"])
    _write_json(tasks_path, payload)
    _write_jsonl(proposals_path, proposals)
    _write_jsonl(truth_path, truth_rows)
    receipt = {
        "status": "materialized",
        "corpus_id": payload["suite_id"],
        "generator_seed": GENERATOR_SEED,
        "task_suffix": TASK_SUFFIX,
        "task_count": len(payload["tasks"]),
        "proposal_count": len(proposals),
        "truth_count": len(truth_rows),
        "checkpoint_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": canonical_file_sha256(CHECKPOINT),
        "tasks_path": tasks_path.relative_to(ROOT).as_posix(),
        "tasks_sha256": canonical_file_sha256(tasks_path),
        "proposals_path": proposals_path.relative_to(ROOT).as_posix(),
        "proposals_sha256": canonical_file_sha256(proposals_path),
        "truth_path": truth_path.relative_to(ROOT).as_posix(),
        "truth_sha256": canonical_file_sha256(truth_path),
        "source_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "source_sha256": canonical_file_sha256(__file__),
        "provider_outcomes_present": False,
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    value = materialize(
        args.tasks.resolve(),
        args.proposals.resolve(),
        args.truth.resolve(),
        args.receipt.resolve(),
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
