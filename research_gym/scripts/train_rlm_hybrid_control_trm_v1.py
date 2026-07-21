"""Train and seal the three CPU-only ControlTRM proposal checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    ACTION_VOCAB,
    LongContextControlTask,
    rank_actions,
)
from research_gym.integrity import canonical_file_sha256
from research_gym.neural.control_trm import ControlTRMProposer


ROOT = Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _append_event(path: Path, value: dict[str, Any]) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **value}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(payload))


def _task_tensors(tasks: list[LongContextControlTask]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_index = {action: index for index, action in enumerate(ACTION_VOCAB)}
    features = torch.tensor([task.public_features for task in tasks], dtype=torch.float32)
    labels = torch.tensor([action_index[task.optimal_action] for task in tasks], dtype=torch.long)
    masks = torch.tensor(
        [[action in task.candidates for action in ACTION_VOCAB] for task in tasks],
        dtype=torch.bool,
    )
    return features, labels, masks


def _masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, -1e9)


def _accuracy(model: ControlTRMProposer, tasks: list[LongContextControlTask]) -> float:
    features, labels, masks = _task_tensors(tasks)
    with torch.no_grad():
        logits = _masked_logits(model(features).action_logits, masks)
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train(config: dict[str, Any], output: Path) -> dict[str, Any]:
    summary_path = output / "training_summary.json"
    if summary_path.exists():
        raise RuntimeError("refusing to overwrite ControlTRM training outputs")
    task_path = (ROOT / config["task_suite_path"]).resolve()
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    tasks = [LongContextControlTask.from_jsonable(row) for row in task_payload["tasks"]]
    train_tasks = [task for task in tasks if task.split == "train"]
    calibration_tasks = [task for task in tasks if task.split == "calibration"]
    train_x, train_y, train_mask = _task_tensors(train_tasks)
    settings = config["training"]
    events_path = output / "training_events.jsonl"
    checkpoints = []
    proposal_rows = []
    seed_summaries = []
    for seed in settings["seeds"]:
        random.seed(seed)
        torch.manual_seed(seed)
        model = ControlTRMProposer(
            feature_dim=int(task_payload["feature_dim"]),
            action_count=len(ACTION_VOCAB),
            latent_dim=int(settings["latent_dim"]),
            recurrence_steps=int(settings["recurrence_steps"]),
        ).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
        generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
        _append_event(events_path, {"event": "seed_start", "seed": seed, "steps_completed": 0})
        last_loss = 0.0
        for step in range(1, int(settings["steps"]) + 1):
            indices = torch.randint(
                0,
                len(train_tasks),
                (int(settings["batch_size"]),),
                generator=generator,
            )
            output_values = model(train_x[indices])
            logits = _masked_logits(output_values.action_logits, train_mask[indices])
            loss = nn.functional.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            optimizer.step()
            last_loss = float(loss.item())
            if step % int(settings["checkpoint_interval"]) == 0:
                checkpoint_path = output / "checkpoints" / f"seed-{seed}-step-{step}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema_version": "1.0.0",
                        "seed": seed,
                        "step": step,
                        "action_vocab": ACTION_VOCAB,
                        "feature_dim": task_payload["feature_dim"],
                        "latent_dim": settings["latent_dim"],
                        "recurrence_steps": settings["recurrence_steps"],
                        "state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )
                digest = canonical_file_sha256(checkpoint_path)
                checkpoints.append(
                    {
                        "seed": seed,
                        "step": step,
                        "path": checkpoint_path.relative_to(ROOT).as_posix(),
                        "sha256": digest,
                        "final": step == int(settings["steps"]),
                    }
                )
                _append_event(
                    events_path,
                    {
                        "event": "checkpoint",
                        "seed": seed,
                        "step": step,
                        "path": checkpoint_path.relative_to(ROOT).as_posix(),
                        "sha256": digest,
                        "loss": last_loss,
                    },
                )
        model.eval()
        seed_summaries.append(
            {
                "seed": seed,
                "final_loss": last_loss,
                "train_accuracy": _accuracy(model, train_tasks),
                "calibration_accuracy": _accuracy(model, calibration_tasks),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        for task in tasks:
            features = torch.tensor(task.public_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(features).action_logits[0]
            scores = {action: float(logits[ACTION_VOCAB.index(action)].item()) for action in task.candidates}
            ranked = rank_actions(scores, task.candidates)
            probabilities = torch.softmax(torch.tensor([scores[action] for action in ranked]), dim=0)
            proposal_rows.append(
                {
                    "record_id": f"seed-{seed}__{task.task_id}",
                    "seed": seed,
                    "task_id": task.task_id,
                    "ranked_actions": ranked,
                    "scores": {action: round(scores[action], 8) for action in task.candidates},
                    "confidence": round(float(probabilities[0].item()), 8),
                    "claimed_provenance": "model_sound",
                }
            )
        del optimizer, model
        gc.collect()
    proposal_path = output / "proposal_table.jsonl"
    with proposal_path.open("wb") as handle:
        for row in sorted(proposal_rows, key=lambda value: value["record_id"]):
            handle.write(_canonical_bytes(row))
    manifest_path = output / "checkpoint_manifest.json"
    _write_json(manifest_path, checkpoints)
    summary = {
        "protocol_id": config["protocol_id"],
        "status": "complete",
        "steps_completed": len(settings["seeds"]) * int(settings["steps"]),
        "task_suite_path": task_path.relative_to(ROOT).as_posix(),
        "task_suite_sha256": canonical_file_sha256(task_path),
        "training": settings,
        "seed_summaries": seed_summaries,
        "checkpoint_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": canonical_file_sha256(manifest_path),
        "proposal_table_path": proposal_path.relative_to(ROOT).as_posix(),
        "proposal_table_sha256": canonical_file_sha256(proposal_path),
        "proposal_count": len(proposal_rows),
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run",), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vram-fraction", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(train(config, output), sort_keys=True))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
