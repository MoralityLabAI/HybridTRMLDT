"""Read-only CPU calibration replay of frozen ControlTRM step-two latents.

Use only the registered 32 calibration tasks and three final step-200 seeds.
No model fitting, checkpoint creation, provider call or held-out scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from research_gym.neural.control_trm import ControlTRMProposer
from research_gym.neural.control_trm_intervention import continue_from_two, latent_after_two

CONFIG = "configs/rlm_trm_ldt_hybrid_neighborhood_v1_1.json"
REGISTRATION = "configs/rlm_trm_ldt_hybrid_neighborhood_v1_1_registration.json"
SEEDS = (211, 223, 227)
MAX_LOGIT_ERROR = 2e-5
MAX_SHAM_ERROR = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_path(root: Path, rel: str, digest: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError("registered path escaped repository root")
    if _sha256(path) != digest:
        raise ValueError(f"registered artifact digest mismatch: {rel}")
    return path


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    registered = json.loads((root / REGISTRATION).read_text(encoding="utf-8"))
    config_path = _pinned_path(root, CONFIG, registered["config_sha256"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if registered["protocol_id"] != config["protocol_id"]:
        raise ValueError("registration protocol identity mismatch")
    suite_pin = config["task_suite"]
    suite = json.loads(_pinned_path(root, suite_pin["path"], suite_pin["sha256"]).read_text(encoding="utf-8"))
    if suite["feature_dim"] != 44 or len(suite["action_vocab"]) != 16:
        raise ValueError("unexpected frozen ControlTRM input/action shape")
    calibration = [row for row in suite["tasks"] if row["split"] == "calibration"]
    families = Counter(row["family"] for row in calibration)
    if len(calibration) != 32 or set(families.values()) != {8} or len(families) != 4:
        raise ValueError("expected 32 calibration tasks across four families")
    if len({row["task_id"] for row in calibration}) != 32:
        raise ValueError("duplicated calibration task IDs")

    proposals_pin = config["trained_proposals"]
    _pinned_path(root, proposals_pin["checkpoint_manifest_path"], proposals_pin["checkpoint_manifest_sha256"])
    proposal_path = _pinned_path(root, proposals_pin["path"], proposals_pin["sha256"])
    rows: dict[str, dict[str, Any]] = {}
    for line in proposal_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "__calibration__" not in row["record_id"]:
            continue
        if row["record_id"] in rows:
            raise ValueError("duplicate frozen calibration proposal")
        rows[row["record_id"]] = row
    if len(rows) != len(calibration) * len(SEEDS):
        raise ValueError("incomplete frozen calibration proposal table")

    action_index = {a: i for i, a in enumerate(suite["action_vocab"])}
    checkpoint_pins = {int(row["seed"]): row for row in proposals_pin["final_checkpoints"]}
    if set(checkpoint_pins) != set(SEEDS):
        raise ValueError("unexpected final checkpoint seed roster")
    results = []
    with torch.inference_mode():
        for seed in SEEDS:
            pin = checkpoint_pins[seed]
            path = _pinned_path(root, pin["path"], pin["sha256"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload["seed"], payload["step"]) != (seed, 200):
                raise ValueError("checkpoint identity mismatch")
            if tuple(payload["action_vocab"]) != tuple(suite["action_vocab"]):
                raise ValueError("checkpoint action vocabulary mismatch")
            if (payload["feature_dim"], payload["latent_dim"], payload["recurrence_steps"]) != (44, 48, 4):
                raise ValueError("checkpoint architecture mismatch")
            model = ControlTRMProposer(44, len(action_index), latent_dim=48, recurrence_steps=4).cpu().eval()
            model.load_state_dict(payload["state_dict"], strict=True)
            if sum(p.numel() for p in model.parameters()) != 17_203:
                raise ValueError("checkpoint parameter count mismatch")
            before = {name: value.clone() for name, value in model.state_dict().items()}
            max_replay = max_sham = max_rescue = 0.0
            ranked_agreement = 0
            for index, task in enumerate(calibration):
                donor_task = next(
                    calibration[j] for j in range(len(calibration))
                    if j != index and calibration[j]["family"] == task["family"]
                )
                if task["candidates"] != donor_task["candidates"]:
                    raise ValueError("donor candidate set differs")
                features = torch.tensor(task["public_features"], dtype=torch.float32).unsqueeze(0)
                donor_features = torch.tensor(donor_task["public_features"], dtype=torch.float32).unsqueeze(0)
                baseline = model(features)
                own = latent_after_two(model, features)
                if not torch.allclose(own, baseline.latents[:, 1], atol=MAX_SHAM_ERROR, rtol=0):
                    raise ValueError("step-two reconstruction failed")
                sham = continue_from_two(model, features, own.clone())
                max_sham = max(max_sham, (sham - baseline.action_logits).abs().max().item())
                if max_sham > MAX_SHAM_ERROR:
                    raise ValueError("sham changed the original action logits")
                # Exercise donor patch without looking at its effect or choosing donors from outputs.
                donor = latent_after_two(model, donor_features)
                injected = continue_from_two(model, features, donor)
                if not torch.isfinite(injected).all().item():
                    raise ValueError("donor injection produced nonfinite logits")
                rescued = continue_from_two(model, features, own)
                max_rescue = max(max_rescue, (rescued - baseline.action_logits).abs().max().item())
                if max_rescue > MAX_SHAM_ERROR:
                    raise ValueError("restoring own h2 did not rescue logits")
                reference = rows[f"seed-{seed}__{task['task_id']}"]
                scores = {action: float(baseline.action_logits[0, action_index[action]].item()) for action in task["candidates"]}
                delta = max(abs(scores[action] - reference["scores"][action]) for action in scores)
                max_replay = max(max_replay, delta)
                if delta > MAX_LOGIT_ERROR:
                    raise ValueError("checkpoint logits disagreed with frozen proposal table")
                order = sorted(task["candidates"], key=lambda a: (-scores[a], task["candidates"].index(a)))
                if order != reference["ranked_actions"]:
                    raise ValueError("checkpoint action ranking disagreed with frozen proposal")
                ranked_agreement += 1
            if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
                raise ValueError("model parameters changed during read-only verification")
            results.append({
                "seed": seed, "calibration_tasks": ranked_agreement,
                "candidate_logit_max_abs_error": max_replay,
                "sham_max_abs_error": max_sham,
                "rescue_max_abs_error": max_rescue,
                "checkpoint_sha256": pin["sha256"],
            })
    return {
        "schema": "control_trm_step_two_calibration_v1",
        "status": "passed", "task_suite_sha256": suite_pin["sha256"],
        "calibration_families": dict(sorted(families.items())), "results": results,
        "training_performed": False, "held_out_evaluated": False,
        "donor_effect_scored": False, "files_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(verify(args.repo_root), sort_keys=True))


if __name__ == "__main__":
    main()
