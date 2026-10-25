"""A disposable synthetic repository; never opens a held-out task for scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from research_gym.neural.control_trm import ControlTRMProposer
from research_gym.scripts.verify_control_trm_latent_calibration import verify

ACTIONS = ("approve", "quarantine", "revise", "escalate", "route_alpha", "route_beta", "route_gamma", "defer", "befriend", "investigate", "defuse", "rush", "wait", "accept", "reject", "reanchor")
FAMILIES = ("latest_rule_action", "multi_hop_reachability", "storyworld_control", "provenance_gate")
CANDIDATES = (ACTIONS[0:4], ACTIONS[4:8], ACTIONS[8:13], ("accept", "reject", "reanchor", "escalate"))


def _write_json(path: Path, data) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_repo(root: Path) -> None:
    tasks = []
    for family, candidates in zip(FAMILIES, CANDIDATES):
        for i in range(8):
            features = [0.0] * 44
            features[FAMILIES.index(family)] = 1.0
            features[4 + i] = 0.1 * i
            tasks.append({"task_id": f"{family}__calibration__{i:03d}", "split": "calibration", "family": family, "candidates": candidates, "public_features": features})
    # This intentionally malformed held-out row must never enter scoring.
    tasks.append({"task_id": "reserved-eval", "split": "eval", "public_features": [None]})
    suite = {"feature_dim": 44, "action_vocab": ACTIONS, "tasks": tasks}
    suite_path = "data/benchmarks/rlm_hybrid_long_context_tasks_v1.json"
    suite_hash = _write_json(root / suite_path, suite)
    checkpoint_rows = []
    proposal_rows = []
    for seed in (211, 223, 227):
        torch.manual_seed(seed)
        model = ControlTRMProposer(44, 16, latent_dim=48, recurrence_steps=4).cpu().eval()
        path = f"experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/checkpoints/seed-{seed}-step-200.pt"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seed": seed, "step": 200, "action_vocab": ACTIONS, "feature_dim": 44, "latent_dim": 48, "recurrence_steps": 4, "state_dict": model.state_dict()}, root / path)
        digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
        checkpoint_rows.append({"seed": seed, "path": path, "sha256": digest})
        with torch.inference_mode():
            for task in tasks[:32]:
                x = torch.tensor(task["public_features"], dtype=torch.float32).unsqueeze(0)
                logits = model(x).action_logits[0]
                scores = {a: round(float(logits[ACTIONS.index(a)].item()), 8) for a in task["candidates"]}
                ranked = sorted(task["candidates"], key=lambda a: (-scores[a], task["candidates"].index(a)))
                proposal_rows.append({"record_id": f"seed-{seed}__{task['task_id']}", "scores": scores, "ranked_actions": ranked})
    proposals_path = "experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/proposal_table.jsonl"
    (root / proposals_path).parent.mkdir(parents=True, exist_ok=True)
    (root / proposals_path).write_text("".join(json.dumps(x) + "\n" for x in proposal_rows), encoding="utf-8")
    proposals_sha = hashlib.sha256((root / proposals_path).read_bytes()).hexdigest()
    manifest_path = "experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/checkpoint_manifest.json"
    manifest_sha = _write_json(root / manifest_path, checkpoint_rows)
    cfg = {"protocol_id": "synthetic-control-trm", "task_suite": {"path": suite_path, "sha256": suite_hash}, "trained_proposals": {"path": proposals_path, "sha256": proposals_sha, "checkpoint_manifest_path": manifest_path, "checkpoint_manifest_sha256": manifest_sha, "final_checkpoints": checkpoint_rows}}
    cfg_sha = _write_json(root / "configs/rlm_trm_ldt_hybrid_neighborhood_v1_1.json", cfg)
    _write_json(root / "configs/rlm_trm_ldt_hybrid_neighborhood_v1_1_registration.json", {"protocol_id": "synthetic-control-trm", "config_sha256": cfg_sha})


def test_synthetic_calibration_replay_and_no_writes(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = verify(tmp_path)
    assert result["status"] == "passed"
    assert all(x["calibration_tasks"] == 32 for x in result["results"])
    assert all(x["sham_max_abs_error"] <= 1e-6 and x["rescue_max_abs_error"] <= 1e-6 for x in result["results"])
    assert all(x["candidate_logit_max_abs_error"] <= 2e-5 for x in result["results"])
    assert not result["held_out_evaluated"]
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_bad_checkpoint_hash_fails_before_load(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    p = tmp_path / "experiments/rlm_trm_ldt_hybrid_neighborhood_v1/training/checkpoints/seed-211-step-200.pt"
    p.write_bytes(p.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        verify(tmp_path)


def test_missing_calibration_task_fails_closed(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    p = tmp_path / "data/benchmarks/rlm_hybrid_long_context_tasks_v1.json"
    rows = json.loads(p.read_text())
    rows["tasks"] = [t for t in rows["tasks"] if t["task_id"] != "latest_rule_action__calibration__000"]
    _write_json(p, rows)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        verify(tmp_path)
