"""Export research trajectories as DataGoblin-compatible JSONL records.

Row shape follows the rlt_gym research-row convention used by earlier Inky exports
({id, kind, root, family, split, input, target, loss, source_model, verification}).
Each architecture iteration keeps: proposal, rationale, predicted effect, observed
outcome, the diagnosis and revision Inky wrote on seeing the outcome (from the next
proposal), and the decision taken by the campaign. A final record binds the frozen
finalists to hidden-root evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "datagoblin.molt_architecture_trajectory.v1"
MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"


def _sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(state_path: Path, out: Path, analysis_path: Path | None = None, freeze_path: Path | None = None) -> dict:
    st = json.loads(state_path.read_text(encoding="utf-8"))
    root = "molt-goblin-02:" + _sha({"started": st["started_at"]})[:12]
    hist = [h for h in st["history"]]
    proposals = [h for h in hist if h["kind"] == "proposal"]
    promoted = {p["id"]: p for p in st.get("promotions", [])}
    rows = []
    for i, h in enumerate(proposals):
        nxt = proposals[i + 1] if i + 1 < len(proposals) else None
        cid = h.get("candidate_id")
        if h.get("error"):
            decision = "rejected_before_training" if "candidate_id" not in h else "recorded_failure"
        elif cid in promoted:
            decision = f"promoted_to_rung_{promoted[cid]['to']}"
        else:
            decision = "kept_at_rung_0"
        target = {"hypothesis": h.get("hypothesis"), "mechanism": h.get("mechanism"), "base": h.get("base"),
                  "candidate_change": h.get("candidate"), "control_change": h.get("control"),
                  "rationale": h.get("rationale"), "predicted_effect": h.get("predicted")}
        outcome = {"observed": h.get("observed"), "verdict": h.get("verdict"), "error": h.get("error"),
                   "diagnosis": nxt.get("diagnosis") if nxt else None, "revision": nxt.get("revision") if nxt else None,
                   "decision": decision}
        context = {"iteration": h["iteration"], "prior_diagnosis": h.get("diagnosis"), "prior_revision": h.get("revision")}
        rows.append({"id": _sha([root, h["iteration"], target]), "schema": SCHEMA, "kind": "architecture_trajectory",
                     "root": root, "family": "molt_architecture_discovery", "split": "train",
                     "semantic_task_id": f"{root}:it{h['iteration']}", "input": context, "target": target,
                     "outcome": outcome,
                     "loss": {"boundary": "response_only", "field": "target", "eligible": not h.get("error")},
                     "source_model": {"provider": "tinker" if h.get("proposer") == "inky" else "deterministic",
                                      "model": MODEL if h.get("proposer") == "inky" else None},
                     "provider_execution_attested": False,
                     "verification": {"kind": "cpu_training_receipts", "scorer": "gym.score exact sequence match",
                                      "roots": "development (DEV_BASE)", "candidate_id": cid,
                                      "control_id": h.get("control_id")}})
    for h in hist:
        if h["kind"] == "adversarial_task":
            rows.append({"id": _sha([root, "adv", h["iteration"], h.get("spec")]), "schema": SCHEMA,
                         "kind": "adversarial_task_proposal", "root": root, "family": "molt_architecture_discovery",
                         "split": "train", "input": {"iteration": h["iteration"]},
                         "target": {"explanations": h.get("explanations"), "spec": h.get("spec"),
                                    "prediction": h.get("prediction")},
                         "outcome": {"accepted": h.get("accepted"), "error": h.get("error"),
                                     "verification": h.get("verification")},
                         "loss": {"boundary": "response_only", "field": "target", "eligible": bool(h.get("accepted"))},
                         "source_model": {"provider": "tinker", "model": MODEL}, "provider_execution_attested": False,
                         "verification": {"kind": "exact_generator_solver_agreement"}})
    if analysis_path and analysis_path.exists() and freeze_path and freeze_path.exists():
        an = json.loads(analysis_path.read_text(encoding="utf-8"))
        fr = json.loads(freeze_path.read_text(encoding="utf-8"))
        rows.append({"id": _sha([root, "final", fr["freeze_sha256"]]), "schema": SCHEMA, "kind": "final_decision",
                     "root": root, "family": "molt_architecture_discovery", "split": "train",
                     "input": {"freeze_sha256": fr["freeze_sha256"], "arms": {k: v.get("kind") for k, v in fr["arms"].items()}},
                     "target": {"pareto_front": an["pareto_front"], "points": an["points"]},
                     "outcome": {"paired_vs_reference": an["paired_vs_reference"]},
                     "loss": {"boundary": "response_only", "field": "target", "eligible": False},
                     "source_model": {"provider": "campaign"}, "provider_execution_attested": False,
                     "verification": {"kind": "hidden_root_cpu_training", "roots": "HIDDEN_BASE"}})
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trajectories.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = {"schema_version": SCHEMA, "rows": len(rows), "root": root,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kinds": {k: sum(r["kind"] == k for r in rows) for k in {r["kind"] for r in rows}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--freeze", type=Path)
    a = ap.parse_args()
    print(json.dumps(build(a.state, a.out, a.analysis, a.freeze)))


if __name__ == "__main__":
    main()
