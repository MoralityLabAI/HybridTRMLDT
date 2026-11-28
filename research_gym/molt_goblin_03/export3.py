"""Export Inky's research trajectory in the DataGoblin research-distillation row format.

One row per hypothesis cycle with the hypothesis, prediction, experiment (candidate and
control configurations), observation, the revision Inky wrote after seeing it, and the
campaign decision; one row per adversarial task proposal; one final row binding the
freeze to the hidden-root contrasts. Rows follow the rlt_gym convention
{id, kind, root, family, split, input, target, outcome, loss, source_model, verification}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "datagoblin.research_distillation.v1"
MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"


def _sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(state_path: Path, out: Path, analysis: Path | None = None, freeze: Path | None = None) -> dict:
    st = json.loads(state_path.read_text(encoding="utf-8"))
    root = "molt-goblin-03:" + _sha({"started": st["started_at"]})[:12]
    props = [h for h in st["history"] if h["kind"] == "proposal"]
    promoted = {}
    for p in st.get("promotions", []):
        promoted[p["id"]] = max(promoted.get(p["id"], 0), p["to"])
    rows = []
    for i, h in enumerate(props):
        nxt = props[i + 1] if i + 1 < len(props) else None
        cid = h.get("candidate_id")
        decision = ("rejected_duplicate" if "duplicate" in (h.get("error") or "") else
                    "rejected_before_training" if h.get("error") and not cid else
                    f"promoted_to_rung_{promoted[cid]}" if cid in promoted else "kept_at_rung_0")
        target = {"hypothesis": h.get("hypothesis"), "mechanism": h.get("mechanism"),
                  "prediction": h.get("predicted"), "rationale": h.get("rationale"),
                  "experiment": {"base": h.get("base"), "candidate_change": h.get("candidate"), "control_change": h.get("control")}}
        rows.append({"id": _sha([root, h["iteration"], target]), "schema": SCHEMA, "kind": "hypothesis_cycle", "root": root,
                     "family": "molt_mixture_source_discovery", "split": "train", "semantic_task_id": f"{root}:it{h['iteration']}",
                     "input": {"iteration": h["iteration"], "prior_diagnosis": h.get("diagnosis"), "prior_revision": h.get("revision")},
                     "target": target,
                     "outcome": {"observation": h.get("observed"), "prediction_check": h.get("verdict"), "error": h.get("error"),
                                 "revision": nxt.get("revision") if nxt else None, "diagnosis": nxt.get("diagnosis") if nxt else None,
                                 "decision": decision},
                     "loss": {"boundary": "response_only", "field": "target", "eligible": not h.get("error")},
                     "source_model": {"provider": "tinker" if h.get("proposer") == "inky" else "deterministic",
                                      "model": MODEL if h.get("proposer") == "inky" else None},
                     "provider_execution_attested": False,
                     "verification": {"kind": "cpu_training_receipts", "roots": "DEV3_BASE", "candidate_id": cid,
                                      "control_id": h.get("control_id")}})
    for h in st["history"]:
        if h["kind"] == "adversarial_task":
            rows.append({"id": _sha([root, "adv", h["iteration"], h.get("spec")]), "schema": SCHEMA, "kind": "adversarial_task",
                         "root": root, "family": "molt_mixture_source_discovery", "split": "train", "input": {"iteration": h["iteration"]},
                         "target": {"explanations": h.get("explanations"), "spec": h.get("spec"), "prediction": h.get("prediction")},
                         "outcome": {"accepted": h.get("accepted"), "error": h.get("error"), "verification": h.get("verification")},
                         "loss": {"boundary": "response_only", "field": "target", "eligible": bool(h.get("accepted"))},
                         "source_model": {"provider": "tinker", "model": MODEL}, "provider_execution_attested": False,
                         "verification": {"kind": "exact_generator_solver_agreement"}})
    if analysis and analysis.exists() and freeze and freeze.exists():
        an = json.loads(analysis.read_text(encoding="utf-8"))
        fr = json.loads(freeze.read_text(encoding="utf-8"))
        rows.append({"id": _sha([root, "final", fr["freeze_sha256"]]), "schema": SCHEMA, "kind": "final_evidence", "root": root,
                     "family": "molt_mixture_source_discovery", "split": "train",
                     "input": {"freeze_sha256": fr["freeze_sha256"], "arms": {k: v["kind"] for k, v in fr["arms"].items()}},
                     "target": {"pareto_front": an["pareto_front"], "retention_K_of_B": an.get("retention_K_of_B")},
                     "outcome": {"contrasts": an["contrasts"]},
                     "loss": {"boundary": "response_only", "field": "target", "eligible": False},
                     "source_model": {"provider": "campaign"}, "provider_execution_attested": False,
                     "verification": {"kind": "hidden_root_cpu_training", "roots": "HIDDEN3_BASE"}})
    out.mkdir(parents=True, exist_ok=True)
    path = out / "research_distillation.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    manifest = {"schema_version": SCHEMA, "rows": len(rows), "root": root, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kinds": {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})}}
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
