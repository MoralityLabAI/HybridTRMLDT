"""DataGoblin-compatible training records.

Row shape follows DataGoblin rlt_gym (corpus.py / inkling.py _aleph_rows), as used by
LoopedTransformers ink_fp_audit_02/exports.py:
  {id, kind, root, family, split, task_digest, input{...}, target, loss, source_model,
   provider_execution_attested, verification}
plus the JINN-GYM-01 fields: persona specification, candidate actions, confidence,
verifier outcome, repair, revised specification, measured utility, lineage, source hashes.
Accepted revisions go to training_records.jsonl; rejected or invalid ones to
failed_trajectories.jsonl, never mixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .inky import MODEL
from .schema import canonical

SCHEMA = "datagoblin.jinn_gym_01.persona_record.v1"


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    out = {}
    for p in sorted(here.glob("*.py")) + [here / "_vendor" / "ldt_experiment.py"]:
        out[p.relative_to(here.parent.parent).as_posix()] = hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def revision_record(*, task, persona: str, parent_spec: dict, parent_hash: str, child_spec: dict | None,
                    child_hash: str | None, candidate_actions: list, inky_response: dict, decision: dict,
                    verifier: dict, diagnostics: dict, patch: dict, measured: dict, accepted: bool,
                    generation: int, iteration: int, source_hashes: dict, failure: str | None = None) -> dict:
    rid = hashlib.sha256(f"{task.task_id}|{persona}|{parent_hash}|{child_hash}|{iteration}".encode()).hexdigest()[:24]
    return {
        "schema": SCHEMA, "id": rid,
        "kind": "persona_revision" if accepted else "persona_revision_rejected",
        "root": task.task_id, "family": f"{task.domain}/f{task.family}", "split": "train" if accepted else "failed",
        "task_digest": task.digest(),
        "input": {"task": task.public_view(), "persona": persona, "persona_spec": parent_spec,
                  "persona_spec_hash": parent_hash, "candidate_actions": candidate_actions,
                  "diagnostics": diagnostics},
        "target": {"repair": patch, "revised_spec": child_spec, "revised_spec_hash": child_hash},
        "loss": "sft_on_target" if accepted else "none",
        "source_model": MODEL, "provider_execution_attested": False,
        "inky_response": inky_response, "decision": decision,
        "confidence": decision.get("confidence"),
        "verification": verifier, "measured_utility": measured, "accepted": accepted, "failure": failure,
        "lineage": {"parent": parent_hash, "child": child_hash, "generation": generation, "iteration": iteration},
        "source_hashes": source_hashes,
    }


def decision_record(row: dict, task, spec_hash: str | None, inky: dict | None, source_hashes_digest: str) -> dict:
    rid = hashlib.sha256(f"{row['task_id']}|{row['persona']}|{row['arm']}|{spec_hash}".encode()).hexdigest()[:24]
    return {"schema": SCHEMA, "id": rid, "kind": "persona_decision", "root": task.task_id,
            "family": f"{task.domain}/f{task.family}", "split": task.split, "task_digest": task.digest(),
            "input": {"task": task.public_view(), "persona": row["persona"], "arm": row["arm"],
                      "persona_spec_hash": spec_hash},
            "target": {"executed": row["executed"]}, "loss": "none", "source_model": MODEL,
            "provider_execution_attested": False, "inky": inky,
            "verification": {k: row[k] for k in ("utility", "best_utility", "optimal", "signature_hit",
                                                 "unauthorized_execution", "abstain_optimal")},
            "confidence": row["confidence"], "source_hashes_digest": source_hashes_digest}


def digest_of(hashes: dict) -> str:
    return hashlib.sha256(canonical(hashes).encode()).hexdigest()


def append_jsonl(path: Path, obj: dict) -> None:
    import os
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
