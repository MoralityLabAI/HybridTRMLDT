"""Sealed stage manifests for the LSPG architecture campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsa.canonical import digest

from .planner import ArchitectureProposal


@dataclass(frozen=True)
class CampaignCell:
    cell_id: str
    stage_id: str
    training_stage: str
    proposal_id: str
    proposal_hash: str
    role: str
    scale_rung: str
    seed: int
    token_visit_budget: int
    allow_locked_evaluation: bool


def _training_stage(stage_id: str) -> str:
    value = stage_id.split("_", 1)[0]
    if value not in {"A1", "A2", "B", "C", "D"}:
        raise ValueError(f"unknown campaign stage: {stage_id}")
    return value


def _proposal_ids_for_stage(
    stage_id: str,
    candidate_ids: Sequence[str],
    proposals: Mapping[str, ArchitectureProposal],
) -> tuple[str, ...]:
    training_stage = _training_stage(stage_id)
    candidates = [proposals[value] for value in candidate_ids]
    if any(value.role != "candidate" for value in candidates):
        raise ValueError("campaign candidate list contains a non-candidate")
    if stage_id == "A1" and any(value.batch != "batch_1" for value in candidates):
        raise ValueError("A1 may contain only the sealed first batch")
    if stage_id == "A1_reserve" and any(
        value.batch != "batch_2_reserve" for value in candidates
    ):
        raise ValueError("A1_reserve may contain only the sealed reserve")
    selected = set(candidate_ids)
    selected.update(
        value.matched_control_id
        for value in candidates
        if value.matched_control_id is not None
    )
    if training_stage == "D":
        for value in candidates:
            topology = value.topology
            selected.add(
                f"LSAD-N-K{int(topology['physical_modules'])}L{int(topology['train_visits'])}"
            )
        selected.update(
            value.proposal_id for value in proposals.values() if value.batch == "context"
        )
    return tuple(sorted(selected))


def build_stage_cells(
    *,
    stage_id: str,
    candidate_ids: Sequence[str],
    proposals: Mapping[str, ArchitectureProposal],
    policy: Mapping[str, Any],
    selected_profile: str,
) -> tuple[CampaignCell, ...]:
    training_stage = _training_stage(stage_id)
    if selected_profile not in policy["profiles"]:
        raise ValueError("campaign profile is not registered")
    stage_policy = policy["funnel"][training_stage]
    if len(candidate_ids) > int(stage_policy["candidate_limit"]):
        raise ValueError("candidate count exceeds registered stage limit")
    proposal_ids = _proposal_ids_for_stage(stage_id, candidate_ids, proposals)
    cells: list[CampaignCell] = []
    for proposal_id in proposal_ids:
        proposal = proposals[proposal_id]
        for seed in stage_policy["seeds"]:
            cells.append(
                CampaignCell(
                    cell_id=(
                        f"{proposal_id}-{training_stage}-{stage_policy['scale']}-s{int(seed)}"
                    ),
                    stage_id=stage_id,
                    training_stage=training_stage,
                    proposal_id=proposal_id,
                    proposal_hash=proposal.proposal_hash,
                    role=proposal.role,
                    scale_rung=str(stage_policy["scale"]),
                    seed=int(seed),
                    token_visit_budget=int(
                        policy["profiles"][selected_profile][training_stage]
                    ),
                    allow_locked_evaluation=training_stage == "D",
                )
            )
    return tuple(sorted(cells, key=lambda value: value.cell_id))


def write_stage_manifest(
    cells: Sequence[CampaignCell],
    path: Path,
    *,
    stage_id: str,
    selected_profile: str,
    profile_selection_hash: str,
    proposal_manifest_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "sealed_before_stage_outcomes",
        "stage_id": stage_id,
        "selected_profile": selected_profile,
        "profile_selection_hash": profile_selection_hash,
        "proposal_manifest_sha256": proposal_manifest_sha256,
        "code_commit": code_commit,
        "cell_count": len(cells),
        "cells": [asdict(value) for value in cells],
    }
    payload["manifest_hash"] = digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def read_stage_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("manifest_hash")
    if digest(payload) != claimed:
        raise ValueError("campaign stage manifest hash mismatch")
    payload["manifest_hash"] = claimed
    if payload["status"] != "sealed_before_stage_outcomes":
        raise ValueError("campaign stage is not sealed")
    return payload


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
