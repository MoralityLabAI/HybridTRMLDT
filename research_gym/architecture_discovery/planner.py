"""Deterministic multi-fidelity planner for schedule-topology discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lsa.canonical import digest
from lsa.scale import solve_decoder_width
from lsa.topology import (
    ScheduleTopology,
    enumerate_schedule_grammar,
    periodic_control,
    random_schedule_null,
    select_discovery_batches,
    standardized_descriptor_vectors,
)
from research_gym.integrity import canonical_file_sha256


CLAIM_SCOPE = (
    "architecture discovery and held-out task transfer; novelty is limited to "
    "non-equivalence within the audited schedule grammar"
)


@dataclass(frozen=True)
class ArchitectureResourceForecast:
    scale_rung: str
    unique_parameters: int
    applied_parameters_per_token: int
    estimated_flops_per_example: int
    estimated_peak_vram_bytes: int
    microbatch_size: int
    gradient_accumulation: int
    executable: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArchitectureProposal:
    proposal_id: str
    proposal_hash: str
    architecture_hash: str
    batch: str
    role: str
    matched_control_id: str | None
    topology: Mapping[str, Any]
    descriptors: Mapping[str, float]
    models: Mapping[str, Mapping[str, Any]]
    theory: Mapping[str, Any]
    empirical: Mapping[str, Any]
    decision: Mapping[str, Any]
    resources: tuple[ArchitectureResourceForecast, ...]
    code_commit: str
    task_bundle_hash: str
    claim_scope: str = CLAIM_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return canonical_file_sha256(path)


def _model_for_scale(
    topology: ScheduleTopology,
    rung: Mapping[str, Any],
    shape_policy: Mapping[str, Any],
) -> dict[str, Any]:
    shape = solve_decoder_width(
        int(rung["target_unique_parameters"]),
        physical_blocks=topology.physical_modules,
        vocab_size=int(shape_policy["vocab_size"]),
        head_dim=int(shape_policy.get("architecture_head_dim", 8)),
        tolerance=float(shape_policy["target_tolerance"]),
    )
    return {
        "family": "looped_decoder_schedule_topology_v1",
        "scale_rung": str(rung["name"]),
        "target_unique_parameters": int(rung["target_unique_parameters"]),
        "unique_parameters": shape.unique_parameters,
        "embedding_parameters": shape.embedding_parameters,
        "core_parameters": shape.core_parameters,
        "hidden_size": shape.hidden_size,
        "num_heads": shape.num_heads,
        "head_dim": shape.head_dim,
        "physical_blocks": shape.physical_blocks,
        "vocab_size": shape.vocab_size,
        "relative_target_error": shape.relative_error,
        "activation_checkpointing": True,
        "precision": "amp_fp16",
    }


def _resources(
    topology: ScheduleTopology,
    models: Mapping[str, Mapping[str, Any]],
    *,
    scale_ladder: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
) -> tuple[ArchitectureResourceForecast, ...]:
    sequence = int(resource_profile["sequence_length"])
    effective_batch = int(resource_profile["effective_batch_size"])
    microbatches = resource_profile["microbatch_by_scale"]
    forecasts: list[ArchitectureResourceForecast] = []
    for rung in scale_ladder["rungs"]:
        name = str(rung["name"])
        model = models[name]
        microbatch = int(microbatches.get(name, 1))
        accumulation = max(1, effective_batch // microbatch)
        unique = int(model["unique_parameters"])
        core_per_block = int(model["core_parameters"]) // topology.physical_modules
        applied = int(model["embedding_parameters"]) + core_per_block * topology.train_visits
        hidden = int(model["hidden_size"])
        flops = int(
            effective_batch
            * topology.train_visits
            * sequence
            * (24 * hidden * hidden + 4 * sequence * hidden)
        )
        persistent = unique * 16
        activations = sequence * hidden * 2 * microbatch * (6 + topology.train_visits)
        peak = persistent + activations
        exclusions: list[str] = []
        if peak > int(resource_profile["vram_bytes"]):
            exclusions.append("estimated_vram_exceeds_cap")
        if name not in {"S0", "S1", "S2"}:
            exclusions.append("campaign_stops_at_30m")
        forecasts.append(
            ArchitectureResourceForecast(
                scale_rung=name,
                unique_parameters=unique,
                applied_parameters_per_token=applied,
                estimated_flops_per_example=flops,
                estimated_peak_vram_bytes=peak,
                microbatch_size=microbatch,
                gradient_accumulation=accumulation,
                executable=not exclusions,
                exclusion_reasons=tuple(exclusions),
            )
        )
    return tuple(forecasts)


def _proposal(
    *,
    proposal_id: str,
    topology: ScheduleTopology,
    batch: str,
    role: str,
    matched_control_id: str | None,
    code_commit: str,
    task_bundle_hash: str,
    scale_ladder: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
    descriptor_vector: Sequence[float],
) -> ArchitectureProposal:
    models = {
        str(rung["name"]): _model_for_scale(topology, rung, scale_ladder["shape_policy"])
        for rung in scale_ladder["rungs"]
    }
    resources = _resources(
        topology,
        models,
        scale_ladder=scale_ladder,
        resource_profile=resource_profile,
    )
    descriptors = asdict(topology.descriptors)
    control_distance = sum(value * value for value in descriptor_vector) ** 0.5
    theory = {
        "expanded_visits": topology.train_visits,
        "physical_modules": topology.physical_modules,
        "tied_fraction": 1.0 - topology.physical_modules / topology.train_visits,
        "descriptors": descriptors,
        "predicted_effect": (
            "schedule topology may alter visit alignment at fixed physical weights, parameters, and visit compute"
        ),
        "falsification_value": control_distance,
    }
    empirical = {
        "capability_prior": {"kind": "beta", "alpha": 1.0, "beta": 1.0},
        "finite_run_probability": None,
        "scale_transfer_probability": None,
        "measured_step_seconds": None,
        "source": "unmeasured_before_campaign",
    }
    decision = {
        "rankable": role == "candidate",
        "batch": batch,
        "selection": "deterministic_descriptor_farthest_point",
        "structural_diversity": control_distance,
        "estimated_information_gain": control_distance,
        "resource_calibration_required": True,
    }
    architecture_hash = topology.topology_hash
    proposal_core = {
        "proposal_id": proposal_id,
        "architecture_hash": architecture_hash,
        "batch": batch,
        "role": role,
        "matched_control_id": matched_control_id,
        "topology": topology.to_dict(),
        "models": models,
        "theory": theory,
        "empirical": empirical,
        "decision": decision,
        "resources": [asdict(value) for value in resources],
        "code_commit": code_commit,
        "task_bundle_hash": task_bundle_hash,
        "claim_scope": CLAIM_SCOPE,
    }
    return ArchitectureProposal(
        proposal_id=proposal_id,
        proposal_hash=digest(proposal_core),
        architecture_hash=architecture_hash,
        batch=batch,
        role=role,
        matched_control_id=matched_control_id,
        topology=topology.to_dict(),
        descriptors=descriptors,
        models=models,
        theory=theory,
        empirical=empirical,
        decision=decision,
        resources=resources,
        code_commit=code_commit,
        task_bundle_hash=task_bundle_hash,
    )


def generate_architecture_proposals(
    *,
    code_commit: str,
    task_manifest: Mapping[str, Any],
    scale_ladder: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
    search_space: Mapping[str, Any] | None = None,
) -> tuple[ArchitectureProposal, ...]:
    grammar = search_space or {
        "physical_modules": [2, 3, 4],
        "expanded_visits": [6, 8],
        "selected_per_stratum": 4,
    }
    module_counts = tuple(int(value) for value in grammar["physical_modules"])
    lengths = tuple(int(value) for value in grammar["expanded_visits"])
    if module_counts != (2, 3, 4) or lengths != (6, 8):
        raise ValueError("v1 freezes K={2,3,4} and L={6,8}")
    if int(grammar["selected_per_stratum"]) != 4:
        raise ValueError("v1 freezes four selected schedules per (K,L) stratum")
    pool = enumerate_schedule_grammar(modules=module_counts, lengths=lengths)
    discovery, reserve = select_discovery_batches(
        pool, per_stratum=int(grammar["selected_per_stratum"])
    )
    controls = {
        (modules, length): periodic_control(modules, length)
        for modules in module_counts
        for length in lengths
    }
    nulls = {
        key: random_schedule_null(pool, modules=key[0], length=key[1])
        for key in controls
    }
    reference = tuple(pool) + tuple(controls.values()) + tuple(nulls.values())
    vectors = standardized_descriptor_vectors(reference)
    proposals: list[ArchitectureProposal] = []
    control_ids: dict[tuple[int, int], str] = {}
    task_bundle_hash = str(task_manifest["bundle_hash"])
    for (modules, length), topology in sorted(controls.items()):
        proposal_id = f"LSAD-C-K{modules}L{length}"
        control_ids[(modules, length)] = proposal_id
        proposals.append(
            _proposal(
                proposal_id=proposal_id,
                topology=topology,
                batch="control",
                role="matched_periodic_control",
                matched_control_id=None,
                code_commit=code_commit,
                task_bundle_hash=task_bundle_hash,
                scale_ladder=scale_ladder,
                resource_profile=resource_profile,
                descriptor_vector=vectors[topology.topology_hash],
            )
        )
    for batch_name, values, prefix in (
        ("batch_1", discovery, "B1"),
        ("batch_2_reserve", reserve, "B2"),
    ):
        counters: dict[tuple[int, int], int] = {}
        for topology in values:
            key = (topology.physical_modules, topology.train_visits)
            counters[key] = counters.get(key, 0) + 1
            proposal_id = f"LSAD-{prefix}-K{key[0]}L{key[1]}-{counters[key]:02d}"
            proposals.append(
                _proposal(
                    proposal_id=proposal_id,
                    topology=topology,
                    batch=batch_name,
                    role="candidate",
                    matched_control_id=control_ids[key],
                    code_commit=code_commit,
                    task_bundle_hash=task_bundle_hash,
                    scale_ladder=scale_ladder,
                    resource_profile=resource_profile,
                    descriptor_vector=vectors[topology.topology_hash],
                )
            )
    for (modules, length), topology in sorted(nulls.items()):
        proposals.append(
            _proposal(
                proposal_id=f"LSAD-N-K{modules}L{length}",
                topology=topology,
                batch="null",
                role="balanced_schedule_null",
                matched_control_id=control_ids[(modules, length)],
                code_commit=code_commit,
                task_bundle_hash=task_bundle_hash,
                scale_ladder=scale_ladder,
                resource_profile=resource_profile,
                descriptor_vector=vectors[topology.topology_hash],
            )
        )
    context_topologies = (
        ScheduleTopology((0,), 1, 8, (4, 8, 12, 16), "fully_tied_context"),
        ScheduleTopology(tuple(range(8)), 8, 8, (4, 8), "fully_untied_context"),
    )
    for index, topology in enumerate(context_topologies, start=1):
        proposals.append(
            _proposal(
                proposal_id=f"LSAD-X-{index:02d}",
                topology=topology,
                batch="context",
                role=topology.topology_kind,
                matched_control_id=None,
                code_commit=code_commit,
                task_bundle_hash=task_bundle_hash,
                scale_ladder=scale_ladder,
                resource_profile=resource_profile,
                descriptor_vector=topology.descriptors.vector(),
            )
        )
    proposals.sort(key=lambda value: value.proposal_id)
    if len({proposal.proposal_hash for proposal in proposals}) != len(proposals):
        raise ValueError("proposal hash collision")
    return tuple(proposals)


def write_proposals(
    proposals: Sequence[ArchitectureProposal],
    directory: Path,
    *,
    input_files: Mapping[str, Path],
    input_root: Path | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    table_path = directory / "proposal_table.jsonl"
    with table_path.open("w", encoding="utf-8", newline="\n") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    derived_artifacts = {
        "theory_predictions.json": {
            proposal.proposal_id: proposal.theory for proposal in proposals
        },
        "empirical_predictions.json": {
            proposal.proposal_id: proposal.empirical for proposal in proposals
        },
        "decision_predictions.json": {
            proposal.proposal_id: proposal.decision for proposal in proposals
        },
        "resource_forecasts.json": {
            proposal.proposal_id: [asdict(value) for value in proposal.resources]
            for proposal in proposals
        },
    }
    derived_receipts: dict[str, Mapping[str, str]] = {}
    for name, value in derived_artifacts.items():
        path = directory / name
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        derived_receipts[name] = {"path": name, "sha256": _sha256(path)}
    role_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    for proposal in proposals:
        role_counts[proposal.role] = role_counts.get(proposal.role, 0) + 1
        batch_counts[proposal.batch] = batch_counts.get(proposal.batch, 0) + 1
    def display_path(path: Path) -> str:
        if input_root is not None:
            try:
                path = path.resolve().relative_to(input_root.resolve())
            except ValueError:
                pass
        return str(path).replace("\\", "/")

    manifest = {
        "schema_version": 1,
        "proposal_count": len(proposals),
        "role_counts": role_counts,
        "batch_counts": batch_counts,
        "proposal_table": {
            "path": "proposal_table.jsonl",
            "sha256": _sha256(table_path),
        },
        "derived_artifacts": derived_receipts,
        "inputs": {
            name: {"path": display_path(path), "sha256": _sha256(path)}
            for name, path in sorted(input_files.items())
        },
        "claim_scope": CLAIM_SCOPE,
    }
    manifest["manifest_hash"] = digest(manifest)
    (directory / "proposal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipt = {
        "schema_version": 1,
        "manifest_path": "proposal_manifest.json",
        "manifest_sha256": _sha256(directory / "proposal_manifest.json"),
        "proposal_table_sha256": _sha256(table_path),
        "proposal_hashes": {
            proposal.proposal_id: proposal.proposal_hash for proposal in proposals
        },
        "code_commits": sorted({proposal.code_commit for proposal in proposals}),
        "task_bundle_hashes": sorted({proposal.task_bundle_hash for proposal in proposals}),
    }
    (directory / "proposal_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def read_proposals(directory: Path) -> tuple[ArchitectureProposal, ...]:
    manifest = json.loads((directory / "proposal_manifest.json").read_text(encoding="utf-8"))
    claimed_manifest_hash = manifest.pop("manifest_hash")
    if digest(manifest) != claimed_manifest_hash:
        raise ValueError("proposal manifest hash mismatch")
    for artifact in manifest["derived_artifacts"].values():
        path = directory / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"derived proposal artifact hash mismatch: {artifact['path']}")
    table = directory / manifest["proposal_table"]["path"]
    if _sha256(table) != manifest["proposal_table"]["sha256"]:
        raise ValueError("proposal table hash mismatch")
    values: list[ArchitectureProposal] = []
    for line in table.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        proposal_core = {
            key: value
            for key, value in row.items()
            if key not in {"proposal_hash", "descriptors"}
        }
        if digest(proposal_core) != row["proposal_hash"]:
            raise ValueError(f"proposal hash mismatch: {row['proposal_id']}")
        row["resources"] = tuple(ArchitectureResourceForecast(**value) for value in row["resources"])
        values.append(ArchitectureProposal(**row))
    return tuple(values)
