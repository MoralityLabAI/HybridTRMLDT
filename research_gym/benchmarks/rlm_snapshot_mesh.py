"""Snapshot-first official-RLM wrapper around the typed controller mesh."""

from __future__ import annotations

from collections import Counter
import json
import time
from typing import Any, Mapping

from research_gym.benchmarks.rlm_architecture_neighborhood import (
    trajectory_metrics,
    usage_totals,
)
from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    LongContextControlTask,
    canonical_sha256,
    ldt_action,
)
from research_gym.benchmarks.rlm_hybrid_runtime import (
    _base_record,
    _proposal_for,
    _restricted_local_environment,
)
from research_gym.benchmarks.rlm_wrapped_mesh import (
    MESH_TOPOLOGY_HASH,
    POLICY_HINTS,
    _parse_policy_hint,
    _safe_top,
    resolve_mesh,
    run_rlm_wrapped_mesh,
    verify_mesh_resolution,
)


ARCHITECTURES = (
    "ldt_only",
    "trained_trm_ldt_fixed",
    "mesh_fixed_consensus",
    "mesh_forced_no_tool_fallback",
    "rlm_mesh_task_first_atomic",
    "rlm_mesh_snapshot_atomic",
    "rlm_mesh_snapshot_text",
)
API_ARCHITECTURES = (
    "rlm_mesh_task_first_atomic",
    "rlm_mesh_snapshot_atomic",
    "rlm_mesh_snapshot_text",
)
LOCAL_ARCHITECTURES = tuple(value for value in ARCHITECTURES if value not in API_ARCHITECTURES)
SNAPSHOT_POLICY_DESCRIPTIONS = {
    "consensus": "Use the majority exact-admissible module recommendation, with exact-LDT tie breaking.",
    "trained_first": "Use the trained module only when its top candidate is exact-admissible and confidence is at least 0.65; otherwise use consensus.",
    "ldt_conservative": "Use the exact-LDT recommendation directly.",
}
SNAPSHOT_CONTRACT = {
    "version": "opaque_mesh_policy_snapshot_v0_1",
    "visible": (
        "opaque_candidate_references",
        "module_confidence",
        "module_provenance",
        "exact_admissibility",
        "module_agreement",
        "policy_previews",
    ),
    "hidden": (
        "task_text",
        "transcript",
        "action_tokens",
        "utilities",
        "optimal_action",
    ),
    "authority": {
        "rlm": "select_one_policy_hint",
        "mesh": "replay_policy_against_immutable_snapshot",
        "executor": "map_opaque_reference_and_commit_certificate_bound_action",
    },
}
SNAPSHOT_CONTRACT_HASH = canonical_sha256(SNAPSHOT_CONTRACT)


def architecture_hashes() -> dict[str, str]:
    specifications = {
        "ldt_only": (0, "ldt", "direct"),
        "trained_trm_ldt_fixed": (0, "trained", "typed_fixed"),
        "mesh_fixed_consensus": (0, MESH_TOPOLOGY_HASH, "consensus"),
        "mesh_forced_no_tool_fallback": (0, MESH_TOPOLOGY_HASH, "forced_fixed_fallback"),
        "rlm_mesh_task_first_atomic": (1, MESH_TOPOLOGY_HASH, "task_first_atomic_v0"),
        "rlm_mesh_snapshot_atomic": (
            1,
            MESH_TOPOLOGY_HASH,
            SNAPSHOT_CONTRACT_HASH,
            "snapshot_first_atomic_replay",
        ),
        "rlm_mesh_snapshot_text": (
            1,
            MESH_TOPOLOGY_HASH,
            SNAPSHOT_CONTRACT_HASH,
            "snapshot_first_text_replay",
        ),
    }
    return {name: canonical_sha256(value) for name, value in specifications.items()}


def _candidate_references(task: LongContextControlTask) -> dict[str, str]:
    references = {
        action: "candidate_"
        + canonical_sha256(("opaque-mesh-candidate-v0.1", task.prompt_sha256, action))[:12]
        for action in task.candidates
    }
    if len(set(references.values())) != len(references):
        raise RuntimeError("opaque candidate-reference collision")
    return references


def build_opaque_snapshot(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a policy snapshot containing no task text, action tokens, or utility values."""

    references = _candidate_references(task)
    proxy_top, proxy_ranked, proxy_confidence, proxy_provenance = _proposal_for(
        task, "proxy", trained_row
    )
    trained_top, trained_ranked, trained_confidence, trained_provenance = _proposal_for(
        task, "trained", trained_row
    )
    ldt_top = ldt_action(task)
    top_actions = (proxy_top, trained_top, ldt_top)
    vote_counts = Counter(references[action] for action in top_actions)
    proxy_safe_top = _safe_top(task, proxy_ranked)
    trained_safe_top = _safe_top(task, trained_ranked)
    policy_previews = {}
    for hint in POLICY_HINTS:
        resolution = resolve_mesh(task, trained_row, hint)
        policy_previews[hint] = {
            "selected_candidate_ref": references[resolution.action],
            "arbitration_reason": resolution.snapshot["arbitration_reason"],
        }
    material = {
        "contract_hash": SNAPSHOT_CONTRACT_HASH,
        "mesh_topology_hash": MESH_TOPOLOGY_HASH,
        "case_ref": canonical_sha256(("snapshot-case-v0.1", task.prompt_sha256))[:20],
        "candidate_refs": sorted(references.values()),
        "exact_admissible_refs": sorted(references[action] for action in task.exact_allowed),
        "modules": {
            "proxy_trm": {
                "top_ref": references[proxy_top],
                "safe_top_ref": references[proxy_safe_top] if proxy_safe_top is not None else None,
                "top_is_admissible": proxy_top in task.exact_allowed,
                "confidence": round(float(proxy_confidence), 8),
                "provenance": proxy_provenance,
            },
            "trained_control_trm": {
                "top_ref": references[trained_top],
                "safe_top_ref": references[trained_safe_top] if trained_safe_top is not None else None,
                "top_is_admissible": trained_top in task.exact_allowed,
                "confidence": round(float(trained_confidence), 8),
                "provenance": trained_provenance,
            },
            "exact_ldt": {
                "top_ref": references[ldt_top],
                "top_is_admissible": True,
                "provenance": "environment_sound",
            },
        },
        "agreement": {
            "all_equal": len(set(top_actions)) == 1,
            "distinct_top_count": len(set(top_actions)),
            "top_vote_counts": dict(sorted(vote_counts.items())),
        },
        "policy_previews": policy_previews,
        "policy_descriptions": SNAPSHOT_POLICY_DESCRIPTIONS,
    }
    return {"snapshot_id": canonical_sha256(material), **material}


def verify_opaque_snapshot(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> bool:
    return dict(snapshot) == build_opaque_snapshot(task, trained_row)


def snapshot_prompt(snapshot: Mapping[str, Any], *, atomic: bool) -> str:
    instruction = (
        "Call mesh_replay exactly once with snapshot_ref equal to snapshot_id and one policy_hint."
        if atomic
        else "Return exactly one policy_hint: consensus, trained_first, or ldt_conservative."
    )
    return (
        "Review this immutable controller-mesh policy snapshot. The candidate references are opaque; "
        "do not infer task content or action semantics. Select policy from evidence quality, confidence, "
        "agreement, provenance, and admissibility only.\nSNAPSHOT_START\n"
        + json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        + "\nSNAPSHOT_END\n"
        + instruction
    )


def _run_snapshot_rlm(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import rlm.core.rlm as core_module
    from rlm import RLM
    from rlm.logger import RLMLogger

    atomic = architecture_id == "rlm_mesh_snapshot_atomic"
    snapshot = build_opaque_snapshot(task, trained_row)
    restricted = _restricted_local_environment()
    original_get_environment = core_module.get_environment

    def get_restricted_environment(environment, kwargs):
        if environment != "local":
            raise ValueError("snapshot-mesh RLM permits only the restricted local environment")
        return restricted(**kwargs)

    core_module.get_environment = get_restricted_environment
    state: dict[str, Any] = {"calls": [], "invalid_calls": []}
    tools: dict[str, Any] = {}
    if atomic:

        def mesh_replay(snapshot_ref: str, policy_hint: str = "consensus"):
            """Replay one policy against the immutable opaque snapshot; returns an opaque candidate receipt."""
            hint = str(policy_hint).strip().lower()
            valid = (
                len(state["calls"]) == 0
                and str(snapshot_ref) == snapshot["snapshot_id"]
                and hint in POLICY_HINTS
                and verify_opaque_snapshot(task, trained_row, snapshot)
            )
            if not valid:
                payload = {
                    "accepted": False,
                    "reason": "snapshot_or_single_call_policy_contract_failed",
                    "valid_policy_hints": list(POLICY_HINTS),
                }
                state["invalid_calls"].append(
                    {"snapshot_ref": str(snapshot_ref), "policy_hint": hint, "result": payload}
                )
                return payload
            resolution = resolve_mesh(task, trained_row, hint)
            selected_ref = _candidate_references(task)[resolution.action]
            payload = {
                "accepted": True,
                "snapshot_ref": snapshot["snapshot_id"],
                "selected_candidate_ref": selected_ref,
                "policy_hint": hint,
                "certificate": resolution.certificate,
            }
            state["calls"].append(
                {"policy_hint": hint, "resolution": resolution, "result": payload}
            )
            return payload

        tools["mesh_replay"] = {"tool": mesh_replay, "description": mesh_replay.__doc__}

    engine = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": runtime["model"],
            "max_retries": int(runtime["provider_max_retries"]),
        },
        environment="local",
        max_depth=1,
        max_iterations=int(runtime["root_max_iterations"]),
        max_timeout=float(runtime["cell_timeout_seconds"]),
        max_tokens=int(runtime["cell_token_soft_cap"]),
        max_errors=int(runtime["max_consecutive_errors"]),
        max_concurrent_subcalls=1,
        sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
        sub_sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
        custom_tools=tools,
        custom_sub_tools={},
        user_prologue=(
            "Registered restricted environment: context is in memory; filesystem, subprocess, and arbitrary "
            "network access are disabled. You review an immutable opaque mesh snapshot. You have policy-hint "
            "authority only; the typed mesh retains candidate selection and action-commit authority."
        ),
        logger=RLMLogger(),
        verbose=False,
    )
    started = time.perf_counter()
    try:
        result = engine.completion(snapshot_prompt(snapshot, atomic=atomic))
    finally:
        engine.close()
        core_module.get_environment = original_get_environment
    elapsed = time.perf_counter() - started
    usage = usage_totals(result.usage_summary.to_dict())
    response = result.response
    if atomic:
        contract_passed = len(state["calls"]) == 1 and not state["invalid_calls"]
        if contract_passed:
            resolution = state["calls"][0]["resolution"]
            selected_hint = resolution.policy_hint
            reason = "rlm_snapshot_atomic_mesh_replay"
        else:
            selected_hint = "consensus"
            resolution = resolve_mesh(task, trained_row, selected_hint)
            reason = "rlm_snapshot_atomic_contract_fixed_mesh_fallback"
    else:
        parsed = _parse_policy_hint(response)
        contract_passed = parsed is not None
        selected_hint = parsed or "consensus"
        resolution = resolve_mesh(task, trained_row, selected_hint)
        reason = (
            "rlm_snapshot_text_mesh_replay"
            if contract_passed
            else "rlm_snapshot_text_parse_fixed_mesh_fallback"
        )
    if not verify_opaque_snapshot(task, trained_row, snapshot) or not verify_mesh_resolution(
        task, trained_row, resolution
    ):
        contract_passed = False
        selected_hint = "consensus"
        resolution = resolve_mesh(task, trained_row, selected_hint)
        reason = "snapshot_or_certificate_failure_fixed_mesh_fallback"
    references = _candidate_references(task)
    wrapper_fallback = not contract_passed
    record = _base_record(
        task,
        architecture_id,
        replicate_seed,
        resolution.action,
        resolution.action,
        wrapper_fallback,
        reason,
        usage,
        elapsed,
    )
    record.update(
        {
            "policy_hint": selected_hint,
            "mesh_certificate": resolution.certificate,
            "mesh_snapshot": dict(resolution.snapshot),
            "opaque_snapshot_id": snapshot["snapshot_id"],
            "snapshot_contract_hash": SNAPSHOT_CONTRACT_HASH,
            "context_mode": "snapshot_first",
            "selected_candidate_ref": references[resolution.action],
            "policy_changed_from_consensus": selected_hint != "consensus",
            "wrapper_contract_passed": contract_passed,
            "wrapper_fallback": wrapper_fallback,
            "mesh_call_count": len(state["calls"]),
            "invalid_mesh_call_count": len(state["invalid_calls"]),
        }
    )
    trajectory = {
        "architecture": architecture_id,
        "response": response,
        "official_completion": result.to_dict(),
        "usage": usage,
        "trajectory_metrics": trajectory_metrics(result.metadata),
        "opaque_snapshot": snapshot,
        "mesh_calls": [
            {"policy_hint": row["policy_hint"], "result": row["result"]}
            for row in state["calls"]
        ],
        "invalid_mesh_calls": state["invalid_calls"],
        "resolution": resolution.to_jsonable(),
    }
    return record, trajectory


def run_snapshot_mesh_architecture(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if architecture_id == "rlm_mesh_task_first_atomic":
        record, trajectory = run_rlm_wrapped_mesh(
            "rlm_mesh_atomic_tool", task, trained_row, replicate_seed, runtime
        )
        record["architecture_id"] = architecture_id
        record["context_mode"] = "task_first"
        record["snapshot_contract_hash"] = None
        record["opaque_snapshot_id"] = None
        record["selected_candidate_ref"] = _candidate_references(task).get(
            str(record["executed_action"])
        )
        record["policy_changed_from_consensus"] = record.get("policy_hint") != "consensus"
        trajectory["architecture"] = architecture_id
        return record, trajectory
    if architecture_id not in {"rlm_mesh_snapshot_atomic", "rlm_mesh_snapshot_text"}:
        raise ValueError(f"not a snapshot-mesh architecture: {architecture_id}")
    return _run_snapshot_rlm(
        architecture_id, task, trained_row, replicate_seed, runtime
    )
