"""Typed controller mesh with an outer official-RLM policy wrapper."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
    rank_actions,
)
from research_gym.benchmarks.rlm_hybrid_runtime import (
    _base_record,
    _proposal_for,
    _restricted_local_environment,
)


POLICY_HINTS = ("consensus", "trained_first", "ldt_conservative")
ARCHITECTURES = (
    "ldt_only",
    "trained_trm_ldt_fixed",
    "mesh_fixed_consensus",
    "mesh_forced_no_tool_fallback",
    "rlm_mesh_atomic_tool",
    "rlm_mesh_text_router",
)
API_ARCHITECTURES = ("rlm_mesh_atomic_tool", "rlm_mesh_text_router")
LOCAL_ARCHITECTURES = tuple(value for value in ARCHITECTURES if value not in API_ARCHITECTURES)
MESH_TOPOLOGY = {
    "nodes": ("proxy_trm", "trained_control_trm", "exact_ldt", "arbiter", "executor"),
    "edges": (
        ("proxy_trm", "arbiter"),
        ("trained_control_trm", "arbiter"),
        ("exact_ldt", "arbiter"),
        ("arbiter", "executor"),
        ("exact_ldt", "executor"),
    ),
    "authority": {
        "rlm": "policy_hint_only",
        "arbiter": "candidate_selection",
        "exact_ldt": "admissibility_and_fallback",
        "executor": "certificate_bound_commit",
    },
}
MESH_TOPOLOGY_HASH = canonical_sha256(MESH_TOPOLOGY)


def architecture_hashes() -> dict[str, str]:
    specifications = {
        "ldt_only": (0, "ldt", "direct"),
        "trained_trm_ldt_fixed": (0, "trained", "typed_fixed"),
        "mesh_fixed_consensus": (0, MESH_TOPOLOGY_HASH, "consensus"),
        "mesh_forced_no_tool_fallback": (0, MESH_TOPOLOGY_HASH, "forced_fixed_fallback"),
        "rlm_mesh_atomic_tool": (1, MESH_TOPOLOGY_HASH, "atomic_tool"),
        "rlm_mesh_text_router": (1, MESH_TOPOLOGY_HASH, "text_policy_hint"),
    }
    return {name: canonical_sha256(value) for name, value in specifications.items()}


@dataclass(frozen=True)
class MeshResolution:
    action: str
    policy_hint: str
    certificate: str
    snapshot: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "policy_hint": self.policy_hint,
            "certificate": self.certificate,
            "snapshot": dict(self.snapshot),
        }


def _safe_top(task: LongContextControlTask, ranked: list[str]) -> str | None:
    return next((action for action in ranked if action in task.exact_allowed), None)


def _consensus_action(
    task: LongContextControlTask,
    proxy_ranked: list[str],
    trained_ranked: list[str],
    ldt: str,
) -> str:
    votes = Counter(
        action
        for action in (_safe_top(task, proxy_ranked), _safe_top(task, trained_ranked), ldt)
        if action is not None
    )
    order = {action: index for index, action in enumerate(task.candidates)}
    return min(votes, key=lambda action: (-votes[action], action != ldt, order[action]))


def resolve_mesh(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    policy_hint: str,
) -> MeshResolution:
    """Resolve one typed policy hint through the fixed inner controller mesh."""

    if policy_hint not in POLICY_HINTS:
        raise ValueError(f"unknown mesh policy hint: {policy_hint}")
    proxy_top, proxy_ranked, proxy_confidence, proxy_provenance = _proposal_for(
        task, "proxy", trained_row
    )
    trained_top, trained_ranked, trained_confidence, trained_provenance = _proposal_for(
        task, "trained", trained_row
    )
    ldt = ldt_action(task)
    consensus = _consensus_action(task, proxy_ranked, trained_ranked, ldt)
    if policy_hint == "ldt_conservative":
        action = ldt
        arbitration_reason = "ldt_conservative"
    elif (
        policy_hint == "trained_first"
        and trained_top in task.exact_allowed
        and trained_confidence >= 0.65
    ):
        action = trained_top
        arbitration_reason = "trained_confident_and_exact"
    else:
        action = consensus
        arbitration_reason = "safe_consensus"
    if action not in task.exact_allowed:
        raise AssertionError("inner mesh selected an inadmissible action")
    snapshot = {
        "task_id": task.task_id,
        "topology_hash": MESH_TOPOLOGY_HASH,
        "policy_hint": policy_hint,
        "proxy": {
            "top": proxy_top,
            "safe_top": _safe_top(task, proxy_ranked),
            "confidence": proxy_confidence,
            "provenance": proxy_provenance,
        },
        "trained": {
            "top": trained_top,
            "safe_top": _safe_top(task, trained_ranked),
            "confidence": trained_confidence,
            "provenance": trained_provenance,
        },
        "ldt": {"top": ldt, "provenance": "environment_sound"},
        "agreement": len({proxy_top, trained_top, ldt}) == 1,
        "selected_action": action,
        "arbitration_reason": arbitration_reason,
    }
    certificate_payload = {
        "task_id": task.task_id,
        "topology_hash": MESH_TOPOLOGY_HASH,
        "policy_hint": policy_hint,
        "action": action,
        "snapshot_hash": canonical_sha256(snapshot),
        "authority": "typed_mesh_executor",
    }
    return MeshResolution(
        action=action,
        policy_hint=policy_hint,
        certificate=canonical_sha256(certificate_payload),
        snapshot=snapshot,
    )


def verify_mesh_resolution(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    resolution: MeshResolution,
) -> bool:
    if resolution.action not in task.exact_allowed:
        return False
    expected = resolve_mesh(task, trained_row, resolution.policy_hint)
    return expected.action == resolution.action and expected.certificate == resolution.certificate


def run_local_mesh_architecture(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if architecture_id not in {"mesh_fixed_consensus", "mesh_forced_no_tool_fallback"}:
        raise ValueError(f"not a local mesh architecture: {architecture_id}")
    resolution = resolve_mesh(task, trained_row, "consensus")
    forced_fallback = architecture_id == "mesh_forced_no_tool_fallback"
    record = _base_record(
        task,
        architecture_id,
        replicate_seed,
        resolution.action,
        resolution.action,
        forced_fallback,
        "forced_fixed_mesh_fallback" if forced_fallback else "fixed_mesh_commit",
        {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reported_cost_usd": None},
        0.0,
    )
    record.update(
        {
            "policy_hint": resolution.policy_hint,
            "mesh_certificate": resolution.certificate,
            "mesh_snapshot": dict(resolution.snapshot),
            "wrapper_contract_passed": False if forced_fallback else None,
            "wrapper_fallback": forced_fallback,
            "mesh_call_count": 0,
        }
    )
    return record, {"architecture": architecture_id, "resolution": resolution.to_jsonable()}


def _parse_policy_hint(response: str) -> str | None:
    lowered = response.lower()
    matches = [(lowered.find(value), value) for value in POLICY_HINTS if value in lowered]
    matches = [value for value in matches if value[0] >= 0]
    return min(matches)[1] if matches else None


def run_rlm_wrapped_mesh(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run an official RLM outside a mesh that retains typed commit authority."""

    if architecture_id not in API_ARCHITECTURES:
        raise ValueError(f"not an RLM-wrapped mesh architecture: {architecture_id}")
    import rlm.core.rlm as core_module
    from rlm import RLM
    from rlm.logger import RLMLogger

    restricted = _restricted_local_environment()
    original_get_environment = core_module.get_environment

    def get_restricted_environment(environment, kwargs):
        if environment != "local":
            raise ValueError("RLM-wrapped mesh permits only the restricted local environment")
        return restricted(**kwargs)

    core_module.get_environment = get_restricted_environment
    state: dict[str, Any] = {"calls": [], "invalid_calls": []}
    tools: dict[str, Any] = {}
    if architecture_id == "rlm_mesh_atomic_tool":

        def mesh_resolve(policy_hint: str = "consensus"):
            """Atomically resolve one registered policy hint through the typed controller mesh."""
            hint = str(policy_hint).strip().lower()
            if len(state["calls"]) >= 1 or hint not in POLICY_HINTS:
                payload = {
                    "accepted": False,
                    "reason": "single_call_or_policy_contract_failed",
                    "valid_policy_hints": list(POLICY_HINTS),
                }
                state["invalid_calls"].append({"policy_hint": hint, "result": payload})
                return payload
            resolution = resolve_mesh(task, trained_row, hint)
            payload = {
                "accepted": True,
                "action": resolution.action,
                "policy_hint": resolution.policy_hint,
                "certificate": resolution.certificate,
                "agreement": resolution.snapshot["agreement"],
            }
            state["calls"].append({"policy_hint": hint, "resolution": resolution, "result": payload})
            return payload

        tools["mesh_resolve"] = {"tool": mesh_resolve, "description": mesh_resolve.__doc__}
        prompt = (
            task.model_prompt()
            + "\nChoose one policy hint from consensus, trained_first, ldt_conservative. "
            "Call mesh_resolve exactly once. Text cannot execute an action."
        )
        prologue = (
            "You wrap an atomic typed controller mesh. You have policy-hint authority only. "
            "The mesh retains proposal, verification, and action-commit authority."
        )
    else:
        prompt = (
            task.model_prompt()
            + "\nReturn exactly one mesh policy hint: consensus, trained_first, or ldt_conservative."
        )
        prologue = (
            "You advise an atomic typed controller mesh. Return a policy hint only; the host invokes "
            "the mesh and retains action authority."
        )
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
            "network access are disabled. "
            + prologue
        ),
        logger=RLMLogger(),
        verbose=False,
    )
    started = time.perf_counter()
    try:
        result = engine.completion(prompt)
    finally:
        engine.close()
        core_module.get_environment = original_get_environment
    elapsed = time.perf_counter() - started
    usage = usage_totals(result.usage_summary.to_dict())
    response = result.response
    if architecture_id == "rlm_mesh_atomic_tool":
        contract_passed = len(state["calls"]) == 1 and not state["invalid_calls"]
        resolution = state["calls"][0]["resolution"] if contract_passed else resolve_mesh(
            task, trained_row, "consensus"
        )
        selected_hint = resolution.policy_hint
        reason = "rlm_atomic_mesh_commit" if contract_passed else "rlm_atomic_contract_fixed_mesh_fallback"
    else:
        parsed = _parse_policy_hint(response)
        contract_passed = parsed is not None
        selected_hint = parsed or "consensus"
        resolution = resolve_mesh(task, trained_row, selected_hint)
        reason = "rlm_text_policy_mesh_commit" if contract_passed else "rlm_text_parse_fixed_mesh_fallback"
    verified = verify_mesh_resolution(task, trained_row, resolution)
    if not verified:
        contract_passed = False
        selected_hint = "consensus"
        resolution = resolve_mesh(task, trained_row, selected_hint)
        reason = "certificate_failure_fixed_mesh_fallback"
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
        "mesh_calls": [
            {"policy_hint": row["policy_hint"], "result": row["result"]} for row in state["calls"]
        ],
        "invalid_mesh_calls": state["invalid_calls"],
        "resolution": resolution.to_jsonable(),
    }
    return record, trajectory
