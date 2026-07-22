"""Official-RLM runtime for bounded two-round evidence acquisition."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from research_gym.benchmarks.rlm_architecture_neighborhood import (
    trajectory_metrics,
    usage_totals,
)
from research_gym.benchmarks.rlm_evidence_acquisition import (
    MAX_QUERIES,
    QUERY_IDS,
    acquisition_context,
    build_initial_snapshot,
    context_privacy_violations,
    execute_acquisition,
    execute_failure_fallback,
    make_evidence_receipt,
    parse_acquisition_decision,
)
from research_gym.benchmarks.rlm_hybrid_neighborhood import LongContextControlTask
from research_gym.benchmarks.rlm_hybrid_runtime import (
    _add_usage,
    _empty_usage,
    _restricted_local_environment,
)


def _error_summary(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "status_code": getattr(exc, "status_code", None),
        "provider_code": getattr(exc, "code", None),
    }


def _completion(
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], float]:
    import rlm.core.rlm as core_module
    from rlm import RLM
    from rlm.logger import RLMLogger

    restricted = _restricted_local_environment()
    original_get_environment = core_module.get_environment

    def get_restricted_environment(environment, kwargs):
        if environment != "local":
            raise ValueError("evidence-acquisition RLM permits only the restricted local environment")
        return restricted(**kwargs)

    core_module.get_environment = get_restricted_environment
    engine = None
    started = time.perf_counter()
    try:
        engine = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": runtime["model"],
                "max_retries": int(runtime["provider_max_retries"]),
            },
            environment="local",
            max_depth=1,
            max_iterations=int(runtime["root_max_iterations"]),
            max_timeout=float(runtime["round_timeout_seconds"]),
            max_tokens=int(runtime["round_token_soft_cap"]),
            max_errors=int(runtime["max_consecutive_errors"]),
            max_concurrent_subcalls=1,
            sampling_args={
                "max_tokens": int(runtime["max_output_tokens_per_call"]),
                "temperature": 0,
            },
            sub_sampling_args={
                "max_tokens": int(runtime["max_output_tokens_per_call"]),
                "temperature": 0,
            },
            custom_tools={},
            custom_sub_tools={},
            user_prologue=(
                "You are a bounded evidence-acquisition controller. context_0 is canonical JSON. "
                "You may select one available_query_id or STOP. You have no action, candidate, threshold, "
                "or schedule authority. Return only a bare registered query ID, bare STOP, or exactly one "
                "JSON object of the form {\"decision\":\"query\",\"query_id\":\"...\"} or "
                "{\"decision\":\"stop\"}. Do not add prose or code fences."
            ),
            logger=RLMLogger(),
            verbose=False,
        )
        result = engine.completion(json.dumps(context, sort_keys=True, separators=(",", ":")))
        return (
            result.response,
            usage_totals(result.usage_summary.to_dict()),
            {
                "official_completion": result.to_dict(),
                "trajectory_metrics": trajectory_metrics(result.metadata),
            },
            time.perf_counter() - started,
        )
    finally:
        if engine is not None:
            engine.close()
        core_module.get_environment = original_get_environment


def run_rlm_acquisition(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    truth: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = build_initial_snapshot(task, trained_row)
    sequence: list[str] = []
    receipts: list[dict[str, Any]] = []
    rounds = []
    usage = _empty_usage()
    elapsed = 0.0
    stopped = False
    invalid_decision = False
    provider_error = None
    for round_index in range(MAX_QUERIES):
        context = acquisition_context(snapshot, receipts, sequence)
        violations = context_privacy_violations(task, context)
        if violations:
            raise RuntimeError(f"provider context violated opacity contract: {violations}")
        try:
            response, round_usage, trajectory, round_elapsed = _completion(context, runtime)
        except Exception as exc:
            provider_error = _error_summary(exc)
            rounds.append(
                {
                    "round_index": round_index,
                    "context": context,
                    "error": provider_error,
                }
            )
            break
        usage = _add_usage(usage, round_usage)
        elapsed += round_elapsed
        available = [query_id for query_id in QUERY_IDS if query_id not in sequence]
        decision = parse_acquisition_decision(response, available)
        rounds.append(
            {
                "round_index": round_index,
                "context": context,
                "response": response,
                "parsed_decision": decision,
                "usage": round_usage,
                "execution_time": round_elapsed,
                **trajectory,
            }
        )
        if decision is None:
            invalid_decision = True
            break
        if decision == "STOP":
            stopped = True
            break
        sequence.append(decision)
        receipts.append(make_evidence_receipt(task, trained_row, truth, decision))
    failed = provider_error is not None or invalid_decision
    outcome = (
        execute_failure_fallback(task, trained_row, truth, sequence)
        if failed
        else execute_acquisition(task, trained_row, truth, sequence)
    )
    contract_passed = provider_error is None and not invalid_decision
    return (
        {
            "outcome": outcome,
            "usage": usage,
            "execution_time": elapsed,
            "contract_passed": contract_passed,
            "valid_decision_rounds": len(sequence) + int(stopped),
            "stopped": stopped,
            "invalid_decision": invalid_decision,
            "provider_error": provider_error,
        },
        {
            "snapshot": snapshot,
            "rounds": rounds,
            "query_sequence": list(sequence),
            "evidence_receipts": receipts,
            "outcome": outcome.to_jsonable(),
        },
    )
