"""Runtime adapters for the registered RLM/TRM/LDT hybrid arms."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Mapping

from research_gym.benchmarks.rlm_architecture_neighborhood import trajectory_metrics, usage_totals
from research_gym.benchmarks.rlm_hybrid_neighborhood import (
    LongContextControlTask,
    ldt_action,
    parse_action,
    rank_actions,
    typed_execute,
)


API_ARCHITECTURES = (
    "rlm_repl_only",
    "rlm_ldt_membrane",
    "proxy_trm_rlm_critic_ldt",
    "trained_trm_rlm_critic_ldt",
    "rlm_tool_conductor",
    "rlm_recursive_conductor",
)

LOCAL_ARCHITECTURES = (
    "ldt_only",
    "proxy_trm_only",
    "trained_trm_only",
    "proxy_trm_ldt_fixed",
    "trained_trm_ldt_fixed",
)


def _restricted_local_environment():
    from rlm.environments.local_repl import LocalREPL

    original_import = __import__
    allowed_modules = {"collections", "json", "math", "re", "statistics"}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in allowed_modules:
            raise ImportError(f"module {root!r} is not permitted in the registered RLM hybrid REPL")
        return original_import(name, globals, locals, fromlist, level)

    class RestrictedLocalREPL(LocalREPL):
        def setup(self):
            super().setup()
            builtins = self.globals["__builtins__"]
            builtins["open"] = None
            builtins["__import__"] = restricted_import

        def load_context(self, context_payload):
            self.locals["context_0"] = context_payload
            self.locals["context"] = context_payload
            self._context_count = 1

    return RestrictedLocalREPL


def _empty_usage() -> dict[str, float | int | None]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reported_cost_usd": None,
    }


def _add_usage(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    costs = [value for value in (left.get("reported_cost_usd"), right.get("reported_cost_usd")) if value is not None]
    return {
        "calls": int(left.get("calls", 0)) + int(right.get("calls", 0)),
        "input_tokens": int(left.get("input_tokens", 0)) + int(right.get("input_tokens", 0)),
        "output_tokens": int(left.get("output_tokens", 0)) + int(right.get("output_tokens", 0)),
        "total_tokens": int(left.get("total_tokens", 0)) + int(right.get("total_tokens", 0)),
        "reported_cost_usd": sum(float(value) for value in costs) if costs else None,
    }


def _proposal_for(
    task: LongContextControlTask,
    backend: str,
    trained_row: Mapping[str, Any],
) -> tuple[str, list[str], float, str]:
    if backend == "proxy":
        ranked = rank_actions(task.proxy_scores, task.candidates)
        values = sorted((float(task.proxy_scores[action]) for action in task.candidates), reverse=True)
        confidence = 1.0 / (1.0 + pow(2.718281828, -(values[0] - values[1])))
        return ranked[0], ranked, confidence, "model_sound_proxy"
    if backend == "trained":
        ranked = [str(value) for value in trained_row["ranked_actions"]]
        return ranked[0], ranked, float(trained_row["confidence"]), str(trained_row["claimed_provenance"])
    raise ValueError(f"unknown TRM backend: {backend}")


def _base_record(
    task: LongContextControlTask,
    architecture_id: str,
    replicate_seed: int,
    proposal: str | None,
    executed: str,
    fallback: bool,
    reason: str,
    usage: Mapping[str, Any],
    execution_time: float,
) -> dict[str, Any]:
    optimal_utility = float(task.utilities[task.optimal_action])
    utility = float(task.utilities.get(executed, 0.0))
    safe = executed in task.exact_allowed
    return {
        "architecture_id": architecture_id,
        "task_id": task.task_id,
        "task_family": task.family,
        "replicate_seed": replicate_seed,
        "checkpoint_seed": replicate_seed if "trained_trm" in architecture_id or "conductor" in architecture_id else None,
        "proposal_action": proposal,
        "executed_action": executed,
        "optimal_action": task.optimal_action,
        "safe": safe,
        "unsafe": not safe,
        "correct": safe and executed == task.optimal_action,
        "utility": utility,
        "regret": optimal_utility - utility if safe else optimal_utility + 1.0,
        "fallback": fallback,
        "decision_reason": reason,
        "usage": dict(usage),
        "execution_time": float(execution_time),
    }


def run_local_architecture(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if architecture_id == "ldt_only":
        proposal = executed = ldt_action(task)
        fallback, reason = False, "explicit_ldt_top"
    elif architecture_id in {"proxy_trm_only", "proxy_trm_ldt_fixed"}:
        proposal, _, _, _ = _proposal_for(task, "proxy", trained_row)
        if architecture_id.endswith("ldt_fixed"):
            executed, fallback, reason = typed_execute(task, proposal)
        else:
            executed, fallback, reason = proposal, False, "proxy_trm_top"
    elif architecture_id in {"trained_trm_only", "trained_trm_ldt_fixed"}:
        proposal, _, _, _ = _proposal_for(task, "trained", trained_row)
        if architecture_id.endswith("ldt_fixed"):
            executed, fallback, reason = typed_execute(task, proposal)
        else:
            executed, fallback, reason = proposal, False, "trained_trm_top"
    else:
        raise ValueError(f"not a local architecture: {architecture_id}")
    record = _base_record(
        task, architecture_id, replicate_seed, proposal, executed, fallback, reason, _empty_usage(), 0.0
    )
    return record, {"architecture": architecture_id, "deterministic_local": True}


def _critic_prompt(task: LongContextControlTask, backend: str, trained_row: Mapping[str, Any]) -> str:
    _, ranked, confidence, provenance = _proposal_for(task, backend, trained_row)
    beam = ranked[:2]
    return (
        task.model_prompt()
        + "\nTRM_BEAM_START\n"
        + f"backend={backend}\nactions={','.join(beam)}\nconfidence={confidence:.6f}\n"
        + f"claimed_provenance={provenance}\nTRM_BEAM_END\n"
        + "Critique the bounded beam using the transcript. Return one candidate action token."
    )


def _run_child_delegate(task: LongContextControlTask, runtime: Mapping[str, Any], question: str) -> tuple[str, dict[str, Any]]:
    from rlm import RLM
    from rlm.logger import RLMLogger

    child = RLM(
        backend="openai",
        backend_kwargs={"model_name": runtime["model"], "max_retries": int(runtime["provider_max_retries"])},
        environment="local",
        max_depth=1,
        max_iterations=int(runtime["child_max_iterations"]),
        max_timeout=float(runtime["cell_timeout_seconds"]),
        max_tokens=int(runtime["child_token_soft_cap"]),
        max_errors=int(runtime["max_consecutive_errors"]),
        max_concurrent_subcalls=1,
        sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
        sub_sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
        custom_tools={},
        custom_sub_tools={},
        user_prologue=(
            "You are a soft-evidence delegate. You have no hard authority or host tools. Analyze the transcript "
            "and return a compact recommendation; do not claim certification."
        ),
        logger=RLMLogger(),
        verbose=False,
    )
    try:
        result = child.completion(f"{question}\n{task.model_prompt()}")
        return result.response, {
            "official_completion": result.to_dict(),
            "usage": usage_totals(result.usage_summary.to_dict()),
            "execution_time": result.execution_time,
            "trajectory_metrics": trajectory_metrics(result.metadata),
        }
    finally:
        child.close()


def _conductor_tools(
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    recursive: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state: dict[str, Any] = {
        "calls": [],
        "proposals": {},
        "certificates": {},
        "commits": [],
        "delegates": [],
        "child_usage": _empty_usage(),
        "child_time": 0.0,
    }

    def trm_propose(k: int = 2):
        """Return a bounded trained-ControlTRM proposal beam and receipt reference."""
        k = max(1, min(2, int(k)))
        top, ranked, confidence, provenance = _proposal_for(task, "trained", trained_row)
        ref = f"proposal-{len(state['proposals']) + 1}"
        payload = {
            "proposal_ref": ref,
            "top_action": top,
            "beam": ranked[:k],
            "confidence": confidence,
            "claimed_provenance": provenance,
        }
        state["proposals"][ref] = payload
        state["calls"].append({"tool": "trm_propose", "args": {"k": k}, "result": payload})
        return payload

    def ldt_verify(action: str):
        """Verify action admissibility under exact task mechanics; returns no utility."""
        action = str(action)
        allowed = action in task.exact_allowed
        ref = f"certificate-{len(state['certificates']) + 1}"
        payload = {
            "certificate_ref": ref,
            "action": action,
            "allowed": allowed,
            "verified_provenance": "environment_sound" if allowed else "environment_reject",
        }
        state["certificates"][ref] = payload
        state["calls"].append({"tool": "ldt_verify", "args": {"action": action}, "result": payload})
        return payload

    def hybrid_commit(action: str, proposal_ref: str, certificate_ref: str):
        """Commit only a proposed action with a matching exact certificate."""
        action = str(action)
        proposal = state["proposals"].get(str(proposal_ref))
        certificate = state["certificates"].get(str(certificate_ref))
        in_beam = bool(proposal and action in proposal["beam"])
        certified = bool(certificate and certificate["action"] == action and certificate["allowed"])
        accepted = in_beam and certified
        payload = {
            "accepted": accepted,
            "action": action if accepted else None,
            "reason": "typed_commit" if accepted else "proposal_or_certificate_mismatch",
        }
        state["commits"].append(payload)
        state["calls"].append(
            {
                "tool": "hybrid_commit",
                "args": {"action": action, "proposal_ref": proposal_ref, "certificate_ref": certificate_ref},
                "result": payload,
            }
        )
        return payload

    tools: dict[str, Any] = {
        "trm_propose": {"tool": trm_propose, "description": trm_propose.__doc__},
        "ldt_verify": {"tool": ldt_verify, "description": ldt_verify.__doc__},
        "hybrid_commit": {"tool": hybrid_commit, "description": hybrid_commit.__doc__},
    }
    if recursive:
        def delegate_soft_analysis(question: str):
            """Invoke the single no-authority child RLM for soft transcript analysis."""
            if state["delegates"]:
                raise RuntimeError("recursive conductor permits exactly one child delegate")
            response, child = _run_child_delegate(task, runtime, str(question))
            state["delegates"].append({"question": str(question), "response": response, **child})
            state["child_usage"] = _add_usage(state["child_usage"], child["usage"])
            state["child_time"] += float(child["execution_time"])
            state["calls"].append(
                {"tool": "delegate_soft_analysis", "args": {"question": str(question)}, "result": response}
            )
            return response

        tools["delegate_soft_analysis"] = {
            "tool": delegate_soft_analysis,
            "description": delegate_soft_analysis.__doc__,
        }
    return tools, state


def run_api_architecture(
    architecture_id: str,
    task: LongContextControlTask,
    trained_row: Mapping[str, Any],
    replicate_seed: int,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import rlm.core.rlm as core_module
    from rlm import RLM
    from rlm.logger import RLMLogger

    if architecture_id not in API_ARCHITECTURES:
        raise ValueError(f"not an API architecture: {architecture_id}")
    restricted = _restricted_local_environment()
    original_get_environment = core_module.get_environment

    def get_restricted_environment(environment, kwargs):
        if environment != "local":
            raise ValueError("registered RLM hybrid permits only the restricted local environment")
        return restricted(**kwargs)

    core_module.get_environment = get_restricted_environment
    tools: dict[str, Any] = {}
    tool_state: dict[str, Any] | None = None
    if architecture_id in {"rlm_tool_conductor", "rlm_recursive_conductor"}:
        tools, tool_state = _conductor_tools(
            task,
            trained_row,
            runtime,
            recursive=architecture_id == "rlm_recursive_conductor",
        )
        prompt = task.model_prompt() + "\nUse the registered tools to propose, verify, and commit one action."
        prologue = (
            "You are a typed module-flow conductor. You must call trm_propose, verify a beam action with "
            "ldt_verify, and call hybrid_commit with matching receipt references. Text alone cannot execute. "
        )
        if architecture_id == "rlm_recursive_conductor":
            prologue += "Call delegate_soft_analysis exactly once before committing. "
        else:
            prologue += "No recursive child is available. "
    elif architecture_id == "proxy_trm_rlm_critic_ldt":
        prompt = _critic_prompt(task, "proxy", trained_row)
        prologue = "Critique the registered proxy-TRM beam, then return only one candidate action token."
    elif architecture_id == "trained_trm_rlm_critic_ldt":
        prompt = _critic_prompt(task, "trained", trained_row)
        prologue = "Critique the registered trained-ControlTRM beam, then return only one candidate action token."
    else:
        prompt = task.model_prompt()
        prologue = "Use the restricted in-memory context and return only one candidate action token."
    engine = RLM(
        backend="openai",
        backend_kwargs={"model_name": runtime["model"], "max_retries": int(runtime["provider_max_retries"])},
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
            "network access are disabled. " + prologue
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
    root_usage = usage_totals(result.usage_summary.to_dict())
    usage = root_usage
    metrics = trajectory_metrics(result.metadata)
    response = result.response
    proposal = parse_action(response, task.candidates)
    manipulation = {"passed": True, "reason": "not_applicable"}
    if tool_state is not None:
        usage = _add_usage(root_usage, tool_state["child_usage"])
        accepted = [row for row in tool_state["commits"] if row["accepted"]]
        proposal = accepted[-1]["action"] if accepted else None
        names = [row["tool"] for row in tool_state["calls"]]
        required = {"trm_propose", "ldt_verify", "hybrid_commit"}
        recursion_ok = (
            len(tool_state["delegates"]) == 1
            if architecture_id == "rlm_recursive_conductor"
            else len(tool_state["delegates"]) == 0
        )
        manipulation = {
            "passed": required <= set(names) and recursion_ok and bool(accepted),
            "required_tools_present": required <= set(names),
            "delegate_count": len(tool_state["delegates"]),
            "accepted_commit_count": len(accepted),
        }
        executed, fallback, reason = typed_execute(task, proposal)
        if not manipulation["passed"]:
            executed, fallback, reason = ldt_action(task), True, "manipulation_failure_ldt_fallback"
    elif architecture_id == "rlm_repl_only":
        executed = proposal or task.candidates[0]
        fallback, reason = False, "rlm_text_action"
    else:
        executed, fallback, reason = typed_execute(task, proposal)
    record = _base_record(
        task, architecture_id, replicate_seed, proposal, executed, fallback, reason, usage, elapsed
    )
    record["manipulation"] = manipulation
    record["tool_calls"] = [] if tool_state is None else tool_state["calls"]
    record["recursive_child_count"] = 0 if tool_state is None else len(tool_state["delegates"])
    trajectory = {
        "architecture": architecture_id,
        "response": response,
        "official_completion": result.to_dict(),
        "root_usage": root_usage,
        "combined_usage": usage,
        "execution_time": elapsed,
        "trajectory_metrics": metrics,
        "tool_state": tool_state,
        "restricted_environment": {
            "filesystem": False,
            "subprocess": False,
            "allowed_imports": ["collections", "json", "math", "re", "statistics"],
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    return record, trajectory
