"""Deterministic tasks and metrics for the RLM architecture neighborhood."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ARCHITECTURE_FEATURES: dict[str, dict[str, int]] = {
    "direct_full_context": {
        "external_context": 0,
        "repl_control": 0,
        "recursive_depth": 0,
        "deterministic_partition": 0,
        "adaptive_subcalls": 0,
    },
    "map_reduce_4": {
        "external_context": 1,
        "repl_control": 0,
        "recursive_depth": 0,
        "deterministic_partition": 1,
        "adaptive_subcalls": 0,
    },
    "rlm_depth_1": {
        "external_context": 1,
        "repl_control": 1,
        "recursive_depth": 1,
        "deterministic_partition": 0,
        "adaptive_subcalls": 1,
    },
    "rlm_depth_2": {
        "external_context": 1,
        "repl_control": 1,
        "recursive_depth": 2,
        "deterministic_partition": 0,
        "adaptive_subcalls": 1,
    },
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def architecture_manifest() -> list[dict[str, Any]]:
    reference = ARCHITECTURE_FEATURES["rlm_depth_1"]
    manifests = []
    for architecture_id, features in ARCHITECTURE_FEATURES.items():
        distance = sum(abs(features[key] - reference[key]) for key in reference)
        manifests.append(
            {
                "architecture_id": architecture_id,
                "features": features,
                "distance_from_rlm_depth_1": distance,
                "architecture_hash": canonical_sha256(features),
            }
        )
    return manifests


def _random_payload(rng: random.Random, words: int = 8) -> str:
    alphabet = "abcdefghijkmnopqrstuvwxyz"
    return " ".join(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 9)))
        for _ in range(words)
    )


def _task_prompt(query: str, lines: list[str]) -> str:
    return (
        "Return only the requested answer token, with no explanation.\n"
        f"QUERY: {query}\n"
        "CONTEXT_START\n"
        + "\n".join(lines)
        + "\nCONTEXT_END\n"
    )


def materialize_tasks(seed: int = 73021, filler_count: int = 640) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    needle_answer = "KX-731-OMEGA"
    needle_lines = [
        f"REC|{index:04d}|payload={_random_payload(rng)}" for index in range(filler_count)
    ]
    needle_lines.insert(417, f"SECRET|needle_key={needle_answer}|scope=registered")
    tasks.append(
        {
            "task_id": "needle_exact",
            "family": "retrieval",
            "expected_answer": needle_answer,
            "prompt": _task_prompt("What is the value of needle_key?", needle_lines),
        }
    )

    join_answer = "ZX-42-Q"
    join_lines = [
        f"LOG|{index:04d}|payload={_random_payload(rng)}" for index in range(filler_count)
    ]
    join_lines.extend(
        [
            "PERSON|alias=ALIAS-17|order=ORD-8841",
            "ORDER|id=ORD-8841|warehouse=WH-290",
            f"WAREHOUSE|id=WH-290|release_code={join_answer}",
        ]
    )
    rng.shuffle(join_lines)
    tasks.append(
        {
            "task_id": "three_hop_join",
            "family": "relational_join",
            "expected_answer": join_answer,
            "prompt": _task_prompt(
                "Follow ALIAS-17 to its order and warehouse. What is the release_code?",
                join_lines,
            ),
        }
    )

    categories = ("amber", "cobalt", "jade", "sienna")
    aggregate_lines: list[str] = []
    aggregate_answer = 0
    for index in range(filler_count):
        category = categories[rng.randrange(len(categories))]
        amount = rng.randint(1, 97)
        aggregate_lines.append(f"TXN|{index:04d}|category={category}|amount={amount}")
        if category == "amber":
            aggregate_answer += amount
    rng.shuffle(aggregate_lines)
    tasks.append(
        {
            "task_id": "distributed_sum",
            "family": "global_aggregation",
            "expected_answer": str(aggregate_answer),
            "prompt": _task_prompt(
                "What is the exact sum of amount over every category=amber transaction?",
                aggregate_lines,
            ),
        }
    )

    latest_answer = "PAYLOAD-V19-FINAL"
    latest_lines = [
        f"AUDIT|{index:04d}|payload={_random_payload(rng)}" for index in range(filler_count)
    ]
    for version in range(1, 20):
        payload = latest_answer if version == 19 else f"PAYLOAD-V{version:02d}-OLD"
        latest_lines.append(f"REV|artifact=TARGET-X|version={version}|value={payload}")
    rng.shuffle(latest_lines)
    tasks.append(
        {
            "task_id": "latest_revision",
            "family": "version_resolution",
            "expected_answer": latest_answer,
            "prompt": _task_prompt(
                "For artifact TARGET-X, return the value at the highest numeric version.",
                latest_lines,
            ),
        }
    )

    for task in tasks:
        task["prompt_sha256"] = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
        task["prompt_characters"] = len(task["prompt"])
        task["prompt_lines"] = task["prompt"].count("\n") + 1
    return {
        "suite_id": "rlm_architecture_neighborhood_tasks_v0",
        "generator_seed": seed,
        "filler_count": filler_count,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def answer_matches(response: str, expected: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9-]+", response.upper())
    return expected.upper() in tokens


def usage_totals(usage: Mapping[str, Any]) -> dict[str, float | int | None]:
    models = usage.get("model_usage_summaries", {})
    calls = sum(int(row.get("total_calls", 0)) for row in models.values())
    input_tokens = sum(int(row.get("total_input_tokens", 0)) for row in models.values())
    output_tokens = sum(int(row.get("total_output_tokens", 0)) for row in models.values())
    costs = [row.get("total_cost") for row in models.values() if row.get("total_cost") is not None]
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reported_cost_usd": sum(float(value) for value in costs) if costs else None,
    }


def trajectory_metrics(metadata: Mapping[str, Any] | None) -> dict[str, int]:
    metrics = {
        "iterations": 0,
        "code_blocks": 0,
        "subcalls": 0,
        "recursive_subcalls": 0,
        "maximum_observed_depth": 0,
        "repl_errors": 0,
    }

    def visit(current: Mapping[str, Any] | None, depth: int) -> None:
        if not current:
            return
        metrics["maximum_observed_depth"] = max(metrics["maximum_observed_depth"], depth)
        iterations = current.get("iterations", [])
        metrics["iterations"] += len(iterations)
        for iteration in iterations:
            for block in iteration.get("code_blocks", []):
                metrics["code_blocks"] += 1
                result = block.get("result", {})
                if str(result.get("stderr", "")).strip():
                    metrics["repl_errors"] += 1
                for call in result.get("rlm_calls", []):
                    metrics["subcalls"] += 1
                    child = call.get("metadata")
                    if child:
                        metrics["recursive_subcalls"] += 1
                        visit(child, depth + 1)

    visit(metadata, 0)
    return metrics


def split_prompt(prompt: str, chunks: int) -> list[str]:
    lines = prompt.splitlines()
    query_end = next(index for index, line in enumerate(lines) if line == "CONTEXT_START")
    header = lines[: query_end + 1]
    context = lines[query_end + 1 : -1]
    width = math.ceil(len(context) / chunks)
    return ["\n".join([*header, *context[index : index + width]]) for index in range(0, len(context), width)]


def _restricted_local_environment():
    from rlm.environments.local_repl import LocalREPL

    original_import = __import__
    allowed_modules = {"collections", "json", "math", "re", "statistics"}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in allowed_modules:
            raise ImportError(f"module {root!r} is not permitted in the registered RLM REPL")
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


def _run_direct(prompt: str, runtime: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    from rlm.clients.openai import OpenAIClient

    client = OpenAIClient(
        model_name=runtime["model"],
        max_retries=int(runtime["provider_max_retries"]),
        sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
    )
    started = time.perf_counter()
    response = client.completion(prompt)
    elapsed = time.perf_counter() - started
    return response, {
        "architecture": "direct_full_context",
        "usage_summary": client.get_usage_summary().to_dict(),
        "execution_time": elapsed,
        "trajectory_metrics": {
            "iterations": 0,
            "code_blocks": 0,
            "subcalls": 0,
            "recursive_subcalls": 0,
            "maximum_observed_depth": 0,
            "repl_errors": 0,
        },
    }


def _run_map_reduce(prompt: str, runtime: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    from rlm.clients.openai import OpenAIClient

    client = OpenAIClient(
        model_name=runtime["model"],
        max_retries=int(runtime["provider_max_retries"]),
        sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
    )
    started = time.perf_counter()
    evidence = []
    prompt_chunks = split_prompt(prompt, int(runtime["map_chunks"]))
    for index, chunk in enumerate(prompt_chunks):
        evidence.append(
            client.completion(
                "Analyze this registered context partition. Return only compact evidence or a partial "
                "aggregate needed to answer the query; say NONE if irrelevant.\n"
                f"PARTITION={index}\n{chunk}"
            )
        )
    query = prompt.split("CONTEXT_START", 1)[0]
    response = client.completion(
        f"{query}\nCombine the partition evidence below and return only the requested answer token.\n"
        + "\n".join(f"PARTITION_{index}: {value}" for index, value in enumerate(evidence))
    )
    elapsed = time.perf_counter() - started
    return response, {
        "architecture": "map_reduce_4",
        "partition_count": len(prompt_chunks),
        "partition_evidence": evidence,
        "usage_summary": client.get_usage_summary().to_dict(),
        "execution_time": elapsed,
        "trajectory_metrics": {
            "iterations": 0,
            "code_blocks": 0,
            "subcalls": len(evidence),
            "recursive_subcalls": 0,
            "maximum_observed_depth": 0,
            "repl_errors": 0,
        },
    }


def _run_rlm(
    prompt: str, runtime: Mapping[str, Any], *, depth: int
) -> tuple[str, dict[str, Any]]:
    import rlm.core.rlm as core_module
    from rlm import RLM
    from rlm.logger import RLMLogger

    restricted = _restricted_local_environment()
    original_get_environment = core_module.get_environment

    def get_restricted_environment(environment, kwargs):
        if environment != "local":
            raise ValueError("registered RLM neighborhood permits only the restricted local environment")
        return restricted(**kwargs)

    core_module.get_environment = get_restricted_environment
    logger = RLMLogger()
    engine = None
    try:
        engine = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": runtime["model"],
                "max_retries": int(runtime["provider_max_retries"]),
            },
            environment="local",
            max_depth=depth,
            max_iterations=int(runtime[f"depth_{depth}_max_iterations"]),
            max_timeout=float(runtime["cell_timeout_seconds"]),
            max_tokens=int(runtime["total_token_soft_cap"]),
            max_errors=int(runtime["max_consecutive_errors"]),
            max_concurrent_subcalls=int(runtime["max_concurrent_subcalls"]),
            sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
            sub_sampling_args={"max_tokens": int(runtime["max_output_tokens_per_call"])},
            user_prologue=(
                "Registered restricted environment: context is an in-memory string. Filesystem access and "
                "OS/subprocess imports are disabled. You may use string operations or import only re, json, "
                "math, statistics, and collections. Use no more than two subcalls per iteration. Set "
                "answer['content'] to only the requested token and answer['ready'] = True when done."
            ),
            logger=logger,
            verbose=False,
        )
        result = engine.completion(prompt)
    finally:
        if engine is not None:
            engine.close()
        core_module.get_environment = original_get_environment
    return result.response, {
        "architecture": f"rlm_depth_{depth}",
        "official_completion": result.to_dict(),
        "usage_summary": result.usage_summary.to_dict(),
        "execution_time": result.execution_time,
        "trajectory_metrics": trajectory_metrics(result.metadata),
        "restricted_environment": {
            "filesystem": False,
            "subprocess": False,
            "allowed_imports": ["collections", "json", "math", "re", "statistics"],
        },
    }


def run_architecture(
    architecture_id: str, prompt: str, runtime: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if architecture_id == "direct_full_context":
        return _run_direct(prompt, runtime)
    if architecture_id == "map_reduce_4":
        return _run_map_reduce(prompt, runtime)
    if architecture_id == "rlm_depth_1":
        return _run_rlm(prompt, runtime, depth=1)
    if architecture_id == "rlm_depth_2":
        return _run_rlm(prompt, runtime, depth=2)
    raise ValueError(f"unknown architecture: {architecture_id}")


def pareto_frontier(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    aggregates = list(rows)
    frontier: list[str] = []
    for candidate in aggregates:
        dominated = False
        for other in aggregates:
            if other is candidate:
                continue
            no_worse = (
                float(other["accuracy"]) >= float(candidate["accuracy"])
                and int(other["total_tokens"]) <= int(candidate["total_tokens"])
                and float(other["execution_time"]) <= float(candidate["execution_time"])
            )
            strictly_better = (
                float(other["accuracy"]) > float(candidate["accuracy"])
                or int(other["total_tokens"]) < int(candidate["total_tokens"])
                or float(other["execution_time"]) < float(candidate["execution_time"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["architecture_id"]))
    return sorted(frontier)
