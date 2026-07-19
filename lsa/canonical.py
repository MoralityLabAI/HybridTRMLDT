"""Canonical algebra, model, and run identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .instances import ArchitectureInstance


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def canonical_algebra(instance: ArchitectureInstance) -> dict[str, Any]:
    visits = instance.word.expand()
    module_aliases: dict[str, str] = {}
    for visit in visits:
        module_aliases.setdefault(
            visit.symbol.module, f"m{len(module_aliases)}"
        )
    modules = {}
    for original, alias in sorted(module_aliases.items(), key=lambda item: item[1]):
        module = instance.modules[original]
        modules[alias] = {
            "residual_sublayers": module.residual_sublayers,
            "parameter_count": module.parameter_count,
            "flops_per_token": module.flops_per_token,
            "residual": _jsonable(module.residual),
        }
    policy = instance.effective_gradient_policy
    return {
        "states": sorted(instance.states),
        "modules": modules,
        "visits": [
            {
                "module": module_aliases[visit.symbol.module],
                "reads": sorted(visit.symbol.reads),
                "writes": visit.symbol.writes,
            }
            for visit in visits
        ],
        "parameter_visits": sorted(policy.parameter_positions(visits)),
        "retained_state_edges": sorted(policy.state_edge_positions(visits)),
        "detach_carry_between_segments": policy.detach_carry_between_segments,
        "supervision": sorted(point.visit_position for point in instance.supervision),
        "carry": {
            "persistent": sorted(instance.carry.persistent_registers),
            "detached": sorted(instance.carry.detached_registers),
        },
    }


def algebra_hash(instance: ArchitectureInstance) -> str:
    return digest(canonical_algebra(instance))


def model_hash(instance: ArchitectureInstance, model_context: Mapping[str, Any]) -> str:
    return digest(
        {"algebra_hash": algebra_hash(instance), "model": _jsonable(model_context)}
    )


def run_hash(
    instance: ArchitectureInstance,
    model_context: Mapping[str, Any],
    run_context: Mapping[str, Any],
) -> str:
    return digest(
        {
            "model_hash": model_hash(instance, model_context),
            "run": _jsonable(run_context),
        }
    )
