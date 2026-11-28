"""Restricted declarative architecture language.

Inky Mini proposes architectures as JSON objects.  The host validates every field
against this schema (closed key set, enumerated choices, integer/float bounds),
fills defaults, and compiles the result into a module.  Nothing in a proposal is
ever executed; unknown keys are rejected, not ignored.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

PARAM_MIN = 5_000
PARAM_MAX = 250_000

# field -> (kind, spec, default)
SCHEMA: Dict[str, Tuple[str, Any, Any]] = {
    "core": ("enum", ["trm", "gru", "ff"], "trm"),
    "hidden": ("int", (16, 160), 32),
    "latent_steps": ("int", (1, 6), 2),          # n: latent updates per answer update
    "cycles": ("int", (1, 4), 1),                # T: answer updates per visit
    "grad_cycles": ("int", (1, 4), 1),           # last cycles of each visit that carry gradient
    "visits": ("int", (1, 6), 3),                # R: supervised visits (deep supervision steps)
    "state_partition": ("enum", ["shared", "split"], "shared"),
    "update_schedule": ("enum", ["latent_then_answer", "interleaved", "answer_first"], "latent_then_answer"),
    "residual": ("enum", ["none", "add", "gated"], "add"),
    "feedback": ("enum", ["direct", "matched_damping", "rlt"], "direct"),
    "aggregation": ("enum", ["pointwise", "set_mean", "relation", "attention", "relation+mean"], "relation+mean"),
    "mixer_layers": ("int", (1, 3), 1),
    "norm": ("enum", ["none", "layer"], "layer"),
    "supervision": ("enum", ["final", "deep", "deep_ramp"], "deep"),
    "ldt_projection": ("bool", None, False),
    "verifier_visits": ("int", (0, 2), 1),       # in-loop verifier calls (host feeds diagnostics back)
    "latent_reset": ("bool", None, False),       # ablation: reset z at every visit
    "lr": ("float", (2e-4, 6e-3), 2e-3),
    "weight_decay": ("float", (0.0, 0.2), 0.01),
}
CHOICE_FIELDS = sorted(SCHEMA)


class ProposalError(ValueError):
    pass


def validate(proposal: Dict[str, Any], base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a complete, normalised configuration or raise ``ProposalError``."""
    if not isinstance(proposal, dict):
        raise ProposalError("proposal must be a JSON object")
    unknown = sorted(set(proposal) - set(SCHEMA))
    if unknown:
        raise ProposalError(f"unknown fields: {unknown}")
    cfg = dict(defaults() if base is None else base)
    for key, value in proposal.items():
        kind, spec, _ = SCHEMA[key]
        if kind == "enum":
            if value not in spec:
                raise ProposalError(f"{key}={value!r} not in {spec}")
        elif kind == "bool":
            if not isinstance(value, bool):
                raise ProposalError(f"{key} must be boolean")
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                else:
                    raise ProposalError(f"{key} must be an integer")
            if not spec[0] <= value <= spec[1]:
                raise ProposalError(f"{key}={value} outside [{spec[0]}, {spec[1]}]")
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProposalError(f"{key} must be a number")
            value = float(value)
            if not spec[0] <= value <= spec[1]:
                raise ProposalError(f"{key}={value} outside [{spec[0]}, {spec[1]}]")
        cfg[key] = value
    if cfg["grad_cycles"] > cfg["cycles"]:
        cfg["grad_cycles"] = cfg["cycles"]
    if cfg["verifier_visits"] > max(cfg["visits"] - 1, 0):
        cfg["verifier_visits"] = max(cfg["visits"] - 1, 0)
    return cfg


def defaults() -> Dict[str, Any]:
    return {k: v[2] for k, v in SCHEMA.items()}


def config_id(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {k: b[k] for k in SCHEMA if a.get(k) != b.get(k)}


def schema_doc() -> str:
    lines = []
    for key in CHOICE_FIELDS:
        kind, spec, default = SCHEMA[key]
        if kind == "enum":
            rng = "|".join(spec)
        elif kind == "bool":
            rng = "true|false"
        else:
            rng = f"{spec[0]}..{spec[1]}"
        lines.append(f"- {key}: {kind} {rng} (default {json.dumps(default)})")
    return "\n".join(lines)


def implementation_label(cfg: Dict[str, Any]) -> str:
    """Every report distinguishes native TRM recursion from simplified proxies."""
    return {"trm": "native_trm_update", "gru": "recurrent_proxy_gru", "ff": "feedforward"}[cfg["core"]]


def mutate(cfg: Dict[str, Any], rng, n_fields: int = 2) -> Dict[str, Any]:
    """Random-search control: change ``n_fields`` fields to random legal values."""
    out = dict(cfg)
    keys = [k for k in CHOICE_FIELDS if k not in ("latent_reset",)]
    for key in rng.choice(keys, size=n_fields, replace=False):
        kind, spec, _ = SCHEMA[key]
        if kind == "enum":
            out[key] = str(rng.choice(spec))
        elif kind == "bool":
            out[key] = bool(rng.random() < 0.5)
        elif kind == "int":
            out[key] = int(rng.integers(spec[0], spec[1] + 1))
        else:
            out[key] = float(round(float(rng.uniform(spec[0], spec[1])), 5))
    return validate({k: out[k] for k in SCHEMA})


def limit_edit(incumbent: Dict[str, Any], proposed: Dict[str, Any], budget: int) -> Tuple[Dict[str, Any], List[str]]:
    """Meta-level RLT: keep at most ``budget`` changed fields (in the proposal's own
    key order), reverting the rest to the incumbent.  Returns config and dropped keys."""
    changed = list(diff(incumbent, proposed))
    keep, drop = changed[:budget], changed[budget:]
    out = dict(incumbent)
    for k in keep:
        out[k] = proposed[k]
    return validate({k: out[k] for k in SCHEMA}), drop
