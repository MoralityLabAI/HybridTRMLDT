"""Mesh search space: router families, hyperparameter ranges, validation and sampling.

A candidate mesh is one JSON object::

    {"family": "trm",
     "core":    {"d": 32, "heads": 2, "net_layers": 1, "inner": 1, "cycles": 2},
     "encoder": "match",
     "train":   {"worlds": 24, "epochs": 2, "lr": 0.0015, "batch_states": 32, "seed": 7},
     "filters": {"prefilter_k": 32, "max_tool_calls": 40}}

Missing sections take the defaults below. Unknown keys and out-of-range values are rejected with
the offending path, so a model under test learns the ranges from its errors as well as from
``list_mesh_families``. The MCP surface (arm and index layout) belongs to the episode, not to the mesh.
"""
from __future__ import annotations

import json
import random
from typing import Any

FAMILIES = ("dense", "transformer", "looped", "trm", "trm_ldt", "hrm")
ENCODERS = ("tokens", "match")

# name -> spec; "choices" are enumerated, "int" and "float" carry bounds, "log" marks log-uniform floats.
_D = {"type": "choices", "choices": [16, 24, 32, 48], "default": 24}
_HEADS = {"type": "choices", "choices": [1, 2, 4], "default": 2}
CORE = {
    "dense": {"d": _D},
    "transformer": {"d": _D, "heads": _HEADS, "layers": {"type": "int", "min": 1, "max": 4, "default": 2}},
    "looped": {"d": _D, "heads": _HEADS, "passes": {"type": "int", "min": 1, "max": 8, "default": 3}},
    "trm": {"d": _D, "heads": _HEADS, "net_layers": {"type": "int", "min": 1, "max": 2, "default": 1},
            "inner": {"type": "int", "min": 1, "max": 3, "default": 1},
            "cycles": {"type": "int", "min": 1, "max": 4, "default": 2}},
    "hrm": {"d": _D, "heads": _HEADS, "low_steps": {"type": "int", "min": 1, "max": 3, "default": 1},
            "high_cycles": {"type": "int", "min": 1, "max": 4, "default": 2}},
}
CORE["trm_ldt"] = CORE["trm"]

TRAIN = {
    "epochs": {"type": "int", "min": 1, "max": 6, "default": 2},
    "lr": {"type": "float", "min": 3e-4, "max": 4e-3, "log": True, "default": 1.5e-3},
    "batch_states": {"type": "choices", "choices": [8, 16, 32, 64], "default": 32},
    "seed": {"type": "int", "min": 0, "max": 9999, "default": 7},
}
FILTERS = {
    "prefilter_k": {"type": "int", "min": 8, "max": 64, "default": 32},
}


def ranges(max_train_worlds: int, max_tool_calls: int) -> dict:
    """The full range table for one episode (world pool and call cap come from the episode)."""
    train = dict(TRAIN, worlds={"type": "int", "min": 4, "max": max_train_worlds, "default": max_train_worlds})
    filters = dict(FILTERS, max_tool_calls={"type": "int", "min": 8, "max": max_tool_calls,
                                            "default": max_tool_calls})
    return {"family": {"type": "choices", "choices": list(FAMILIES)},
            "core": CORE, "encoder": {"type": "choices", "choices": list(ENCODERS), "default": "tokens"},
            "train": train, "filters": filters}


def _check(path: str, spec: dict, value: Any) -> Any:
    kind = spec["type"]
    if kind == "choices":
        if value not in spec["choices"] or isinstance(value, bool):
            raise ValueError(f"{path} must be one of {spec['choices']}")
        return value
    if kind == "int":
        if type(value) is not int or not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"{path} must be an integer in [{spec['min']}, {spec['max']}]")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{path} must be a number in [{spec['min']}, {spec['max']}]")
    return float(value)


def _section(path: str, table: dict, given: Any) -> dict:
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise ValueError(f"{path} must be an object")
    extra = sorted(set(given) - set(table))
    if extra:
        raise ValueError(f"{path} has unknown keys {extra}; allowed {sorted(table)}")
    return {name: _check(f"{path}.{name}", spec, given.get(name, spec["default"])) for name, spec in table.items()}


def normalize(candidate: Any, max_train_worlds: int, max_tool_calls: int) -> dict:
    """Return the candidate with every default filled in, or raise ``ValueError``."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    extra = sorted(set(candidate) - {"family", "core", "encoder", "train", "filters"})
    if extra:
        raise ValueError(f"candidate has unknown keys {extra}")
    table = ranges(max_train_worlds, max_tool_calls)
    family = candidate.get("family")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {list(FAMILIES)}")
    out = {"family": family, "core": _section("core", CORE[family], candidate.get("core")),
           "encoder": _check("encoder", table["encoder"], candidate.get("encoder", "tokens")),
           "train": _section("train", table["train"], candidate.get("train")),
           "filters": _section("filters", table["filters"], candidate.get("filters"))}
    core = out["core"]
    if "heads" in core and core["d"] % core["heads"]:
        raise ValueError("core.d must be divisible by core.heads")
    return out


def router_config(candidate: dict) -> dict:
    """The ``Router`` config dict for a normalized candidate."""
    config = {"arch": candidate["family"], **candidate["core"]}
    if candidate["encoder"] != "tokens":
        config["encoder"] = candidate["encoder"]
    return config


def default_candidate(max_train_worlds: int, max_tool_calls: int) -> dict:
    """The fixed reference mesh: a width-24 TRM with two cycles, exact-match features, default training."""
    return normalize({"family": "trm", "encoder": "match"}, max_train_worlds, max_tool_calls)


def sample(rng: random.Random, max_train_worlds: int, max_tool_calls: int) -> dict:
    """Uniform draw over families and ranges (log-uniform learning rate, quartile train-world counts)."""
    import math
    family = rng.choice(FAMILIES)
    table = ranges(max_train_worlds, max_tool_calls)

    def draw(spec: dict) -> Any:
        if spec["type"] == "choices":
            return rng.choice(spec["choices"])
        if spec["type"] == "int":
            return rng.randint(spec["min"], spec["max"])
        if spec.get("log"):
            return math.exp(rng.uniform(math.log(spec["min"]), math.log(spec["max"])))
        return rng.uniform(spec["min"], spec["max"])

    core = {name: draw(spec) for name, spec in CORE[family].items()}
    if "heads" in core and core["d"] % core["heads"]:
        core["heads"] = 1
    train = {name: draw(spec) for name, spec in table["train"].items()}
    # Quartile world counts keep the number of distinct train datasets (and their build time) at four.
    train["worlds"] = max(4, round(max_train_worlds * rng.choice((0.25, 0.5, 0.75, 1.0))))
    return normalize({"family": family, "core": core, "encoder": rng.choice(ENCODERS),
                      "train": train,
                      "filters": {name: draw(spec) for name, spec in table["filters"].items()}},
                     max_train_worlds, max_tool_calls)


def candidate_key(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))
