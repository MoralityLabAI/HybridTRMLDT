"""Pure invariant calculators over loop-schedule architecture instances."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Mapping

from .instances import ArchitectureInstance
from .schedule import ExpandedVisit, selected_visits


def visits(instance: ArchitectureInstance) -> tuple[ExpandedVisit, ...]:
    return instance.word.expand()


def gradient_visible_visits(instance: ArchitectureInstance) -> tuple[ExpandedVisit, ...]:
    visits_ = instance.word.expand()
    selected = instance.effective_gradient_policy.parameter_positions(visits_)
    return tuple(visit for visit in visits_ if visit.position in selected)


def retained_state_edges(instance: ArchitectureInstance) -> frozenset[int]:
    visits_ = instance.word.expand()
    return instance.effective_gradient_policy.state_edge_positions(visits_)


def applied_parameter_count(instance: ArchitectureInstance) -> int | None:
    total = 0
    for visit in visits(instance):
        count = instance.modules[visit.symbol.module].parameter_count
        if count is None:
            return None
        total += count
    return total


def gradient_applied_parameter_count(instance: ArchitectureInstance) -> int | None:
    total = 0
    for visit in gradient_visible_visits(instance):
        count = instance.modules[visit.symbol.module].parameter_count
        if count is None:
            return None
        total += count
    return total


def gradient_visible_labels(instance: ArchitectureInstance) -> tuple[str, ...]:
    return tuple(visit.label for visit in gradient_visible_visits(instance))


def residual_visit_count(instance: ArchitectureInstance) -> int:
    return sum(instance.modules[v.symbol.module].residual_sublayers for v in visits(instance))


def gradient_residual_visit_count(instance: ArchitectureInstance) -> int:
    return sum(
        instance.modules[v.symbol.module].residual_sublayers
        for v in gradient_visible_visits(instance)
    )


def rounds(instance: ArchitectureInstance) -> dict[str, int]:
    return dict(Counter(v.symbol.module for v in visits(instance)))


def gradient_rounds(instance: ArchitectureInstance) -> dict[str, int]:
    counts = Counter(v.symbol.module for v in gradient_visible_visits(instance))
    return {module: counts.get(module, 0) for module in instance.modules}


def parameter_count(instance: ArchitectureInstance) -> int | None:
    counts = [module.parameter_count for module in instance.modules.values()]
    return None if any(count is None for count in counts) else sum(counts)  # type: ignore[arg-type]


def flops_per_token(instance: ArchitectureInstance) -> int | None:
    total = 0
    for visit in visits(instance):
        cost = instance.modules[visit.symbol.module].flops_per_token
        if cost is None:
            return None
        total += cost
    return total


def stability_functional(
    instance: ArchitectureInstance,
    kappa_by_module: Mapping[str, float],
) -> float:
    visible_rounds = gradient_rounds(instance)
    total = 0.0
    for name, module in instance.modules.items():
        if name not in kappa_by_module:
            raise KeyError(f"missing kappa for module {name}")
        kappa = float(kappa_by_module[name])
        if kappa < 0 or not isfinite(kappa):
            raise ValueError("kappa values must be finite and non-negative")
        ratio_sq = (module.residual.beta / module.residual.alpha) ** 2
        total += module.residual_sublayers * visible_rounds[name] * kappa * ratio_sq
    return total


def predicted_minimal_exponent(gamma_by_module: Mapping[str, float]) -> float:
    if not gamma_by_module:
        raise ValueError("at least one gamma estimate is required")
    gamma_binding = max(float(value) for value in gamma_by_module.values())
    if not isfinite(gamma_binding):
        raise ValueError("gamma values must be finite")
    return (1.0 + gamma_binding) / 4.0
