"""Dependency-free paired, lineage-clustered uncertainty summaries."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from random import Random


def cluster_paired_stats(pairs: list[tuple[str, float]], *, seed: int = 11,
                         bootstrap_draws: int = 1999) -> dict:
    if not pairs:
        raise ValueError("Matched paired observations required")
    groups = defaultdict(list)
    for group, delta in pairs:
        if not group:
            raise ValueError("Cluster label is mandatory")
        groups[group].append(float(delta))
    values = [sum(v) / len(v) for _, v in sorted(groups.items())]
    n = len(values)
    observed = sum(values) / n
    rng = Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in range(n)) / n
                   for _ in range(bootstrap_draws))
    lo = draws[int(0.025 * (bootstrap_draws - 1))]
    hi = draws[int(0.975 * (bootstrap_draws - 1))]
    if n <= 14:
        flips = (sum(sign * value for sign, value in zip(signs, values)) / n
                 for signs in product((-1, 1), repeat=n))
        extreme = sum(abs(x) >= abs(observed) - 1e-12 for x in flips)
        p = extreme / (1 << n)
        method = "exact_cluster_signflip"
    else:
        iterations = 4095
        extreme = sum(
            abs(sum(rng.choice((-1, 1)) * value for value in values) / n)
            >= abs(observed) - 1e-12
            for _ in range(iterations)
        )
        p = (extreme + 1) / (iterations + 1)
        method = "mc_cluster_signflip"
    return {
        "cluster_macro_delta": observed, "n_clusters": n,
        "cluster_bootstrap_95ci": [lo, hi],
        "cluster_signflip_p": p, "test_method": method,
        "bootstrap_draws": bootstrap_draws, "seed": seed,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(p < 0 or p > 1 for p in p_values):
        raise ValueError("p-values must be in [0,1]")
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)),
                                        key=lambda i: p_values[i])):
        previous = max(previous, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = previous
    return adjusted
