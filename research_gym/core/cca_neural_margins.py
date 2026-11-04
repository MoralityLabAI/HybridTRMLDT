"""CCA-12: exact last-layer logit robustness and conservative hidden-state bounds.

The action permission set must be independently supplied and held fixed.
These are certificates for action choice under specified additive noise,
not certificates for the environment, language interpretation or upstream GRU.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf, isfinite, sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LogitCertificate:
    selected: str
    selected_certified: bool
    viable: bool
    all_certified: bool
    gap: float | None
    radius: float | None


def winner(logits: Sequence[float], actions: Sequence[str]) -> str:
    if not actions or len(actions) != len(logits) or len(set(actions)) != len(actions):
        raise ValueError("Expected a distinct nonempty action vocabulary")
    if any(not isfinite(float(v)) for v in logits):
        raise ValueError("All logits must be finite")
    return actions[max(range(len(actions)), key=lambda i: (float(logits[i]), -i))]


def certify_logit_infinity(
    logits: Sequence[float], actions: Sequence[str], certified: Iterable[str]
) -> LogitCertificate:
    chosen = winner(logits, actions)
    allowed = frozenset(certified)
    if not allowed <= set(actions):
        raise ValueError("Certificate refers to an action outside the vocabulary")
    if not allowed:
        return LogitCertificate(chosen, False, False, False, None, None)
    if len(allowed) == len(actions):
        return LogitCertificate(chosen, True, True, True, inf, inf)
    best_safe = max(float(z) for a, z in zip(actions, logits) if a in allowed)
    best_unsafe = max(float(z) for a, z in zip(actions, logits) if a not in allowed)
    gap = best_safe - best_unsafe
    return LogitCertificate(chosen, chosen in allowed, True, False, gap, max(0., gap / 2.))


def guaranteed_safe_logit_noise(cert: LogitCertificate, epsilon: float) -> bool:
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite nonnegative bound")
    return cert.viable and (cert.all_certified or cert.gap > 2 * epsilon)


def adversarial_logit_witness(
    logits: Sequence[float], actions: Sequence[str],
    certified: Iterable[str], epsilon: float
) -> tuple[float, ...]:
    """Independent per-action infinity-norm extreme: safe down, unsafe up."""
    winner(logits, actions)
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    allowed = frozenset(certified)
    if not allowed <= set(actions):
        raise ValueError("Unknown certified action")
    return tuple(float(v) + (-epsilon if a in allowed else epsilon)
                 for a, v in zip(actions, logits))


def certified_ranked_action(
    logits: Sequence[float], actions: Sequence[str], certified: Iterable[str]
) -> str | None:
    """Fail closed on empty certified set; rank ONLY host-certified actions."""
    winner(logits, actions)
    allowed = frozenset(certified)
    if not allowed <= set(actions):
        raise ValueError("Unknown certified action")
    indices = [i for i, a in enumerate(actions) if a in allowed]
    if not indices:
        return None
    return actions[max(indices, key=lambda i: (float(logits[i]), -i))]


def anchored_hidden_l2_radius(
    logits: Sequence[float], head_rows: Sequence[Sequence[float]],
    actions: Sequence[str], certified: Iterable[str]
) -> float | None:
    """Sufficient final-hidden L2 safety radius.

    z'=z+W*delta_h and the original UNMASKED winner a is certified.
    If ||delta_h||_2 < r, a dominates every unsafe u, hence whichever
    action wins is certified.  r is an anchored sufficient bound, not the
    exact distance to the unsafe decision polyhedron.
    """
    a = winner(logits, actions)
    allowed = frozenset(certified)
    if not allowed <= set(actions):
        raise ValueError("Unknown certified action")
    if len(head_rows) != len(actions) or not head_rows or len(head_rows[0]) == 0:
        raise ValueError("Invalid head dimensions")
    dim = len(head_rows[0])
    if any(len(row) != dim or any(not isfinite(float(v)) for v in row)
           for row in head_rows):
        raise ValueError("Nonfinite or inconsistent head weights")
    if not allowed:
        return None
    if a not in allowed:
        return 0.
    if len(allowed) == len(actions):
        return inf
    anchor = actions.index(a)
    radii = []
    for u, candidate in enumerate(actions):
        if candidate in allowed:
            continue
        gap = float(logits[anchor]) - float(logits[u])
        if gap <= 0:
            return 0.
        norm = sqrt(sum((float(p)-float(q))**2
                        for p, q in zip(head_rows[anchor], head_rows[u])))
        if norm:
            radii.append(gap / norm)
    return min(radii) if radii else inf
