"""CCA-10: count-based optimistic utility bounds for the frozen storyworld.

Proof assumptions: befriend raises trust <=1, investigate raises evidence
<=1, defuse lowers heat <=2, befriend/wait lower heat <=1, rush advances
scene <=2, other non-wait actions advance scene <=1, and each rival action
raises trust <=1, never increases evidence, never lowers heat, nor advances
scene. Enumerating all self-action counts yields a sound upper bound for
every action sequence and any rival sequence in this stated alphabet.
"""
from __future__ import annotations
from functools import lru_cache

@lru_cache(maxsize=None)
def action_count_vectors(horizon: int) -> tuple[tuple[int,int,int,int,int], ...]:
    if type(horizon) is not int or horizon < 0:
        raise ValueError("nonnegative integer horizon required")
    if horizon > 20:
        raise ValueError("reference algorithm limited to horizon <= 20")
    result = []
    for b in range(horizon + 1):
        for i in range(horizon-b+1):
            for d in range(horizon-b-i+1):
                for r in range(horizon-b-i-d+1):
                    for w in range(horizon-b-i-d-r+1):
                        result.append((b,i,d,r,w))
    return tuple(result)

@lru_cache(maxsize=None)
def optimistic_moral_upper(
    trust: int, evidence: int, heat: int, scene: int, horizon: int
) -> float:
    """Upper envelope over reachable moral terminal values within horizon."""
    if (type(horizon) is not int or horizon < 0 or
            not -3 <= trust <= 3 or not 0 <= evidence <= 4 or
            not 0 <= heat <= 4 or not 0 <= scene <= 5):
        raise ValueError("state or horizon outside registered domain")
    best = -10**9
    for b,i,d,r,w in action_count_vectors(horizon):
        steps = b+i+d+r+w
        t_up = min(3, trust+b+steps)
        e_up = min(4, evidence+i)
        h_low = max(0, heat-b-2*d-w)
        s_up = min(5, scene+b+i+d+2*r)
        best = max(best, 3*t_up+2*e_up-3*h_low+s_up)
    return min(1., max(0., (best+21.)/43.))
