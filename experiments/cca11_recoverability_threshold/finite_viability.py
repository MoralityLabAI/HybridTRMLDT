"""CCA-11B: host-model viability geometry on the registered CCA-05 storyworld.

Finite reenactment of CoupledStoryworldEnv's actions, rival policy and
secret-ending predicate, pinned at blob 545b2a1c283c1da8ac6c9fdb375a1bafd695e9b2.
No neural model, activation metric or SAE claim is made.
"""
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
import json

ACTIONS = ("befriend", "investigate", "defuse", "rush", "wait")
RIVALS = ("rival_sabotage", "rival_flatter", "rival_wait")


@dataclass(frozen=True)
class State:
    trust: int
    evidence: int
    heat: int
    scene: int


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def target(s):
    return s.scene >= 5 and s.trust >= 1 and s.evidence >= 3 and s.heat <= 2


def terminal(s):
    return s.scene >= 5


def other_policy(s):
    if s.evidence >= 2 and s.heat <= 3:
        return "rival_sabotage"
    if s.trust <= -1 and s.evidence >= 1:
        return "rival_flatter"
    return "rival_wait"


def step(s, a, b):
    t, e, h, scene = s.trust, s.evidence, s.heat, s.scene
    if a == "befriend":
        t += 1
        h = max(0, h-1)
    elif a == "investigate":
        e += 1
        h += 1
        if h >= 2:
            t -= 1
    elif a == "defuse":
        h -= 2
    elif a == "rush":
        scene += 1
        h += 1
    elif a == "wait":
        h = max(0, h-1)
    else:
        raise ValueError(a)
    if b == "rival_sabotage":
        t -= 1
        h += 1
    elif b == "rival_flatter":
        t += 1
        e = max(0, e-1)
    elif b != "rival_wait":
        raise ValueError(b)
    if a != "wait":
        scene += 1
    return State(clamp(t, -3, 3), clamp(e, 0, 4),
                 clamp(h, 0, 4), clamp(scene, 0, 5))


ALL = tuple(State(t, e, h, scene)
            for t in range(-3, 4) for e in range(5)
            for h in range(5) for scene in range(6))


@lru_cache(maxsize=None)
def win(s, k, robust=False):
    """Finite-horizon secret-ending winning set, with goal states absorbing."""
    if target(s):
        return True
    if k <= 0 or terminal(s):
        return False
    return any(
        all(win(step(s, a, b), k-1, robust)
            for b in (RIVALS if robust else (other_policy(s),)))
        for a in ACTIONS
    )


def authorized(s, a, k, robust=False):
    if terminal(s):
        return False
    return all(win(step(s, a, b), k-1, robust)
               for b in (RIVALS if robust else (other_policy(s),)))


def manhattan(s, t):
    return sum(abs(x-y) for x, y in zip(
        (s.trust, s.evidence, s.heat, s.scene),
        (t.trust, t.evidence, t.heat, t.scene)))


def radius_from_winning_set(s, losers):
    """Largest nonnegative integer ball around s entirely within K_k."""
    if not losers:
        return None
    return min(manhattan(s, t) for t in losers) - 1


def one_step_resilience(horizon, robust):
    """Maximize minimum post-action viability radius among certified actions.

    A goal terminal is scored as completed, since no next decision occurs.
    This is a post-action integer-coordinate perturbation model only.
    """
    future_winning = {s for s in ALL if win(s, horizon - 1, robust)}
    future_losers = tuple(s for s in ALL if s not in future_winning)
    radius = {s: radius_from_winning_set(s, future_losers)
              for s in future_winning}
    candidates = [s for s in ALL
                  if any(authorized(s, a, horizon, robust) for a in ACTIONS)]
    original_good = chosen_good = improved = 0
    for s in candidates:
        actions = [a for a in ACTIONS if authorized(s, a, horizon, robust)]

        def score(a):
            successors = [step(s, a, b)
                          for b in (RIVALS if robust else (other_policy(s),))]
            return min(10**6 if target(ns) else radius[ns]
                       for ns in successors)

        first = actions[0]
        best = max(actions, key=score)  # fixed vocabulary tie rule
        r0, rb = score(first), score(best)
        original_good += r0 >= 1
        chosen_good += rb >= 1
        improved += rb > r0
    return {
        "decision_states": len(candidates),
        "priority_action_one_step_resilient": original_good,
        "depth_selected_one_step_resilient": chosen_good,
        "states_with_depth_improvement": improved,
    }


def main():
    result = {"model": "CCA-05 secret_ending, 1050 exact symbolic states",
              "horizons": {}}
    for robust in (False, True):
        hs = {}
        for horizon in (3, 4, 5, 6):
            winning = {s for s in ALL if win(s, horizon, robust)}
            losing = tuple(s for s in ALL if s not in winning)
            radii = Counter(radius_from_winning_set(s, losing) for s in winning)
            certified_actions = sum(authorized(s, a, horizon, robust)
                                    for s in ALL for a in ACTIONS)
            hs[str(horizon)] = {
                "viable_states": len(winning),
                "winning_actions": certified_actions,
                "radius_distribution": dict(sorted((str(k), v)
                                                   for k, v in radii.items())),
                "min_radius": min(radii) if radii else None,
                "fraction_boundary": (float(radii.get(0, 0) / len(winning))
                                      if winning else None),
                "successor_resilience": (one_step_resilience(horizon, robust)
                                         if horizon == 6 else None),
            }
        result["horizons"]["nondeterministic_rival" if robust
                           else "modeled_rival"] = hs
    print(json.dumps(result, sort_keys=True, indent=2))
    return result


if __name__ == "__main__":
    main()
