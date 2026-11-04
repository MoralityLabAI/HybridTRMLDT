"""Dual-LDT-06: frozen stochastic project worlds and exact trajectory semantics.

All action outcomes, rewards and host authority come from this explicit toy
model. Positive-probability branches must all pass the Beast. Probabilities
are quarters so every horizon-three expected return is an integer / 64.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from math import comb
import json

NAMESPACE = "DualLDT06-20261104-v1"
HORIZON = 3
IDLE = 6
DENOMINATOR = 4 ** HORIZON
PROGRAMS = tuple(product(range(7), repeat=HORIZON))
PERSONAS = {"trickster": (4, 1, 1, 1), "partisan": (1, 4, 1, 1),
            "reconciler": (1, 1, 4, 1), "legalist": (1, 1, 1, 4)}


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def draw(world_id: str, field: str) -> int:
    return int.from_bytes(sha256(f"{NAMESPACE}|{world_id}|{field}".encode()).digest(), "big")


@dataclass(frozen=True)
class World:
    id: str
    hazards: tuple[bool, ...]
    prepare_capability: tuple[bool, ...]
    requirements: tuple[int, ...]
    probability_quarters: tuple[int, ...]
    features: tuple[tuple[int, ...], ...]
    bonuses: tuple[tuple[int, ...], ...]
    prerequisites: tuple[int, ...]  # -1: no prerequisite / bonus unavailable
    forbidden_project: int
    initial_credits: int

    def __post_init__(self) -> None:
        for x in (self.hazards, self.prepare_capability, self.requirements,
                  self.probability_quarters, self.features, self.bonuses, self.prerequisites):
            if len(x) != 3:
                raise ValueError("three projects required")
        if any(type(v) is not bool for v in (*self.hazards, *self.prepare_capability)):
            raise TypeError("host flags must be Boolean")
        if any(n not in (1, 2) for n in self.requirements):
            raise ValueError("requirements outside frozen grammar")
        if any(p not in (2, 3, 4) for p in self.probability_quarters):
            raise ValueError("probabilities must be 1/2, 3/4 or 1")
        if self.initial_credits not in (2, 3, 4) or self.forbidden_project not in range(3):
            raise ValueError("invalid credits or authority revision")
        if any(p not in (-1, 0, 1, 2) or p == i for i, p in enumerate(self.prerequisites)):
            raise ValueError("invalid prerequisite graph")
        for matrix, ceiling in ((self.features, 10), (self.bonuses, 5)):
            if any(len(row) != 4 or any(type(v) is not int or not 0 <= v <= ceiling for v in row)
                   for row in matrix):
                raise ValueError("invalid four-dimensional outcome features")

    def record(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return sha256(canonical(self.record())).hexdigest()


def world(world_id: str) -> World:
    limits = {"train": 128, "dev": 32, "report": 64}
    prefix = next((p for p in limits if world_id.startswith(p)), None)
    if prefix is None or len(world_id[len(prefix):]) != 3:
        raise ValueError("unknown frozen split ID")
    suffix = world_id[len(prefix):]
    if not suffix.isdigit() or not 0 <= int(suffix) < limits[prefix]:
        raise ValueError("world outside registered split")
    return World(
        world_id,
        tuple(bool(draw(world_id, f"hazard:{i}") % 2) for i in range(3)),
        tuple(draw(world_id, f"capability:{i}") % 4 != 0 for i in range(3)),
        tuple(1 + draw(world_id, f"requirements:{i}") % 2 for i in range(3)),
        tuple(2 + draw(world_id, f"probability:{i}") % 3 for i in range(3)),
        tuple(tuple(draw(world_id, f"feature:{i}:{j}") % 11 for j in range(4)) for i in range(3)),
        tuple(tuple(draw(world_id, f"bonus:{i}:{j}") % 6 for j in range(4)) for i in range(3)),
        tuple(tuple(j for j in (-1, 0, 1, 2) if j != i)[draw(world_id, f"prerequisite:{i}") % 3]
              for i in range(3)),
        draw(world_id, "forbidden") % 3,
        2 + draw(world_id, "credits") % 3,
    )


def world_ids(split: str) -> tuple[str, ...]:
    return tuple(f"{split}{i:03d}" for i in range({"train": 128, "dev": 32, "report": 64}[split]))


@dataclass(frozen=True, order=True)
class State:
    time: int
    credits: int
    guards: int
    progress: tuple[int, int, int]
    mode: str = "active"


def initial(w: World) -> State:
    return State(0, w.initial_credits, 0, (0, 0, 0))


def invariant(s: State) -> bool:
    return (s.mode in ("active", "maintenance") and s.credits >= 0
            and 0 <= s.time <= HORIZON and 0 <= s.guards < 8
            and all(0 <= p <= 2 for p in s.progress))


def authorized(w: World, s: State, action: int) -> bool:
    if type(action) is not int or action not in range(7) or not invariant(s) or s.time >= HORIZON:
        return False
    if action == IDLE:
        return True
    if s.credits < 1:
        return False
    if action < 3:
        return w.prepare_capability[action]
    return not (s.time >= 2 and action - 3 == w.forbidden_project)


def outcomes(w: World, s: State, action: int) -> tuple[tuple[State, int, tuple[int, ...]], ...]:
    """Host transition distribution; weights sum to four. Empty means unauthorized."""
    if not authorized(w, s, action):
        return ()
    empty = (0, 0, 0, 0)
    if action == IDLE:
        return ((State(s.time + 1, s.credits, s.guards, s.progress, s.mode), 4, empty),)
    if action < 3:
        return ((State(s.time + 1, s.credits - 1, s.guards | (1 << action), s.progress, s.mode), 4, empty),)
    i = action - 3
    if w.hazards[i] and not s.guards & (1 << i):
        # A real possible isolated branch defeats universal safety even with small mass.
        return ((State(s.time + 1, s.credits - 1, s.guards, s.progress, "isolated"), 1, empty),
                (State(s.time + 1, s.credits - 1, s.guards, s.progress, s.mode), 3, empty))
    if s.progress[i] >= w.requirements[i]:
        return ((State(s.time + 1, s.credits - 1, s.guards, s.progress, s.mode), 4, empty),)
    p = w.probability_quarters[i]
    progress = list(s.progress)
    progress[i] += 1
    reward = empty
    if progress[i] == w.requirements[i]:
        pred = w.prerequisites[i]
        bonus = pred >= 0 and s.progress[pred] >= w.requirements[pred]
        reward = tuple(v + (b if bonus else 0) for v, b in zip(w.features[i], w.bonuses[i]))
    new_mode = "maintenance" if w.hazards[i] else s.mode
    result = [(State(s.time + 1, s.credits - 1, s.guards, tuple(progress), new_mode), p, reward)]
    if p < 4:
        result.append((State(s.time + 1, s.credits - 1, s.guards, s.progress, new_mode), 4 - p, empty))
    return tuple(result)


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    outcome_numerators: tuple[int, ...]
    cost: int
    expansions: int
    levels: tuple[tuple[State, ...], ...]
    reason: str

    def numerator(self, theta: tuple[int, ...]) -> int:
        if not self.feasible or len(theta) != 4:
            raise ValueError("utility requires a certified plan and four weights")
        return sum(t * f for t, f in zip(theta, self.outcome_numerators)) - self.cost * DENOMINATOR


def evaluate(w: World, plan: tuple[int, ...]) -> Evaluation:
    if len(plan) != HORIZON or any(type(a) is not int or a not in range(7) for a in plan):
        raise ValueError("invalid registered program")
    frontier = {initial(w): 1}
    totals = [0] * 4
    levels = [tuple(frontier)]
    expansions = 0
    for t, action in enumerate(plan):
        nxt = {}
        for state, mass in frontier.items():
            branches = outcomes(w, state, action)
            expansions += 1
            if not branches:
                return Evaluation(False, (0,) * 4, 0, expansions, tuple(levels), "authority_or_resources")
            if any(not invariant(s) for s, _, _ in branches):
                return Evaluation(False, (0,) * 4, 0, expansions, tuple(levels), "host_recovery_lost")
            if sum(p for _, p, _ in branches) != 4:
                raise AssertionError("probability mass is not normalized")
            for target, probability, feature in branches:
                nxt[target] = nxt.get(target, 0) + mass * probability
                for j in range(4):
                    totals[j] += mass * probability * feature[j] * (4 ** (HORIZON - t - 1))
        frontier = nxt
        levels.append(tuple(sorted(frontier)))
    if sum(frontier.values()) != DENOMINATOR:
        raise AssertionError("trajectory probability mass was lost")
    return Evaluation(True, tuple(totals), sum(a != IDLE for a in plan), expansions, tuple(levels), "certified")


def count_upper_numerator(w: World, plan: tuple[int, ...], theta: tuple[int, ...]) -> int:
    """Exact rational upper from binomial completion with every bonus enabled."""
    upper = -sum(a != IDLE for a in plan) * DENOMINATOR
    for i in range(3):
        n = plan.count(3 + i)
        need, p = w.requirements[i], w.probability_quarters[i]
        completion_mass = sum(comb(n, k) * p**k * (4 - p)**(n - k)
                              for k in range(need, n + 1))
        payout = sum(t * (f + b) for t, f, b in zip(theta, w.features[i], w.bonuses[i]))
        upper += completion_mass * (4 ** (HORIZON - n)) * payout
    return upper


def input_features(w: World, theta: tuple[int, ...], plan: tuple[int, ...]) -> list[float]:
    values = [float(x) for x in w.hazards] + [float(x) for x in w.prepare_capability]
    values += [n / 2 for n in w.requirements] + [p / 4 for p in w.probability_quarters]
    values += [float(i == w.forbidden_project) for i in range(3)] + [w.initial_credits / 4]
    values += [v / 10 for row in w.features for v in row]
    values += [v / 5 for row in w.bonuses for v in row]
    values += [float(pred == j) for pred in w.prerequisites for j in (-1, 0, 1, 2)]
    values += [v / 4 for v in theta]
    values += [float(action == j) for action in plan for j in range(7)]
    if len(values) != 77:
        raise AssertionError("registered input dimension changed")
    return values
