"""Routing objective J(a) = E[R_hat(a)] + lambda*C(a) + mu*KL_hat(a) + gamma*F(a).

The router reads only exposed ``MeshUnit`` fields plus the cheap local answer. Regret
estimates come from a ``RegretModel``; the default is the deterministic structural
model, and any learned model with the same ``predict`` signature plugs in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Protocol

from .units import K, Block, Mesh, MeshUnit, unit_of

LOW, HIGH = 0.2, 0.8
ROUTED = ("local", "bp", "exact", "abstain")


@dataclass(frozen=True)
class CostModel:
    """Kernel seconds = a_op + b_op * work, plus the J weights and the exact budget."""

    a: Mapping[str, float] = field(default_factory=lambda: {o: 0.0 for o in ROUTED})
    b: Mapping[str, float] = field(default_factory=lambda: {"local": 1e-9, "bp": 1e-9, "exact": 1e-9, "abstain": 0.0})
    lam: float = 0.01      # utility per millisecond of kernel time
    mu: float = 0.05       # utility per nat of discarded correlation when answering locally
    gamma: float = 0.002   # utility per operator kernel not yet launched this step
    exact_budget: int = 1 << 22

    def seconds(self, op: str, work: float) -> float:
        return 0.0 if op == "abstain" else self.a[op] + self.b[op] * work


@dataclass(frozen=True)
class RouteFeatures:
    p_local: float
    margin: float
    conf: float
    n_query_blocks: int
    comp_units: int
    n_cross_comp: int
    kl_hat_query: float
    kl_hat_comp: float
    entropy_query: float
    staleness: int
    work: Dict[str, float]


class RegretModel(Protocol):
    """Hook for a learned regret model: expected regret of operator ``op``."""

    def predict(self, features: RouteFeatures, op: str) -> float: ...


@dataclass(frozen=True)
class StructuralRegret:
    """R_hat for approximations = kappa*KL_hat, plus kappa0 within `near` of a threshold.

    Exact has zero regret. Abstaining resolves nothing, so it carries the
    near-threshold term and the forgone information; away from both its regret is the
    forgone utility of a confident answer.
    """

    kappa_local: float = 1.0
    kappa_bp: float = 0.5
    kappa_abstain: float = 1.0
    kappa0: float = 0.5
    near: float = 0.1

    def predict(self, f: RouteFeatures, op: str) -> float:
        near = self.kappa0 * (f.margin < self.near)
        if op == "exact":
            return 0.0
        if op == "local":
            return self.kappa_local * f.kl_hat_query + near
        if op == "bp":
            return self.kappa_bp * f.kl_hat_query + near
        return max(0.0, 5 * f.conf - 4) + self.kappa_abstain * f.kl_hat_query + near


def route_features(mesh: Mesh, q: int) -> RouteFeatures:
    """Features built only from exposed unit fields and the local answer."""
    p, _ = mesh.local(q)
    qb = sorted({mesh.block_of(unit_of(v)) for v in range(K) if (q >> v) & 1})
    comp = mesh.relevant(q)
    eq = [mesh.expose(b) for b in qb]
    ec = [mesh.expose(b) for b in comp]
    exact_work = mesh.exact_cost(q)[0]
    bp_work = sum(u.cost_estimates["bp"] for u in ec) // 2
    return RouteFeatures(
        p_local=p,
        margin=min(abs(p - HIGH), abs(p - LOW)),
        conf=max(p, 1 - p),
        n_query_blocks=len(qb),
        comp_units=sum(len(b) for b in comp),
        n_cross_comp=sum(1 for _, bl in mesh.cross_factors() if bl[0] in comp),
        kl_hat_query=sum(u.info_lost for u in eq),
        kl_hat_comp=sum(u.info_lost for u in ec),
        entropy_query=sum(u.uncertainty for u in eq),
        staleness=min(u.verification.factors_since for u in eq),
        work={"local": sum(u.cost_estimates["local"] for u in eq), "bp": max(bp_work, 1),
              "exact": exact_work, "abstain": 0},
    )


def objective_J(f: RouteFeatures, op: str, regret: RegretModel, cost: CostModel,
                launched: Iterable[str] = ()) -> float:
    """J for one operator: regret + lambda*cost_ms + mu*KL_hat (local) + gamma*fragmentation."""
    j = regret.predict(f, op)
    j += cost.lam * 1000.0 * cost.seconds(op, f.work[op])
    if op == "local":
        j += cost.mu * f.kl_hat_query
    if op != "abstain" and op not in set(launched):
        j += cost.gamma
    return j


@dataclass(frozen=True)
class RoutingDecision:
    op: str
    J: Dict[str, float]
    features: RouteFeatures


class Router:
    """argmin_a J(a) over operators whose work fits the budget; ties break in ROUTED order."""

    def __init__(self, regret: RegretModel | None = None, cost: CostModel | None = None) -> None:
        self.regret = regret or StructuralRegret()
        self.cost = cost or CostModel()

    def route(self, mesh: Mesh, q: int, launched: Iterable[str] = ()) -> RoutingDecision:
        f = route_features(mesh, q)
        ops = [o for o in ROUTED if o != "exact" or f.work["exact"] <= self.cost.exact_budget]
        if not mesh.cross_factors():
            ops = [o for o in ops if o != "bp"]
        js = {o: objective_J(f, o, self.regret, self.cost, launched) for o in ops}
        best = min(ops, key=lambda o: (js[o], ROUTED.index(o)))
        return RoutingDecision(best, js, f)

    def answer(self, mesh: Mesh, q: int, launched: Iterable[str] = ()):
        """Route, run the chosen operator, and map the probability to an action."""
        d = self.route(mesh, q, launched)
        if d.op == "abstain":
            return d, None, 0
        p, work = getattr(mesh, d.op)(q)
        return d, p, work
