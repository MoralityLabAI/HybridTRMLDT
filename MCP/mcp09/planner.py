"""Budgeted pre-commit lookahead using only retrieved actions and native previews.

A missing record is UNKNOWN, never a certificate of impossibility. All speculative
calls go through the same charged adapter as live actions. No simulator state or
private witness is available to this module.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class BudgetExhausted(RuntimeError):
    pass


class Viability(IntEnum):
    DEAD = 0
    UNKNOWN = 1
    FRONTIER = 2
    SOLVED = 3


@dataclass(frozen=True)
class Candidate:
    action: int
    state: Any
    info: dict


class Planner:
    def __init__(self, adapter, depth=0, max_backtracks=32):
        if depth < 0 or max_backtracks < 0:
            raise ValueError("negative search limit")
        self.adapter = adapter
        self.depth = depth
        self.max_backtracks = max_backtracks
        self.backtracks = 0
        self.lookahead_nodes = 0
        self.irreversible_commits = 0
        self.rejected_dead = 0
        self.unknown_commits = 0
        self.trace = []
        self.route = ()
        self.lock_depth = 0

    def expand(self, state):
        # Adapter returns only decoded, externally retrieved records.
        options, complete = self.adapter.options(state)
        candidates = []
        for action, option in sorted(options.items()):
            nxt, info = self.adapter.preview(state, option)
            candidates.append(Candidate(action, nxt, info))
        return candidates, complete

    @staticmethod
    def extend(route, state, candidate):
        return route + ((state.node_id, candidate.action),)

    def viability(self, state, route, depth):
        """A bounded, three-valued search with an explicit open frontier."""
        self.lookahead_nodes += 1
        if depth == 0:
            return Viability.FRONTIER
        candidates, complete = self.expand(state)
        best = Viability.DEAD if complete else Viability.UNKNOWN
        for candidate in sorted(candidates, key=lambda c: (-c.info["score"], c.action)):
            v = self.inspect(state, route, candidate, depth - 1)
            best = max(best, v)
            if best == Viability.SOLVED:
                break
        return best

    def inspect(self, state, route, candidate, depth):
        info = candidate.info
        if not info.get("legal", True) or info["dead_end"]:
            return Viability.DEAD
        new_route = self.extend(route, state, candidate)
        if info["terminal"]:
            return (Viability.SOLVED if info["solved"] and self.adapter.final(new_route)
                    else Viability.DEAD)
        return self.viability(candidate.state, new_route, depth)

    def order(self, state, route, candidates):
        ranked = []
        for candidate in candidates:
            info = candidate.info
            if not info.get("legal", True) or info["dead_end"]:
                continue
            if self.depth and info["irreversible"]:
                v = self.inspect(state, route, candidate, self.depth)
                if v == Viability.DEAD:
                    self.rejected_dead += 1
                    self.trace.append({"route": list(route), "action": candidate.action,
                                       "event": "reject_proven_dead"})
                    continue
            else:
                v = Viability.FRONTIER
            ranked.append((v, float(info["score"]), -candidate.action, candidate))
        ranked.sort(key=lambda x: x[:3], reverse=True)
        return ranked

    def run(self, initial):
        try:
            solved, locked = self.visit(initial, (), 0)
            reason = "solved" if solved else ("irreversible_lock" if locked else "search_exhausted")
        except BudgetExhausted:
            solved, reason = False, "budget_exhausted"
        return {"success": solved, "reason": reason, "route": list(self.route),
                "lock_depth": self.lock_depth, "backtracks": self.backtracks,
                "lookahead_nodes": self.lookahead_nodes,
                "irreversible_commits": self.irreversible_commits,
                "rejected_dead": self.rejected_dead,
                "unknown_commits": self.unknown_commits, "trace": self.trace}

    def visit(self, state, route, depth):
        self.route = route
        if depth >= self.adapter.max_depth:
            return False, False
        candidates, _complete = self.expand(state)
        ranked = self.order(state, route, candidates)
        for i, (v, _score, _, candidate) in enumerate(ranked):
            if i:
                if self.backtracks >= self.max_backtracks:
                    return False, False
                self.backtracks += 1
            info = candidate.info
            new_route = self.extend(route, state, candidate)
            self.route = new_route
            irrev = bool(info["irreversible"])
            if irrev:
                self.irreversible_commits += 1
                self.lock_depth = len(new_route)
                self.unknown_commits += int(v == Viability.UNKNOWN)
            self.trace.append({"route": list(route), "action": candidate.action,
                               "event": "commit", "irreversible": irrev,
                               "viability": v.name})
            if info["terminal"]:
                solved, locked = bool(info["solved"] and self.adapter.final(new_route)), False
            else:
                solved, locked = self.visit(candidate.state, new_route, depth + 1)
            if solved:
                return True, False
            # A failed descendant irreversible commit also locks every ancestor.
            if irrev or locked:
                return False, True
        return False, False
