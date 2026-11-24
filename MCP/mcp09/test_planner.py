"""Dependency-free regression tests for the reference planner."""
import unittest
from dataclasses import dataclass
from .planner import Planner, Viability, BudgetExhausted


@dataclass(frozen=True)
class State:
    node_id: int


def edge(target, score=0, irrev=False, terminal=False, solved=False, dead=False):
    return (target, dict(score=score, irreversible=irrev, terminal=terminal,
                         solved=solved, dead_end=dead, legal=True))


class Mock:
    max_depth = 10

    def __init__(self, graph, missing=(), budget=100, final_ok=True):
        self.graph, self.missing = graph, set(missing)
        self.budget, self.cost, self.final_ok = budget, 0, final_ok

    def tick(self):
        if self.cost >= self.budget:
            raise BudgetExhausted()
        self.cost += 1

    def options(self, state):
        self.tick()
        opts = self.graph.get(state.node_id, {})
        out = {j: e for j, e in opts.items() if (state.node_id, j) not in self.missing}
        return out, bool(opts) and len(out) == len(opts)

    def preview(self, state, option):
        self.tick()
        return State(option[0]), option[1]

    def final(self, route):
        self.tick()
        return self.final_ok


class PlannerTests(unittest.TestCase):
    def trap(self):
        return {0: {0: edge(1, 10, True), 1: edge(2, 1, True)},
                1: {0: edge(3, dead=True)},
                2: {0: edge(4, terminal=True, solved=True)}}

    def test_precommit_avoids_irreversible_trap(self):
        base = Planner(Mock(self.trap()), depth=0).run(State(0))
        probe = Planner(Mock(self.trap()), depth=2).run(State(0))
        self.assertFalse(base["success"])
        self.assertEqual(base["reason"], "irreversible_lock")
        self.assertTrue(probe["success"])
        self.assertEqual(probe["route"][0], (0, 1))
        self.assertEqual(probe["rejected_dead"], 1)

    def test_missing_record_is_unknown(self):
        p = Planner(Mock(self.trap(), missing=[(1, 0)]), depth=2)
        self.assertEqual(p.viability(State(1), (), 1), Viability.UNKNOWN)

    def test_nested_lock_forbids_ancestor_restore(self):
        graph = {0: {0: edge(1, 10), 1: edge(4, terminal=True, solved=True)},
                 1: {0: edge(2, 10, True)}, 2: {0: edge(3, dead=True)}}
        rec = Planner(Mock(graph), depth=0).run(State(0))
        self.assertEqual(rec["reason"], "irreversible_lock")
        self.assertEqual(rec["backtracks"], 0)

    def test_reversible_failure_allows_alternative(self):
        graph = {0: {0: edge(1, 10), 1: edge(4, terminal=True, solved=True)},
                 1: {0: edge(3, dead=True)}}
        rec = Planner(Mock(graph), depth=0).run(State(0))
        self.assertTrue(rec["success"])
        self.assertEqual(rec["backtracks"], 1)

    def test_budget_charges_speculation(self):
        m = Mock(self.trap(), budget=4)
        rec = Planner(m, depth=2).run(State(0))
        self.assertEqual(rec["reason"], "budget_exhausted")
        self.assertEqual(m.cost, 4)
        self.assertEqual(rec["irreversible_commits"], 0)

    def test_final_verification_is_required(self):
        graph = {0: {0: edge(1, terminal=True, solved=True)}}
        self.assertFalse(Planner(Mock(graph, final_ok=False)).run(State(0))["success"])

    def test_unknown_is_not_pruned(self):
        p = Planner(Mock(self.trap(), missing=[(1, 0)]), depth=2)
        candidates, _ = p.expand(State(0))
        order = p.order(State(0), (), candidates)
        self.assertEqual(len(order), 2)
        self.assertEqual(order[-1][0], Viability.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
