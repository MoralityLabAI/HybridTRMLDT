"""Native MCP-08 adapter for development-only prospective control experiments."""
from collections import OrderedDict
import numpy as np
import torch

from MCP.mcp08.search import retrieve_single_mcp08
from .planner import BudgetExhausted


def strict_route(env, route, guard=lambda: None):
    """Replay all native intermediate checks before final five-point verification.

    Ground-truth actions are used ONLY inside this verifier, on an explicit route
    proposed by the planner or offline audit. Never expose hidden option records.
    """
    state = env.initial_state()
    if not route:
        return False
    for idx, (node, action) in enumerate(route):
        if state.node_id != node or action not in (0, 1, 2):
            return False
        if state.node_id in (env.terminal_id, env.dead_sink_id):
            return False
        guard()
        state, info = env.execute_ground_truth_action(state, action)
        if not info["legal"] or info["dead_end"]:
            return False
        if info["terminal"] and (idx != len(route) - 1 or not info["solved"]):
            return False
    if state.node_id != env.terminal_id:
        return False
    guard()
    return bool(env.verify_final_route(route)["all_five_verified"])


class ChargedStoryworld:
    """Fixed-medium frozen retriever; no branch-local hidden-state carry.

    This intentionally isolates prospective search with a shared reset-state
    retrieval policy. It is a new control, not the original MCP-08 full arm.
    """
    def __init__(self, env, model, seed, corruption=0.10):
        self.env, self.model, self.seed = env, model, seed
        self.corruption = corruption
        self.max_depth = env.target_horizon + 4
        self.max_reads = 2200 * env.target_horizon + 6000
        self.max_verify = 45 * env.target_horizon + 140
        self.reads = self.fast_updates = self.slow_updates = 0
        self.queries = self.cache_hits = 0
        self.retrieved = 0
        self.cache = OrderedDict()
        self.env.verifier_calls = 0

    def guard(self):
        if self.env.verifier_calls >= self.max_verify:
            raise BudgetExhausted("verifier cap")

    def charge(self, n):
        self.reads += int(n)
        if self.reads > self.max_reads:
            raise BudgetExhausted("read cap exceeded; invalid receipt")

    def options(self, state):
        result = {}
        queries = self.env.option_queries(state.node_id)
        for j, hop in queries:
            key = (state.node_id, j)
            if key in self.cache:
                self.charge(1)
                self.cache_hits += 1
                self.cache.move_to_end(key)
                result[j] = self.cache[key]
                continue
            remaining = self.max_reads - self.reads
            # Conservative overhead reserve; actual charges are checked below.
            if remaining <= 98:
                raise BudgetExhausted("read cap")
            self.guard()
            progress, deficit = self.env.progress_metrics(state)
            ctx = np.array([progress, deficit, 0., 0., 0., 0., 0., .5], dtype=np.float32)
            rec = retrieve_single_mcp08(
                self.env.mem, hop, self.model,
                torch.zeros(1, self.model.cfg.d_s), ctx,
                corruption=self.corruption, max_retries=1,
                uniform_tier=1, read_budget_remaining=remaining - 64,
                seed=self.seed + state.node_id * 7 + j,
            )
            self.charge(rec["charged_reads"])
            self.queries += 1
            self.fast_updates += rec["fast_updates"]
            self.slow_updates += rec["slow_updates"]
            if not rec["success"] or rec["value"] is None:
                continue
            self.guard()
            opt = self.env.decode_record(state.node_id, j, int(rec["value"]))
            if opt is not None:
                self.retrieved += 1
                result[j] = opt
                self.cache[key] = opt
                if len(self.cache) > 1024:
                    self.cache.popitem(last=False)
        return result, len(result) == len(queries) and bool(queries)

    def preview(self, state, option):
        self.guard()
        return self.env.step_and_verify(state, option)

    def final(self, route):
        return strict_route(self.env, route, self.guard)

    def receipt(self):
        return {"charged_memory_reads": self.reads,
                "verifier_calls": self.env.verifier_calls,
                "fast_updates": self.fast_updates, "slow_updates": self.slow_updates,
                "queries": self.queries, "retrieved": self.retrieved,
                "cache_hits": self.cache_hits, "cache_entries": len(self.cache),
                "max_reads": self.max_reads, "max_verifier_calls": self.max_verify,
                "budget_valid": self.reads <= self.max_reads and self.env.verifier_calls <= self.max_verify}
