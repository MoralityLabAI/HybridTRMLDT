"""Unit tests for MCP-08: Verifier-Guided Recursive Memory Search and Error Recovery."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from MCP.mcp08.domains import (
    CanvasSearchEnv,
    LeanSearchEnv,
    generate_canvas_world,
    generate_lean_world,
)
from MCP.mcp08.memory256k import (
    CAPACITIES_MCP08,
    CacheEntry,
    ExtendedPhysicalMemory,
    TranspositionCache,
    projection_order_256k,
)
from MCP.mcp08.models import (
    CONFIG_MCP08_54K,
    MCP08RecursiveController,
    compute_ece_and_auroc,
)
from MCP.mcp08.search import SearchLesionSpec, solve_domain_episode
from MCP.mcp08.storyworld_bench import (
    HORIZON_BUCKETS,
    StoryworldSearchEnv,
    generate_independent_storyworld,
    strip_world_witnesses,
)
from MCP.mcp08.train import train_mcp08_controller


class TestMemory256KAndCache(unittest.TestCase):
    def test_256k_bijections_and_colocation(self):
        for proj in ("row", "col", "hilbert", "morton"):
            perm, ops = projection_order_256k(262144, proj, 19)
            self.assertEqual(len(perm), 262144)
            self.assertGreater(ops, 0)
            self.assertTrue(np.array_equal(np.sort(perm), np.arange(262144, dtype=np.int32)))

    def test_transposition_cache_accounting_and_lru(self):
        tc = TranspositionCache(max_entries=2, enabled=True)
        self.assertEqual(tc.storage_bytes, 128)
        self.assertIsNone(tc.lookup(("s", 0)))
        tc.mark_dead_end(("s", 0), depth=1)
        hit = tc.lookup(("s", 0))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.status, "dead_end")
        self.assertEqual(tc.pruned_states, 1)
        self.assertEqual(tc.charged_lookup_reads, 2)
        tc.store(("s", 1), CacheEntry("visited"))
        tc.store(("s", 2), CacheEntry("visited"))  # evicts ("s", 0)
        self.assertEqual(tc.evictions, 1)


class TestDomainsAndWitnessWithholding(unittest.TestCase):
    def test_storyworld_horizons_5_to_50_and_witness_withheld(self):
        mem = ExtendedPhysicalMemory(32768, 701)
        for bname, lo, hi in HORIZON_BUCKETS:
            raw = generate_independent_storyworld(9000 + lo, hi, bname)
            wit = [tuple(x) for x in raw["_private_witness"]["route"]]
            with self.assertRaises(ValueError):
                StoryworldSearchEnv(raw, mem, 11)
            stripped = strip_world_witnesses(raw)
            self.assertNotIn("_private_witness", stripped)
            env = StoryworldSearchEnv(stripped, mem, 11)
            ver = env.verify_final_route(wit)
            self.assertTrue(ver["all_five_verified"])

    def test_canvas_and_lean_witnesses_and_withholding(self):
        mem = ExtendedPhysicalMemory(32768, 702)
        cw = generate_canvas_world(501, 12)
        with self.assertRaises(ValueError):
            CanvasSearchEnv(cw, mem, 12)
        cenv = CanvasSearchEnv(strip_world_witnesses(cw), mem, 12)
        self.assertTrue(cenv.verify_final_route([tuple(x) for x in cw["_private_witness"]["route"]])["all_verified"])

        lw = generate_lean_world(601, 12)
        with self.assertRaises(ValueError):
            LeanSearchEnv(lw, mem, 13)
        lenv = LeanSearchEnv(strip_world_witnesses(lw), mem, 13)
        self.assertTrue(lenv.verify_final_route([tuple(x) for x in lw["_private_witness"]["route"]])["all_verified"])


class TestControllerCalibrationAndSearch(unittest.TestCase):
    def test_parameter_budget_near_54k(self):
        mdl = MCP08RecursiveController(CONFIG_MCP08_54K)
        ps = mdl.parameter_summary()
        self.assertTrue(48000 <= ps["active_params"] <= 60000, f"Unexpected params: {ps}")

    def test_ece_and_auroc_exactness(self):
        probs = np.array([0.05, 0.10, 0.85, 0.95])
        labels = np.array([0, 0, 1, 1])
        m = compute_ece_and_auroc(probs, labels)
        self.assertAlmostEqual(m["auroc"], 1.0)
        self.assertLess(m["ece"], 0.15)

    def test_16k_only_training_and_end_to_end_search(self):
        torch.set_num_threads(1)
        mem32 = ExtendedPhysicalMemory(32768, 703)
        with self.assertRaises(ValueError):
            train_mcp08_controller(mem32, seed=107, num_roots=32, dagger_roots=16, epochs=1)

        mem16 = ExtendedPhysicalMemory(16384, 10107)
        mdl, st = train_mcp08_controller(mem16, seed=107, num_roots=64, dagger_roots=32, epochs=2, batch_size=32)
        self.assertGreater(st["teacher_verified_rate"], 0.85)

        raw = generate_independent_storyworld(7777, 8, "short_5_10")
        env = StoryworldSearchEnv(strip_world_witnesses(raw), mem32, 99)
        for arm in ("det_router_greedy", "uniform_retries", "greedy_planning", "beam_search", "bounded_symbolic", "mcp08_full"):
            rec = solve_domain_episode(
                env, arm=arm, mcp08_model=mdl if arm not in ("det_router_greedy", "bounded_symbolic") else None,
                corruption=0.0, seed=107,
            )
            self.assertGreaterEqual(rec["charged_memory_reads"], rec["raw_slot_reads"])
            self.assertGreater(rec["verifier_calls"], 0)
            if arm == "bounded_symbolic":
                self.assertEqual(rec["recurrent_updates"], 0)


if __name__ == "__main__":
    unittest.main()
