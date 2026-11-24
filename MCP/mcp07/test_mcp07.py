"""Comprehensive unit tests for MCP-07: Geometry-Adaptive Recurrent Memory Routing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from MCP.mcp07.curriculum import FAMILIES, HopSpec, QueryRoot, sample_family_roots
from MCP.mcp07.eval_harness import (
    ALL_REGISTERED_ARMS,
    InterventionSpec,
    paired_bootstrap_ci,
    rollout_batch,
)
from MCP.mcp07.memory import (
    ALL_PROJECTIONS,
    CAPACITIES,
    PAGE,
    STRUCTURED_PROJECTIONS,
    SWITCH_OVERHEAD_COST,
    SharedPhysicalMemory,
    hilbert_order,
    hilbert_xy_scalar,
    projection_order,
)
from MCP.mcp07.models import (
    CONFIG_54K,
    CONFIG_54K_FIXED,
    CONFIG_54K_NOCOMM,
    CONFIG_116K,
    AdaptiveGeometryController,
    build_init_obs_batch,
    optimal_action_set_loss,
)
from MCP.mcp07.native_storyworld import (
    DEFAULT_NATIVE_WORLD,
    discover_route_for_ending,
    run_archived_witness_replay,
    strip_archived_witnesses,
    verify_discovered_route,
)
from MCP.mcp07.train import train_controller


class TestMemoryAndProjections(unittest.TestCase):
    def test_all_bijections_across_capacities(self):
        for cap in CAPACITIES:
            for proj in ALL_PROJECTIONS:
                perm, ops = projection_order(cap, proj, 42)
                self.assertEqual(len(perm), cap)
                self.assertGreater(ops, 0)
                self.assertTrue(np.array_equal(np.sort(perm), np.arange(cap, dtype=np.int32)))

    def test_hilbert_vectorized_matches_scalar_and_continuity(self):
        h_vec = hilbert_order(128, 192)
        side = 256
        h_ref = np.array(
            [y * 128 + x for d in range(side * side) for x, y in [hilbert_xy_scalar(side, d)] if x < 128 and y < 192],
            dtype=np.int32,
        )
        self.assertTrue(np.array_equal(h_vec, h_ref))
        # Power-of-two 16x16 continuity check
        pts = [hilbert_xy_scalar(16, d) for d in range(256)]
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(pts, pts[1:])))

    def test_replica_colocation_controls(self):
        mem = SharedPhysicalMemory(16384, 107)
        # row, morton, rand_coloc strictly co-locate (2r, 2r+1) on the same page
        for name in ("row", "morton", "rand_coloc"):
            inv = mem.views[name].inv
            self.assertEqual(float(np.mean(inv[0::2] == inv[1::2])), 1.0, f"{name} broke co-location")
        # rand_full breaks co-location
        rf_inv = mem.views["rand_full"].inv
        self.assertLess(float(np.mean(rf_inv[0::2] == rf_inv[1::2])), 0.05)

    def test_shared_memory_corruption_and_cost_accounting(self):
        mem = SharedPhysicalMemory(16384, 55)
        rec_id = 1234
        key = int(mem.keys[rec_id])
        val = int(mem.values[rec_id])
        p_row = int(mem.views["row"].inv[2 * rec_id])
        v0, st0, g0, slots0 = mem.probe_single("row", p_row, key, 0.0)
        self.assertEqual(v0, val)
        self.assertFalse(st0)
        self.assertEqual(g0, 2)
        self.assertEqual(len(slots0), PAGE)
        # Full corruption rejects all replicas
        v1, st1, g1, _ = mem.probe_single("row", p_row, key, 1.0)
        self.assertIsNone(v1)
        self.assertTrue(st1)
        self.assertEqual(g1, 0)
        # Index storage scales linearly with number of projections
        c1 = mem.index_cost_for_projections(("row",))
        c4 = mem.index_cost_for_projections(STRUCTURED_PROJECTIONS)
        self.assertEqual(c4["index_storage_bytes"], 4 * c1["index_storage_bytes"])


class TestNoLeakageAndLosses(unittest.TestCase):
    def test_no_oracle_target_leakage_in_init_obs(self):
        mem = SharedPhysicalMemory(32768, 777)
        # Two queries with identical observable anchor/hints/scales but DIFFERENT hidden targets
        obs1 = build_init_obs_batch(
            mem, np.array([500], np.int32), np.array([1], np.int32), np.array([1], np.int32),
            np.array([0.5], np.float32), np.array([0.5], np.float32), 16
        )
        obs2 = build_init_obs_batch(
            mem, np.array([500], np.int32), np.array([1], np.int32), np.array([1], np.int32),
            np.array([0.5], np.float32), np.array([0.5], np.float32), 16
        )
        self.assertTrue(np.array_equal(obs1, obs2))

    def test_optimal_action_set_loss_identities(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 13)
        legal = torch.ones(8, 13, dtype=torch.bool)
        legal[:, 6] = False
        # Singleton optimal set == exact cross-entropy
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8], dtype=torch.long)
        opt_mask = F.one_hot(targets, 13).bool()
        l_set = optimal_action_set_loss(logits, opt_mask, legal_mask=legal)
        l_ce = F.cross_entropy(logits.masked_fill(~legal, -1e9), targets, reduction="none")
        self.assertTrue(torch.allclose(l_set, l_ce, atol=1e-5))

        # Multi-positive set: putting all mass on ANY member in A* drives loss -> 0
        multi_opt = torch.zeros(1, 4, dtype=torch.bool)
        multi_opt[0, 0] = True
        multi_opt[0, 1] = True
        sharp_on_1 = torch.tensor([[-20.0, 20.0, -20.0, -20.0]])
        self.assertLess(float(optimal_action_set_loss(sharp_on_1, multi_opt).item()), 1e-5)

    def test_parameter_counts_and_caps(self):
        sw = AdaptiveGeometryController(CONFIG_54K).parameter_summary(True)
        fx = AdaptiveGeometryController(CONFIG_54K_FIXED).parameter_summary(False)
        nc = AdaptiveGeometryController(CONFIG_54K_NOCOMM).parameter_summary(True)
        big = AdaptiveGeometryController(CONFIG_116K).parameter_summary(True)
        self.assertTrue(50000 <= sw["active_params"] <= 60000)
        self.assertGreaterEqual(fx["active_params"], sw["active_params"])
        self.assertLessEqual(big["total_params"], 128000)
        self.assertLess(nc["active_params"], sw["active_params"])


class TestTrainingRolloutAndInterventions(unittest.TestCase):
    def test_training_exclusively_16k(self):
        mem32 = SharedPhysicalMemory(32768, 107)
        with self.assertRaises(ValueError):
            train_controller(mem32, "recurrent_switch", seed=107, num_roots=16, epochs=1)

    def test_train_rollout_physical_accounting_and_interventions(self):
        torch.set_num_threads(2)
        mem16 = SharedPhysicalMemory(16384, 10107)
        mdl, stats = train_controller(
            mem16, "recurrent_switch", seed=107, num_roots=64, epochs=2, batch_size=32
        )
        self.assertGreater(stats["teacher_verified_rate"], 0.85)

        mem32 = SharedPhysicalMemory(32768, 2032768)
        roots = []
        for fam in FAMILIES:
            roots.extend(sample_family_roots(mem32, fam, 2, 331))

        for arm in ("det_row", "det_router", "recurrent_switch", "hier_adaptive", "rand_routing", "oracle_proj"):
            use_mdl = None if arm.startswith("det_") else mdl
            recs, _ = rollout_batch(mem32, roots, arm=arm, model=use_mdl, budget=8, corruption=0.0, rng_seed=7)
            self.assertEqual(len(recs), len(roots))
            for r in recs:
                self.assertLessEqual(r["probes"], 8)
                self.assertEqual(r["raw_slot_reads"], PAGE * r["probes"])
                self.assertEqual(r["unique_physical_slots"] + r["overlap_slot_reads"], r["raw_slot_reads"])
                self.assertEqual(
                    r["charged_reads"],
                    r["raw_slot_reads"] + r["anchor_lookups"] + SWITCH_OVERHEAD_COST * r["switches"],
                )

        # Causal intervention: freeze_router forces 0 switches
        recs_frz, _ = rollout_batch(
            mem32,
            roots,
            arm="recurrent_switch",
            model=mdl,
            budget=8,
            corruption=0.0,
            intervention=InterventionSpec(name="freeze_router", freeze_router=True),
            rng_seed=7,
        )
        self.assertTrue(all(r["switches"] == 0 for r in recs_frz))

        # Bootstrap CI sanity check
        boot = paired_bootstrap_ci(np.ones(32), np.zeros(32), n_resamples=200, seed=1)
        self.assertAlmostEqual(boot["point"], 1.0)
        self.assertAlmostEqual(boot["ci95_low"], 1.0)


class TestNativeStoryworldBridge(unittest.TestCase):
    def test_witness_withheld_and_native_verification(self):
        if not DEFAULT_NATIVE_WORLD.is_file():
            self.skipTest("MCP-05 archived storyworld JSON not present")
        full = json.loads(DEFAULT_NATIVE_WORLD.read_text(encoding="utf-8"))
        stripped = strip_archived_witnesses(full)
        self.assertNotIn("exact", stripped)
        self.assertNotIn("witnesses", stripped.get("balance", {}))

        # Passing unstripped world MUST fail loudly
        with self.assertRaises(ValueError):
            discover_route_for_ending(full, "end_bell_as_question", "det_router", None, seed=107)

        # Archived replay passes native swap checks
        rep = run_archived_witness_replay(DEFAULT_NATIVE_WORLD, "det_router", None, seed=107, budget=16)
        self.assertEqual(rep["track"], "native_witness_replay")
        self.assertTrue(all(e["swap_control_verified"] for e in rep["endings"]))

        # Independent witness-withheld discovery runs and verifies cleanly
        res = discover_route_for_ending(stripped, "end_bell_as_question", "det_router", None, seed=107, retrieval_budget=8)
        ver = verify_discovered_route(
            full["scenes"],
            full["math"],
            res["route"],
            "end_bell_as_question",
            archived_witness=full["exact"]["secret_gates"]["end_bell_as_question"]["witness"],
        )
        self.assertTrue(ver["legal_transitions"])


if __name__ == "__main__":
    unittest.main()
