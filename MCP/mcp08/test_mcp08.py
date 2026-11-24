"""MCP-08A CPU unit test suite: fixture generation, strict witness isolation, all 4 env verifiers,
both full storyworld verifiers, eta metric, paired bootstrap CI, and preregistered gates."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from MCP.mcp08.fixtures import ENVIRONMENTS, assert_zero_witness_leakage, generate_suite
from MCP.mcp08.verifiers import (
    compute_eta,
    evaluate_gates,
    paired_bootstrap_ci,
    verify_clockmakers_republic,
    verify_garden_of_unmeasured_things,
    verify_submission,
)


class TestMCP08A(unittest.TestCase):
    def test_eval_suite_counts_disjointness_and_zero_leakage(self) -> None:
        dev_roots, dev_wit = generate_suite("dev", roots_per_env=2)
        pilot_roots, pilot_wit = generate_suite("pilot", roots_per_env=2)
        eval_roots, eval_wit = generate_suite("eval", roots_per_env=8)

        self.assertEqual(len(dev_roots), 8)
        self.assertEqual(len(pilot_roots), 8)
        self.assertEqual(len(eval_roots), 32)

        # Verify 2 roots/env in wide+oracle subset = 8 total
        subset = [r for r in eval_roots if r["in_wide_oracle_subset"]]
        self.assertEqual(len(subset), 8)
        for env in ENVIRONMENTS:
            self.assertEqual(sum(1 for r in subset if r["env"] == env), 2)

        # Verify strict seed disjointness across dev/pilot/eval
        all_seeds = [r["seed"] for r in dev_roots + pilot_roots + eval_roots]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

        # Verify zero witness leakage into searchable federation corpora
        assert_zero_witness_leakage(dev_roots, dev_wit)
        assert_zero_witness_leakage(pilot_roots, pilot_wit)
        assert_zero_witness_leakage(eval_roots, eval_wit)

    def test_all_32_oracle_submissions_pass_independent_verifiers(self) -> None:
        eval_roots, eval_wit = generate_suite("eval", roots_per_env=8)
        for root in eval_roots:
            wit = eval_wit[root["root_id"]]
            res = verify_submission(root, wit, wit["oracle_submission"])
            self.assertTrue(res["verified"], f"Oracle failed on {root['root_id']}: {res}")
            self.assertEqual(res["score"], 1.0)
            self.assertEqual(res["evidence_recall"], 1.0)
            self.assertEqual(res["retrieval_precision"], 1.0)

    def test_adversarial_failures_rejected_across_all_4_envs(self) -> None:
        dev_roots, dev_wit = generate_suite("dev", roots_per_env=1)
        by_env = {r["env"]: (r, dev_wit[r["root_id"]]) for r in dev_roots}

        # 1. Flooded City: citing obsolete lore or entering submerged district must fail
        fc_root, fc_wit = by_env["flooded_city"]
        bad_fc = json.loads(json.dumps(fc_wit["oracle_submission"]))
        bad_fc["cited_docs"].append(fc_wit["forbidden_stale_ids"][0])
        bad_fc["scenes"][1]["district"] = fc_wit["flooded_districts"][0]
        res_fc = verify_submission(fc_root, fc_wit, bad_fc)
        self.assertFalse(res_fc["verified"])
        self.assertFalse(res_fc["checks"]["no_submerged_entry"])
        self.assertFalse(res_fc["checks"]["rejected_stale_lore"])

        # 2. Quantum Labyrinth: reversing noncommuting ops must fail target probability check
        ql_root, ql_wit = by_env["quantum_labyrinth"]
        bad_ql = {"cited_docs": ql_wit["required_evidence_ids"], "operators": ["Z0", "Z0"]}
        res_ql = verify_submission(ql_root, ql_wit, bad_ql)
        self.assertFalse(res_ql["verified"])
        self.assertFalse(res_ql["checks"]["terminal_probabilities_match"])

        # 3. Arena Architect: original buggy source must fail dynamic Node verifier
        aa_root, aa_wit = by_env["arena_architect"]
        buggy_doc = next(d for d in aa_root["federation_corpus"] if d["kind"] == "source")
        res_aa = verify_submission(aa_root, aa_wit, {"cited_docs": [], "code": buggy_doc["text"]})
        self.assertFalse(res_aa["verified"])
        self.assertFalse(res_aa["checks"]["bounds_and_walls_ok"])
        self.assertFalse(res_aa["checks"]["canvas_rendering_ok"])

        # 4. Broken Archive: returning stale symbol & stale doc must fail
        ba_root, ba_wit = by_env["broken_archive"]
        bad_ba = {
            "cited_docs": ba_wit["stale_doc_ids"],
            "canonical_symbol": ba_root["legacy_symbol"],
            "canonical_value": -1,
            "canonical_commit": "0000000000",
            "dependency_chain": [ba_root["legacy_symbol"]],
        }
        res_ba = verify_submission(ba_root, ba_wit, bad_ba)
        self.assertFalse(res_ba["verified"])
        self.assertFalse(res_ba["checks"]["exact_canonical_symbol"])
        self.assertFalse(res_ba["checks"]["valid_citations_only"])

    def test_storyworld_verifiers(self) -> None:
        good_clock = {
            "districts": [f"d{i}" for i in range(8)],
            "start_scene": "s0",
            "endings": ["end_guild", "end_council"],
            "scenes": [
                {"id": "s0", "choices": [{"target": "s1", "faction_delta": {"guild": 1}, "resource_delta": {"brass": -1}},
                                         {"target": "s2", "faction_delta": {"council": 1}, "resource_delta": {"brass": 0}}]},
                {"id": "s1", "requires_prior_flag": "guild_pact", "choices": [{"target": "end_guild"}]},
                {"id": "s2", "choices": [{"target": "end_council"}]},
                {"id": "end_guild", "choices": []},
                {"id": "end_council", "choices": []},
            ],
        }
        self.assertTrue(verify_clockmakers_republic(good_clock)["verified"])

        # Add unreachable scene -> must fail
        bad_clock = json.loads(json.dumps(good_clock))
        bad_clock["scenes"].append({"id": "orphan", "choices": []})
        self.assertFalse(verify_clockmakers_republic(bad_clock)["verified"])

        good_garden = {
            "routes": [
                {"operators": ["H0", "CNOT", "X1"], "ending": "entangled_bloom"},
                {"operators": ["X0", "H0", "CNOT"], "ending": "phase_shadow"},
            ],
            "counterfactual_pair": {"seq_a": ["H0", "CNOT"], "seq_b": ["CNOT", "H0"]},
        }
        self.assertTrue(verify_garden_of_unmeasured_things(good_garden)["verified"])

        bad_garden = json.loads(json.dumps(good_garden))
        bad_garden["counterfactual_pair"] = {"seq_a": ["Z0", "T1"], "seq_b": ["T1", "Z0"]}
        self.assertFalse(verify_garden_of_unmeasured_things(bad_garden)["verified"])

    def test_eta_bootstrap_and_gates(self) -> None:
        eps = [{"verified": True, "wall_s": 30.0}, {"verified": False, "wall_s": 60.0}, {"verified": True, "wall_s": 90.0}]
        self.assertAlmostEqual(compute_eta(eps), 40.0, places=3)  # 2 verified in 180s = 40/GPU-hr
        ci = paired_bootstrap_ci([1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1])
        self.assertGreater(ci["mean_diff"], 0.0)
        g = evaluate_gates({
            "arm_a_completion_pct": 55.0,
            "arm_b_completion_pct": 62.5,
            "arm_c_completion_pct": 71.875,
            "time_reduction_pct": 31.4,
            "pure_gpu_cpu_offload_tensors": 0,
            "pure_gpu_paging_events": 0,
            "fresh_worlds_verified": True,
            "bonsai_delta_eta": 14.2,
            "inky_delta_eta": 6.8,
        })
        self.assertTrue(g["all_passed"])

    def test_manifest_matches_episode_math(self) -> None:
        m = json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8"))
        # 2 models * (3 arms * 32 roots + 2 arms * 8 roots) = 2 * (96 + 16) = 224 episodes
        self.assertEqual(m["counts"]["total_registered_episodes"], 224)


if __name__ == "__main__":
    unittest.main()
