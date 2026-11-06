"""CPU-only contract tests; fake tasks and metrics carry no Kimi claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Kimilings.cli import main
from Kimilings.index import (
    ARC_REVISION, audit_index, build_index, sha256_file, stable_split
)
from Kimilings.router import route
from Kimilings.score import score_arc, score_logic_verdicts
from Kimilings.vpd import causal_transfer, summarize_vpd


def task_json(output):
    return {"train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": output}]}


def write_json(path: Path, body: dict):
    path.write_text(json.dumps(body), encoding="utf-8")


class IndexAndArcTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "training"
        self.final = self.root / "evaluation"
        self.train.mkdir()
        self.final.mkdir()
        write_json(self.train / "task_a.json", task_json([[4]]))
        write_json(self.final / "task_z.json", task_json([[7]]))
        self.manifest = build_index(self.train, self.final)

    def test_index_seals_final_and_has_stable_splits(self):
        self.assertEqual(self.manifest["arc_revision"], ARC_REVISION)
        self.assertEqual(self.manifest["sealed_final_count"], 1)
        audit_index(self.manifest)
        self.assertTrue(all("output" not in r and "prompt" not in r
                            for r in self.manifest["records"]))
        self.assertEqual(stable_split("arc_agi_2", "task_a"),
                         stable_split("arc_agi_2", "task_a"))

    def test_duplicate_final_id_and_checksum_leakage_rejected(self):
        write_json(self.final / "task_a.json", task_json([[8]]))
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            build_index(self.train, self.final)
        (self.final / "task_a.json").unlink()
        write_json(self.final / "task_z.json", task_json([[4]]))
        with self.assertRaisesRegex(ValueError, "Identical task"):
            build_index(self.train, self.final)

    def test_arc_pass_at_two_and_missing_counts_wrong(self):
        split = stable_split("arc_agi_2", "task_a")
        predictions = self.root / "preds.jsonl"
        predictions.write_text("\n".join(json.dumps(x) for x in [
            {"system": "hybrid", "task_id": "arc_agi_2/task_a", "seed": 11,
             "test_index": 0, "attempt": 1, "grid": [[0]]},
            {"system": "hybrid", "task_id": "arc_agi_2/task_a", "seed": 11,
             "test_index": 0, "attempt": 2, "grid": [[4]]},
            {"system": "trm", "task_id": "arc_agi_2/task_a", "seed": 11,
             "test_index": 0, "attempt": 1, "grid": [[0]]},
        ]) + "\n", encoding="utf-8")
        report = score_arc(self.manifest, self.train, predictions, split=split)
        scores = {r["system"]: r["solved"] for r in report["systems"]}
        self.assertEqual(scores, {"hybrid": 1, "trm": 0})
        self.assertEqual(report["benchmark"], "official_arc_agi_2_training_partition")
        with self.assertRaises(PermissionError):
            score_arc(self.manifest, self.final, predictions, split="final")

    def test_final_ledger_is_one_shot(self):
        predictions = self.root / "final_predictions.jsonl"
        predictions.write_text(json.dumps({
            "system": "trm", "task_id": "arc_agi_2/task_z", "seed": 1,
            "test_index": 0, "attempt": 1, "grid": [[7]]
        }) + "\n", encoding="utf-8")
        ledger = self.root / "FINAL.lock"
        result = score_arc(self.manifest, self.final, predictions, split="final",
                           unlock_final=True, ledger_path=ledger,
                           strict_final_count=False)
        self.assertEqual(result["systems"][0]["solved"], 1)
        self.assertTrue(ledger.exists())
        with self.assertRaises(FileExistsError):
            score_arc(self.manifest, self.final, predictions, split="final",
                      unlock_final=True, ledger_path=ledger,
                      strict_final_count=False)

    def test_logic_metadata_no_prompt_and_verifier_cost(self):
        meta = self.root / "logic.jsonl"
        digest = hashlib.sha256(b"distinct logic problem").hexdigest()
        meta.write_text(json.dumps({
            "task_id": "a", "group_id": "series-a", "content_sha256": digest,
        }) + "\n", encoding="utf-8")
        m = build_index(self.train, self.final, meta)
        logic = [r for r in m["records"] if r["task_id"] == "logic/a"][0]
        verdicts = [
            {"task_id": "logic/a", "system": name, "seed": 1,
             "correct": good, "verifier_id": "logic-env",
             "verifier_revision": "0.1.2", "receipt_sha256": digest,
             "cost": 2.0, "cost_unit": "inference_ms"}
            for name, good in (("trm", False), ("hybrid", True))
        ]
        rep = score_logic_verdicts(m, verdicts, split=logic["split"])
        self.assertEqual(rep["paired"][0]["paired_n"], 1)
        self.assertEqual(rep["paired"][0]["accuracy_delta_a_minus_b"], 1.0)
        meta.write_text(json.dumps({
            "task_id": "a", "group_id": "series-a", "content_sha256": digest,
            "prompt": "leaky!"
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IDs, family and hashes"):
            build_index(self.train, self.final, meta)

    def test_cli_index_and_audit(self):
        manifest_path = self.root / "manifest.json"
        out_path = self.root / "audit.json"
        self.assertEqual(main(["index", "--arc-training", str(self.train),
                               "--arc-evaluation", str(self.final),
                               "--out", str(manifest_path)]), 0)
        self.assertEqual(main(["audit", "--manifest", str(manifest_path),
                               "--out", str(out_path)]), 0)
        self.assertEqual(json.loads(out_path.read_text())["task_count"], 2)


class RouterAndVpdTests(unittest.TestCase):
    @staticmethod
    def example_route():
        return {
            "task_id": "logic/a", "candidate_indices": [0, 1, 2, 3],
            "bank_size": 2, "top_k_per_token": 2,
            "trm_scores": {"0": 1.0, "1": 4.0, "2": 3.0, "3": 0.0},
            "ldt_allowed": [0, 2],
            "ldt_verification": "claimed",
        }

    def test_unverified_ldt_cannot_hard_gate(self):
        r = self.example_route()
        self.assertEqual(route(r, "trm")["selected_indices"], [1, 2])
        self.assertEqual(route(r, "hybrid")["selected_indices"], [1, 2])
        self.assertEqual(route(r, "hybrid")["verification_mode"], "unverified_ldt_ignored")
        self.assertTrue(route(r, "ldt_verified")["abstained"])
        r["ldt_verification"], r["verifier_receipt"] = "independent", "trusted-sha"
        self.assertEqual(route(r, "hybrid")["selected_indices"], [1, 2])
        self.assertEqual(route(r, "hybrid", verify_ldt=lambda _: True)["selected_indices"], [2, 0])

    def test_vpd_router_requires_real_activation_and_calibration(self):
        r = self.example_route()
        r["vpd_scores"] = {"0": 0, "1": 0, "2": 9, "3": 0}
        with self.assertRaisesRegex(ValueError, "routed Kimi"):
            route(r, "hybrid_vpd")
        r["activation_source"], r["activation_receipt"] = "routed_kimi", "capture"
        with self.assertRaisesRegex(ValueError, "calibration"):
            route(r, "hybrid_vpd")
        r["calibration_receipt"] = "fit-on-calibration"
        r["vpd_weight"] = 1.0
        self.assertEqual(route(r, "hybrid_vpd")["selected_indices"], [2, 0])

    def test_strict_budget_blocks_underfilled_verified_bank(self):
        r = self.example_route()
        r["ldt_verification"], r["verifier_receipt"] = "independent", "receipt"
        r["ldt_allowed"] = [2]
        self.assertTrue(route(r, "hybrid", verify_ldt=lambda _: True)["abstained"])
        self.assertEqual(route(r, "hybrid", verify_ldt=lambda _: True)["verification_mode"],
                         "insufficient_verified_candidates")
        r["ldt_allowed"] = [0, 1, 2]
        r["top_k_per_token"] = 3
        with self.assertRaisesRegex(ValueError, "top_k_per_token"):
            route(r, "hybrid")

    def test_random_route_reproducible(self):
        r = self.example_route()
        self.assertEqual(route(r, "random_matched"), route(r, "random_matched"))

    def test_synthetic_vpd_fails_rescue(self):
        base = {
            "target_id": "layer1expert0", "task_family": "logic",
            "activation_source": "synthetic", "method": "vpd",
            "seed": 11, "step": 12, "mask_budget": 4,
            "active_count": 1, "total_components": 4,
            "parameter_relative_error": 0.001,
            "all_on_mse": 0.00007, "importance_mse": 0.01106,
            "adversarial_mse": 0.01084, "all_off_mse": 0.01107,
        }
        output = summarize_vpd([base])
        self.assertEqual(output["groups"][0]["status"], "instrument_only")
        self.assertLess(output["groups"][0]["importance_rescue_ratio"], 0.01)
        invalid = {**base, "activation_source": "routed_kimi"}
        with self.assertRaisesRegex(ValueError, "capture"):
            summarize_vpd([invalid])

    def test_transfer_needs_real_sources_and_matched_budget(self):
        row = {
            "component_family": "logic", "test_family": "arc_agi_2",
            "task_id": "arc_agi_2/a", "seed": 11, "mask_budget": 4,
            "loss_unmasked": 0.01, "loss_ablated": 0.21,
            "loss_control": 0.11, "control_method": "random_rank_matched",
            "activation_source": "synthetic", "matched_budget": True,
            "activation_receipt": "capture", "reference_forward_receipt": "forward",
        }
        with self.assertRaisesRegex(ValueError, "real routed"):
            causal_transfer([row])
        row["activation_source"] = "routed_kimi"
        report = causal_transfer([row])
        self.assertAlmostEqual(report["cells"][0]["differential_delta"], 0.1)
        row["matched_budget"] = False
        with self.assertRaisesRegex(ValueError, "budgets"):
            causal_transfer([row])


if __name__ == "__main__":
    unittest.main()
