"""Preflight and actual-routed-activation contract tests."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from Kimilings.capture import audit_capture
from Kimilings.index import build_index, stable_split
from Kimilings.slices import SLICE_COMMIT, SLICE_REPO, verify_slice_receipt


def fake_receipt():
    return {
        "hf_repo": SLICE_REPO, "hf_commit": SLICE_COMMIT, "tensor_count": 8832,
        "bundles": [
            {"expert_index": i, "path": f"index_{i}.bin",
             "size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            for i in range(16)
        ],
    }


class PreflightTests(unittest.TestCase):
    def test_local_bundle_bytes_and_missing_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(16):
                (root / f"index_{i}.bin").write_bytes(b"abc")
            r = fake_receipt()
            self.assertEqual(verify_slice_receipt(r, root)["status"],
                             "local_sha256_and_size_verified")
            self.assertEqual(verify_slice_receipt(r)["status"],
                             "metadata_inspected_bytes_unverified")
            (root / "index_5.bin").write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "Bundle bytes mismatch"):
                verify_slice_receipt(r, root)
            r["bundles"][0]["path"] = "../escape"
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_slice_receipt(r, root)

    def test_captures_fail_closed_without_full_model_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "training"
            train.mkdir()
            (train / "a.json").write_text('{"train":[],"test":[]}')
            m = build_index(train)
            split = stable_split("arc_agi_2", "a")
            sha64 = hashlib.sha256(b"capture").hexdigest()
            cap = {
                "task_id": "arc_agi_2/a", "split": split,
                "activation_source": "synthetic",
                "source_model_revision": "a" * 40,
                "activation_sha256": sha64, "router_receipt": sha64,
                "reference_forward_receipt": sha64,
                "layer": 1, "expert_index": 0, "token_count": 123,
                "capture_id": "run0",
            }
            with self.assertRaisesRegex(ValueError, "actual routed"):
                audit_capture(m, [cap])
            cap["activation_source"] = "routed_kimi"
            cap["split"] = "final"
            with self.assertRaisesRegex(ValueError, "split-mismatched"):
                audit_capture(m, [cap])
            cap["split"] = split
            report = audit_capture(m, [cap])
            self.assertEqual(report["ready_sites"], 0)
            self.assertEqual(report["candidate_sites"][0]["token_counts_by_family"],
                             {"arc_agi_2": 123})


if __name__ == "__main__":
    unittest.main()
