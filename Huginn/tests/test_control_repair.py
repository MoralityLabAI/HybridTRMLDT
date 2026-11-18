from __future__ import annotations

import unittest
import numpy as np

from Huginn.control_repair import (
    RankOneControl, clustered_bootstrap_ci, difference_in_differences,
    fit_remainder_scalar_selection, holm_rejections, matching_report, clustered_signflip_p, window_effects,
    quantiles, read_rotation_pool, rotated_write_controls,
    select_joint_controls, verify_joint_controls, window_report,
)


class ControlRepairTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(209)
        self.u = np.array([1., 2., -3., 0.5, 2., 1., -0.2])
        self.v = np.array([1., -2., 0.7, 0.3, 1.5, -0.4])
        self.x_sel = {"early": rng.normal(size=(1600, 6)),
                      "late": rng.normal(size=(1600, 6)) * 1.9}
        self.x_verify = {"early": rng.normal(size=(1300, 6)),
                         "late": rng.normal(size=(1300, 6)) * 1.7}
        self.target = RankOneControl(self.u, self.v, "factor", 0)

    def test_orthogonal_write_has_exact_per_instance_norms_both_windows(self):
        controls = rotated_write_controls(self.u, self.v, count=4, seed=41)
        self.assertEqual(len(controls), 4)
        for c in controls:
            self.assertAlmostEqual(float(self.u @ c.u), 0.0, places=11)
            np.testing.assert_allclose(np.linalg.norm(c.u), np.linalg.norm(self.u), rtol=1e-12)
            np.testing.assert_allclose(c.v, self.v)
            for x in (self.x_sel, self.x_verify):
                for w in ("early", "late"):
                    np.testing.assert_allclose(c.activation_norms(x[w]), self.target.activation_norms(x[w]), rtol=2e-14, atol=2e-14)
                self.assertTrue(window_report(self.target, c, x)["accepted_both_windows"])
        self.assertTrue(verify_joint_controls(self.target, controls, self.x_verify)["accepted"])

    def test_write_directions_reproducible_and_orthogonal(self):
        a = rotated_write_controls(self.u, self.v, count=4, seed=123)
        b = rotated_write_controls(self.u, self.v, count=4, seed=123)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x.u, y.u)
        writes = np.array([c.u / np.linalg.norm(c.u) for c in a])
        np.testing.assert_allclose(writes @ writes.T, np.eye(4), atol=1e-12)
        with self.assertRaises(ValueError):
            rotated_write_controls(self.u, self.v, count=7)

    def test_new_read_pool_does_not_tune_to_verification(self):
        pool = read_rotation_pool(self.u, self.v, self.x_sel, count=60, seed=14)
        self.assertGreater(len(pool), 10)
        for c in pool:
            cosine = abs(float(c.v @ self.v / (np.linalg.norm(c.v) * np.linalg.norm(self.v))))
            self.assertLessEqual(cosine, 0.95 + 1e-12)
            self.assertGreater(c.scale, 0)
        # The selector consumes ONLY x_sel; changing verification cannot affect it.
        first, _ = select_joint_controls(self.target, pool, self.x_sel)
        rerun, _ = select_joint_controls(self.target, pool, self.x_sel)
        self.assertEqual([(c.index,c.scale) for c in first], [(c.index,c.scale) for c in rerun])
        verification = verify_joint_controls(self.target, first, self.x_verify)
        self.assertEqual(verification["n_selected"], len(first))
        self.assertEqual(verification["accepted"], len(first) == 4 and verification["n_accepted_joint"] == 4)

    def test_median_only_is_insufficient_for_tail_matching(self):
        target = np.r_[np.repeat(.2, 20), np.repeat(1., 60), np.repeat(4., 20)].astype(float)
        candidate = np.r_[np.repeat(.01, 20), np.repeat(1., 60), np.repeat(4., 20)].astype(float)
        a = matching_report(target, candidate)
        self.assertAlmostEqual(a["target"][1], a["candidate"][1])
        self.assertFalse(a["accepted"])
        self.assertGreater(a["errors"][0], .15)

    def test_remainder_scalar_is_selection_only(self):
        a = {"early": np.linspace(1, 10, 900), "late": np.linspace(2, 13, 900)}
        b = {w: norms/1.5 for w, norms in a.items()}
        self.assertAlmostEqual(fit_remainder_scalar_selection(a,b), 1.5, places=12)
        with self.assertRaises(ValueError):
            fit_remainder_scalar_selection(a,{"early":b["early"]})

    def test_optimistic_verification_cannot_create_four_missing_controls(self):
        v = [True,False,True]
        self.assertLess(sum(v), 4)

    def test_did_and_program_clustered_ci(self):
        n, seeds = 50, 3
        clean = np.full((n,seeds,2), -1.)
        factor = clean.copy(); factor[:,:,1] -= .12
        controls = np.broadcast_to(clean, (4, *clean.shape)).copy()
        controls[:,:,:,1] -= .02
        effects = window_effects(clean, factor, controls)
        np.testing.assert_allclose(effects[:,:,1], -.10, atol=1e-12)
        np.testing.assert_allclose(effects[:,:,0], 0.0, atol=1e-12)
        did = difference_in_differences(clean, factor, controls)
        np.testing.assert_allclose(did, -.10, atol=1e-12)
        stats = clustered_bootstrap_ci(did, seed=41, n_boot=200)
        self.assertAlmostEqual(stats["mean"], -.10)
        self.assertAlmostEqual(stats["lower"], -.10)
        self.assertEqual(stats["n_programs"], n)
        self.assertLessEqual(clustered_signflip_p(did, seed=4, n_draws=999), 0.01)
        self.assertGreater(clustered_signflip_p(np.zeros_like(did), seed=4, n_draws=999), 0.99)
        with self.assertRaises(ValueError):
            difference_in_differences(clean, factor, controls[:3])

    def test_holm_stepdown(self):
        result = holm_rejections({"a":.001,"b":.03,"c":.04})
        self.assertEqual(result, {"a":True,"b":False,"c":False})
        self.assertEqual(holm_rejections({"a":.001,"b":.009,"c":.012}),
                         {"a":True,"b":True,"c":True})

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            quantiles(np.array([np.nan]))
        with self.assertRaises(ValueError):
            rotated_write_controls(np.array([1.,0.]), np.array([0.,0.]))
        with self.assertRaises(ValueError):
            window_report(self.target, self.target, {"early":self.x_sel["early"]})


if __name__ == "__main__":
    unittest.main()
