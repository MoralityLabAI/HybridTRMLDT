"""Offline algebraic audit; standard library only."""
import math
import unittest

from research_gym.analysis.stalk_source_specificity_math import (
    certificate, eigvalsh_small, heat_trace, kronecker_sum,
    normalize_max, pairwise_spectrum,
)


class SourceSpecificityMathTests(unittest.TestCase):
    def assert_close_arrays(self, x, y):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b, places=9)

    def test_diagonal(self):
        cert = certificate([[0., 0.], [0., 2.]], [[0., 0.], [0., 3.]])
        self.assert_close_arrays(cert["explicit_spectrum"], [0., 2., 3., 5.])
        self.assertEqual(len(cert["heat"]), 4)

    def test_symmetric_psd_with_off_diagonals(self):
        # Two weighted 2-vertex graph Laplacians, both PSD.
        c = [[2., -2.], [-2., 2.]]
        q = [[3., -3.], [-3., 3.]]
        cert = certificate(c, q)
        self.assert_close_arrays(cert["explicit_spectrum"], [0., 4., 6., 10.])

    def test_zero_spectrum(self):
        z = [[0., 0.], [0., 0.]]
        cert = certificate(z, z)
        self.assertEqual(normalize_max(cert["pairwise_spectrum"]), [0.] * 4)
        for row in cert["heat"]:
            self.assertAlmostEqual(row["normalized"], 1.)

    def test_isospectral_graph_relabeling(self):
        path = [[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]]
        # Vertex permutation changes labeled source, preserves full spectrum.
        order = [1, 0, 2]
        relabeled = [[path[i][j] for j in order] for i in order]
        self.assertNotEqual(path, relabeled)
        c = [[0., 0.], [0., 2.]]
        self.assert_close_arrays(pairwise_spectrum(c, path),
                                 pairwise_spectrum(c, relabeled))
        self.assert_close_arrays(
            normalize_max(pairwise_spectrum(c, path)),
            normalize_max(pairwise_spectrum(c, relabeled)))
        certificate(c, path)
        certificate(c, relabeled)

    def test_normalized_counterexample(self):
        c = [0., 1.]
        q = [0., 3.]
        actual = heat_trace([x + y for x in c for y in q], 1., normalized=True)
        wrong = heat_trace(c, 1., normalized=True) * heat_trace(q, 1., normalized=True)
        expected = (1 + math.exp(-.25) + math.exp(-.75) + math.exp(-1.)) / 4
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(abs(actual - wrong), .01)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            eigvalsh_small([[0., 1.], [0., 0.]])
        with self.assertRaises(ValueError):
            normalize_max([-1., 1.])
        with self.assertRaises(ValueError):
            kronecker_sum([[1., 2.]], [[1.]])
        with self.assertRaises(ValueError):
            heat_trace([0.], -1.)


if __name__ == "__main__":
    unittest.main()
