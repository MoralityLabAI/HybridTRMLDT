"""Offline algebraic audit; standard library only."""
import math
import unittest

from research_gym.analysis.stalk_source_specificity_math import (
    certificate, eigvalsh_small, heat_trace, kronecker_sum,
    normalize_max, pairwise_spectrum, production_style_spectrum,
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

    def test_nonisomorphic_laplacian_cospectral_graphs(self):
        # Two connected six-vertex simple graphs with seven edges each.
        # Distinct degree multisets prove they are NOT isomorphic; both
        # Laplacians have spectrum {0, 3-sqrt(5), 2, 3, 3, 3+sqrt(5)}.
        edges_a = [(0, 2), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
        edges_b = [(0, 2), (0, 5), (1, 3), (1, 4), (2, 3), (3, 4), (3, 5)]

        def laplacian(edges):
            a = [[0.0] * 6 for _ in range(6)]
            for u, v in edges:
                a[u][u] += 1.0
                a[v][v] += 1.0
                a[u][v] -= 1.0
                a[v][u] -= 1.0
            return a

        a, b = laplacian(edges_a), laplacian(edges_b)
        self.assertNotEqual(sorted(a[i][i] for i in range(6)),
                            sorted(b[i][i] for i in range(6)))
        self.assert_close_arrays(eigvalsh_small(a), eigvalsh_small(b))
        c = [[1.0, -1.0], [-1.0, 1.0]]
        self.assert_close_arrays(production_style_spectrum(c, a),
                                 production_style_spectrum(c, b))
        certificate(c, a)
        certificate(c, b)

    def test_two_stage_production_normalization(self):
        c = [[0., 0.], [0., 1.]]
        q = [[0., 0.], [0., 3.]]
        actual = production_style_spectrum(c, q)
        self.assert_close_arrays(actual, [0., .5, .5, 1.])
        # A direct maximum normalization of unscaled factors is different.
        self.assert_close_arrays(normalize_max(pairwise_spectrum(c, q)),
                                 [0., .25, .75, 1.])
        self.assertAlmostEqual(heat_trace(actual, 4.),
                               heat_trace([0., 1.], 2.) ** 2)
        self.assert_close_arrays(production_style_spectrum(c, [[0., 0.], [0., 0.]]),
                                 [0., 0., 1., 1.])

    def test_strict_positive_maximum_even_when_tiny(self):
        self.assertEqual(normalize_max([0., 1e-20]), [0., 1.])
        c = [[0., 0.], [0., 1e-20]]
        z = [[0., 0.], [0., 0.]]
        cert = certificate(c, z)
        self.assertEqual(normalize_max(cert["explicit_spectrum"]), [0., 0., 1., 1.])
        self.assert_close_arrays(production_style_spectrum(c, z), [0., 0., 1., 1.])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            eigvalsh_small([[0., 1.], [0., 0.]])
        with self.assertRaises(ValueError):
            normalize_max([-1., 1.])
        with self.assertRaises(ValueError):
            kronecker_sum([[1., 2.]], [[1.]])
        with self.assertRaises(ValueError):
            heat_trace([0.], -1.)
        with self.assertRaises(ValueError):
            production_style_spectrum([[1.]], [[1.]], coupling=-1.)


if __name__ == "__main__":
    unittest.main()
