import itertools
import unittest

from research_gym.benchmarks.molt_hybrid.prefix_memory import (
    bound, invalidate, synchronous, topological, validate,
)


class PrefixMemoryTest(unittest.TestCase):
    def test_all_small_graphs(self):
        for n in range(1, 5):
            edges = [(i, j) for j in range(n) for i in range(j)]
            for bits in itertools.product((0, 1), repeat=len(edges)):
                parents = [[] for _ in range(n)]
                for (i, j), bit in zip(edges, bits, strict=True):
                    if bit:
                        parents[j].append(i)
                x = tuple(range(n))
                exact = topological(x, parents)
                for initial in itertools.product((0, 1), repeat=n):
                    self.assertEqual(synchronous(x, parents, initial, bound(parents)), exact)
                    self.assertEqual(synchronous(x, parents, initial, bound(parents) + 1), exact)

    def test_invalid_edges(self):
        for p in ([[0]], [[], [1]], [[], [-1]], [[], [0, 0]]):
            with self.assertRaises(ValueError):
                validate(p)

    def test_invalidate_only_descendants(self):
        self.assertEqual(invalidate([[], [0], [], [1], [2]], {0}), {0, 1, 3})


if __name__ == "__main__":
    unittest.main()
