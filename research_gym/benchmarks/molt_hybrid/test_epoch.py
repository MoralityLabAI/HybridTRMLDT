import itertools
import unittest

from research_gym.core.hybrid import CandidateState, LatticeProposal
from research_gym.core.typed_soundness import SoundnessType

from research_gym.benchmarks.molt_hybrid.epoch import FrameMachine, Premise, ScalarBank
from research_gym.benchmarks.molt_hybrid.prefix_memory import topological


def state(*values):
    return CandidateState({"choice": frozenset(values)})


def claim(*values):
    return LatticeProposal(state(*values), SoundnessType.ENV_SOUND_DEAD)


class EpochTests(unittest.TestCase):
    def test_strict_membrane_and_frame_reset(self):
        machine = FrameMachine(state("a", "b", "c"), (Premise("s0", 1, "choice", "b"),), "root")
        initial = machine.snapshot()
        self.assertFalse(machine.apply(initial.key, claim("a")).accepted)
        self.assertTrue(machine.apply(initial.key, claim("a", "c")).accepted)
        after = machine.snapshot()
        self.assertEqual(after.epoch, 1)
        self.assertFalse(machine.apply(initial.key, claim("a", "c")).accepted)
        self.assertFalse(machine.apply(after.key, claim("a", "b", "c")).accepted)
        fresh = machine.replace_premise(Premise("s0", 2, "choice", "c"))
        self.assertEqual(fresh.state, state("a", "b"))
        self.assertEqual((fresh.frame, fresh.epoch), (1, 0))
        self.assertFalse(machine.apply(after.key, claim("a")).accepted)
        self.assertEqual(machine.history[1].state, state("a", "c"))
        with self.assertRaises(ValueError):
            machine.replace_premise(Premise("s0", 2, "choice", "a"))

    def test_local_and_global_invalidation(self):
        bank = ScalarBank()
        p = ((), (0,), (), (1,), (2,))
        first = bank.evaluate((1, 2, 3, 4, 5), p, "frame0")
        self.assertEqual(first.recomputed, (0, 1, 2, 3, 4))
        second = bank.evaluate((2, 2, 3, 4, 5), p, "frame0")
        self.assertEqual(second.recomputed, (0, 1, 3))
        self.assertEqual(second.reused, (2, 4))
        self.assertEqual(second.values, topological((2, 2, 3, 4, 5), p))
        changed = bank.evaluate((2, 2, 3, 4, 5), p, "frame1")
        self.assertEqual(changed.reused, ())

    def test_exhaustive_graph_edits(self):
        n = 4
        edges = [(i, j) for j in range(n) for i in range(j)]
        graphs = []
        for bits in itertools.product((0, 1), repeat=len(edges)):
            p = [[] for _ in range(n)]
            for (i, j), bit in zip(edges, bits, strict=True):
                if bit:
                    p[j].append(i)
            graphs.append(tuple(tuple(row) for row in p))
        x = (1, 2, 3, 4)
        for p in graphs:
            for q in graphs:
                bank = ScalarBank()
                bank.evaluate(x, p, "same")
                self.assertEqual(bank.evaluate(x, q, "same").values, topological(x, q))


if __name__ == "__main__":
    unittest.main()
