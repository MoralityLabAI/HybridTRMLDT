import math
import unittest

from .hilbert_storyworld import GATES, START, apply, counterfactual, exact_routes, order_changes_operator, probability


class HilbertStoryworldTests(unittest.TestCase):
    def test_primitives_preserve_norm_and_inner_products(self):
        left = apply(START, ["H0", "H1", "S1"])
        right = apply(START, ["X0", "H0", "H1", "S1"])
        for name in GATES:
            clean, orthogonal = apply(left, [name]), apply(right, [name])
            self.assertAlmostEqual(sum(abs(value) ** 2 for value in clean), 1)
            self.assertAlmostEqual(sum(a.conjugate() * b for a, b in zip(clean, orthogonal)), 0)

    def test_complex_phase_changes_measurement_and_order(self):
        y_plus = apply(START, ["H0", "S0"])
        self.assertAlmostEqual(probability(y_plus, "Y0+"), 1)
        self.assertAlmostEqual(probability(y_plus, "Y0-"), 0)
        self.assertAlmostEqual(probability(y_plus, "Z0+"), .5)
        self.assertTrue(order_changes_operator(["H0"], ["S0"]))
        self.assertFalse(order_changes_operator(["X0"], ["Z0"]))

    def test_controlled_not_entangles_the_two_factors(self):
        state = apply(START, ["H0", "CX01"])
        self.assertAlmostEqual(state[0], math.sqrt(.5))
        self.assertAlmostEqual(state[3], math.sqrt(.5))
        self.assertAlmostEqual(probability(state, "01") + probability(state, "10"), 0)

    def test_ending_counterfactual_keeps_clues(self):
        model = {"operators": {"mix": ["H0"], "phase": ["S0"]},
                 "secrets": {"end_echo": {"requires": ["c0", "c1"], "projectors": [["Y0+", .9, 1]]}}}
        scenes = [{"options": [["mix", "reply", "mix", "c0", 1]]},
                  {"options": [["phase", "reply", "phase", "c1", 1]]}]
        result = counterfactual(scenes, model, [[0, 0], [1, 0]], "end_echo")
        self.assertEqual(result["clues"], ["c0", "c1"])
        original, swapped = result["probabilities"]["Y0+"]
        self.assertAlmostEqual(original, 1)
        self.assertAlmostEqual(swapped, .5)

    def test_exact_routes_merges_duplicate_ways_to_the_same_final_ray(self):
        model = {"operators": {"mix": ["H0"], "phase": ["S0"], "idle": ["I"]},
                 "secrets": {"end_echo": {"requires": ["c0", "c1"], "projectors": [["Y0+", .9, 1]]}}}
        scenes = [{"options": [["mix", "reply", "mix", "c0", 1],
                              ["stay", "reply", "idle", "c0", 1], ["wait", "reply", "idle", "c0", 1]]},
                  {"options": [["phase", "reply", "phase", "c1", 1]] * 3}]
        output = exact_routes(scenes, model)
        self.assertEqual(output["states_per_scene"], [1, 2, 2])
        gate = output["secret_gates"]["end_echo"]
        self.assertAlmostEqual(gate["eligible_probability"], 1 / 3)
        self.assertAlmostEqual(gate["clue_probability"], 1)
        self.assertIsNotNone(gate["order_witness"])
