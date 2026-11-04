import unittest
from research_gym.fixed_weight_conditioning import State, authorize, run_sequence, transition

class FixedWeightConditioningTests(unittest.TestCase):
    def test_same_cue_before_and_after_pairing(self):
        cue = (True, False)
        before, zero = transition(State(), *cue)
        _, trained = transition(State(), True, True)
        after, retained = transition(trained, *cue)
        self.assertFalse(before)
        self.assertTrue(after)
        self.assertFalse(zero.association)
        self.assertTrue(retained.association)

    def test_reset_eliminates_association(self):
        _, learned = run_sequence([(True, True), (True, False)])
        self.assertTrue(learned.association)
        after_reset, _ = transition(State(), True, False)
        self.assertFalse(after_reset)

    def test_unpaired_inputs_do_not_condition(self):
        results, state = run_sequence([(True, False), (False, True), (True, False)])
        self.assertEqual(results, [False, True, False])
        self.assertFalse(state.association)

    def test_gate_enforces_allowlist(self):
        for allowed in (set(), {"read"}, {"write"}, {"read", "write"}):
            for proposal in ("read", "write", "delete", ""):
                granted = authorize(allowed, proposal)
                self.assertTrue(granted is None or granted in allowed)
        self.assertIsNone(authorize({"read"}, "write"))

if __name__ == "__main__":
    unittest.main()
