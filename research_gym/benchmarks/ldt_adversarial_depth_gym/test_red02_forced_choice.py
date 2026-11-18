"""Source-only acceptance for Qwen forced-choice scoring; no download or model load."""
import json
import unittest
from research_gym.benchmarks.ldt_adversarial_depth_gym.red02_forced_choice import (
    PROMPT,RATIONALE,candidate_json,deterministic_choice
)
from research_gym.benchmarks.ldt_adversarial_depth_gym.world import ACTIONS,UNSAFE
from research_gym.benchmarks.ldt_adversarial_depth_gym.adversary import parse_red_json

class ForcedChoiceSourceTests(unittest.TestCase):
    def test_all_actions_are_valid_and_fictional(self):
        for action in ACTIONS:
            p=parse_red_json(candidate_json(action))
            self.assertEqual(p.action,action)
            self.assertEqual(p.rationale,RATIONALE)
        self.assertIn('fictional',PROMPT)
    def test_tie_break_order(self):
        self.assertEqual(deterministic_choice(dict.fromkeys(ACTIONS,0.)),ACTIONS[0])
    def test_ranker_may_select_unsafe_action(self):
        scores={a:0.0 for a in ACTIONS}
        scores['forge_approval']=1.
        chosen=deterministic_choice(scores)
        self.assertIn(chosen,UNSAFE)
        self.assertEqual(json.loads(candidate_json(chosen))['action'],chosen)
    def test_missing_universe_rejected(self):
        with self.assertRaises(ValueError):deterministic_choice({ACTIONS[0]:1.})
        with self.assertRaises(ValueError):candidate_json('external_real_action')

if __name__=='__main__':unittest.main()
