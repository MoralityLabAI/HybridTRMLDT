"""Source-only tests for RED02 live model loader; never download on unit tests."""
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from research_gym.benchmarks.ldt_adversarial_depth_gym.red02_live_ci import (
    MODEL_ID,REVISION_PREFIX,ROOTS,DEPTHS,checksum,RED_SYSTEM_PROMPT,
)
from research_gym.benchmarks.ldt_adversarial_depth_gym.eval import evaluate_episode
from research_gym.benchmarks.ldt_adversarial_depth_gym.adversary import QwenAdversary,deterministic_red
from research_gym.benchmarks.ldt_adversarial_depth_gym.planner import optimal_safe_plan
from research_gym.benchmarks.ldt_adversarial_depth_gym.world import MoralControlWorld

class Red02SourceOnly(unittest.TestCase):
    def test_registered_model(self):
        self.assertEqual(MODEL_ID,'Qwen/Qwen2.5-0.5B-Instruct')
        self.assertEqual(REVISION_PREFIX,'7ae5576')
        self.assertEqual(ROOTS,(901,902))
        self.assertEqual(DEPTHS,(2,8))
    def test_prompt_is_fictional(self):
        self.assertIn('fictional',RED_SYSTEM_PROMPT)
        self.assertIn('no real',RED_SYSTEM_PROMPT)
    def test_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'abc')
            self.assertEqual(checksum(p),sha256(b'abc').hexdigest())
    def test_native_gate_blocks_mock_and_host_regret_ceiling(self):
        for root in ROOTS:
            for d in DEPTHS:
                r=evaluate_episode(root,d,QwenAdversary(deterministic_red),mode='ldt_gate')
                self.assertEqual(r['host_violations'],0)
                optimum=optimal_safe_plan(MoralControlWorld(root,horizon=d))
                self.assertGreaterEqual(optimum['max_terminal_utility'],r['terminal']['scalar_for_host'])

if __name__=='__main__':unittest.main()
