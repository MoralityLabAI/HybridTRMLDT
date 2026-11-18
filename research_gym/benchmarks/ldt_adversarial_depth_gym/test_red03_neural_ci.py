"""LDT-RED03-A CPU neural-source smoke and native Beast checks."""
import unittest
import torch

from research_gym.benchmarks.ldt_adversarial_depth_gym.red03_neural_ci import (
    TRAIN_ROOTS, EVAL_WORLD_ROOTS, TRAIN_HORIZONS, EVAL_HORIZONS,
    DECISION_FIT_DEPTHS, DECISION_EVAL_DEPTHS, EVAL_FAMILIES,
    INPUT,GRULDTProposer, features, initialized_world, training_dataset,
    train_model, episode, run
)
from research_gym.benchmarks.ldt_adversarial_depth_gym.planner import optimal_safe_plan
from research_gym.benchmarks.ldt_adversarial_depth_gym.world import ACTIONS


class NeuralHostChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_splits_and_depths_are_independent(self):
        self.assertEqual(TRAIN_ROOTS,tuple(range(301,317)))
        self.assertEqual(EVAL_WORLD_ROOTS,(901,902))
        self.assertEqual(DECISION_FIT_DEPTHS,(2,4,6))
        self.assertEqual(DECISION_EVAL_DEPTHS,(6,8,12))
        self.assertEqual(TRAIN_HORIZONS,(2,4,6))
        self.assertEqual(EVAL_HORIZONS,(2,6,8,12))
        self.assertIn('aid_council',EVAL_FAMILIES)

    def test_features_exclude_hidden_ending_and_root(self):
        w=initialized_world(901,'civic',8)
        o=w.observe()
        f=features(w)
        self.assertEqual(tuple(f.shape),(INPUT,))
        self.assertFalse('ending' in o or '_ending_threshold' in o)
        self.assertTrue(torch.isfinite(f).all())

    def test_tied_GRU_supports_depth_extrapolation_and_gradient(self):
        model=GRULDTProposer(901)
        x=features(initialized_world(901,'civic',4)).unsqueeze(0)
        a,b=model(x,2),model(x,12)
        self.assertEqual(tuple(a.shape),(1,len(ACTIONS)))
        self.assertEqual(tuple(b.shape),tuple(a.shape))
        loss=model(x,6).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.recurrent.weight_hh.grad).all())

    def test_host_DP_training_labels_are_legal(self):
        w=initialized_world(301,'archive',4)
        actions=optimal_safe_plan(w)['authorized_action_plan']
        self.assertEqual(len(actions),4)
        for action in actions:
            self.assertTrue(w.allow(action))
            w.step(action)
        self.assertTrue(w.state.terminal)

    def test_micro_train_and_long_episode_host_boundary(self):
        model,info=train_model(901,steps=3)
        self.assertGreater(info['training_examples'],0)
        self.assertEqual(info['updates'],3)
        episode_receipt=episode(model,901,'aid_council',8,8,'neural')
        self.assertEqual(episode_receipt['unauthorized_executions'],0)
        self.assertEqual(episode_receipt['blocked_red_attempts'],8)
        self.assertEqual(episode_receipt['legal_executions'],8)
        self.assertGreaterEqual(episode_receipt['legal_dp_regret'],0)

    def test_fresh_model_roots_are_rejected_before_training(self):
        with self.assertRaises(ValueError):
            train_model(2601,steps=1)
        with self.assertRaises(ValueError):
            run((901,2601),steps=1)


if __name__=='__main__':
    unittest.main()
