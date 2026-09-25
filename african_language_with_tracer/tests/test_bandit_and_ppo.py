#!/usr/bin/env python3
"""
Unit tests for ContextualBanditController and ConstrainedPPOController.
"""

import unittest
import torch
from deep_noir_rl.bandit_controller import ContextualBanditController, SteeringAction, BanditDecision
from deep_noir_rl.ppo_controller import ConstrainedPPOController, PPODecision


class TestBanditAndPPO(unittest.TestCase):
    def setUp(self):
        self.state_dim = 15
        self.candidate_layers = [10, 14, 18]
        self.candidate_magnitudes = [5.0, 15.0]

    def test_bandit_action_space_and_selection(self):
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            include_head_subsets=True,
            exploration_c=1.0,
        )
        # Action 0 is No Steering, plus 3 layers * 2 magnitudes * 2 head options = 1 + 12 = 13
        self.assertEqual(len(bandit.actions), 13)
        self.assertTrue(bandit.actions[0].is_no_op)
        self.assertEqual(bandit.actions[0].magnitude, 0.0)

        fake_state = torch.randn(self.state_dim)
        decision = bandit.select_action(fake_state)
        self.assertIsInstance(decision, BanditDecision)
        self.assertIsInstance(decision.action, SteeringAction)
        self.assertIn(decision.action.action_id, range(len(bandit.actions)))

    def test_bandit_dynamic_causal_heads(self):
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=[10],
            candidate_magnitudes=[5.0],
            include_head_subsets=True,
        )
        fake_state = torch.randn(self.state_dim)
        causal_heads = {10: [7, 8, 9]}
        head_action = [a for a in bandit.actions if a.target_heads is not None][0]
        bandit.update(fake_state, head_action, reward=100.0)
        decision = bandit.select_action(fake_state, active_causal_heads_by_layer=causal_heads)
        self.assertEqual(decision.action.target_heads, [7, 8, 9])

    def test_bandit_online_learning_update(self):
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
        )
        fake_state = torch.ones(self.state_dim)
        action = bandit.actions[1]

        # Initial b vector for action 1 is zeros
        self.assertEqual(torch.norm(bandit.b[1]).item(), 0.0)

        # Update with positive reward 1.0
        bandit.update(fake_state, action, reward=1.0)
        self.assertGreater(torch.norm(bandit.b[1]).item(), 0.0)
        self.assertEqual(bandit.pull_counts[1], 1)
        self.assertEqual(bandit.step_count, 1)

    def test_constrained_ppo_action_and_update(self):
        ppo = ConstrainedPPOController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            max_safety_cost_limit=0.20,
            lagrangian_lr=0.1,
            initial_lagrangian=0.5,
            device="cpu",
        )
        self.assertGreater(ppo.num_actions, 1)

        fake_state = torch.randn(self.state_dim)
        decision = ppo.select_action(fake_state)
        self.assertIsInstance(decision, PPODecision)
        self.assertIsInstance(decision.value_estimate, float)
        self.assertIsInstance(decision.cost_estimate, float)

        # Buffer several transitions
        for _ in range(8):
            s = torch.randn(self.state_dim)
            d = ppo.select_action(s)
            ppo.record_step(
                state=s,
                action_index=d.action_index,
                reward=0.8,
                cost=0.45,  # Exceeds limit 0.20 -> triggers Lagrangian increase
                log_prob=d.log_prob,
                value=d.value_estimate,
                cost_val=d.cost_estimate,
            )

        self.assertEqual(len(ppo.buffer), 8)
        initial_lambda = ppo.lagrangian_lambda

        # Run PPO update
        metrics = ppo.update(ppo_epochs=2, batch_size=4)
        self.assertIn("training_loss", metrics)
        self.assertIn("lagrangian_lambda", metrics)
        self.assertEqual(len(ppo.buffer), 0)  # Buffer cleared
        # Since cost was 0.45 > limit 0.20, Lagrangian lambda should have increased
        self.assertGreater(ppo.lagrangian_lambda, initial_lambda)


if __name__ == "__main__":
    unittest.main()
