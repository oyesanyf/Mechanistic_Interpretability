#!/usr/bin/env python3
"""
Unit tests for ReST-GRPO group advantage normalization, surrogate clipping, and KL penalty.
"""

import unittest
import torch
import torch.nn as nn
from rest_rl.grpo_trainer import (
    compute_group_advantages,
    compute_surrogate_loss,
    compute_kl_penalty,
    GRPOTrainer,
    GRPOTrainerConfig,
)
from rest_rl.sampler import PromptGroupSample, CompletionSample
from rest_rl.verifiers import VerificationResult


class TestGRPOAdvantagesAndLoss(unittest.TestCase):
    def test_advantage_normalization_zero_mean(self):
        rewards = torch.tensor([0.2, 0.4, 0.8, 1.0])
        adv = compute_group_advantages(rewards, eps=1e-5)
        self.assertEqual(adv.shape, rewards.shape)
        # Mean of normalized advantages must be zero
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=4)
        # Monotonicity: higher reward must have higher advantage
        self.assertTrue((adv[3] > adv[2] > adv[1] > adv[0]).item())

    def test_equal_rewards_zero_advantages(self):
        # Edge case: All samples get identical reward -> zero variance
        rewards = torch.tensor([0.8, 0.8, 0.8, 0.8])
        adv = compute_group_advantages(rewards, eps=1e-5)
        self.assertEqual(adv.tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_single_sample_advantage(self):
        rewards = torch.tensor([0.5])
        adv = compute_group_advantages(rewards)
        self.assertEqual(adv.tolist(), [0.0])

    def test_surrogate_clipping_behavior(self):
        # Ratio = 1.0 (un-updated policy)
        log_p = torch.tensor([0.0])
        old_log_p = torch.tensor([0.0])
        advantage = torch.tensor([2.0])
        surr, clip_frac = compute_surrogate_loss(log_p, old_log_p, advantage, clip_eps=0.2)
        self.assertAlmostEqual(surr.item(), 2.0, places=4)
        self.assertEqual(clip_frac.item(), 0.0)

        # Ratio = 1.5 (exceeds 1 + clip_eps=1.2)
        log_p_high = torch.tensor([0.405465])  # ln(1.5)
        surr_clipped, clip_frac_high = compute_surrogate_loss(log_p_high, old_log_p, advantage, clip_eps=0.2)
        # Should be clipped to 1.2 * 2.0 = 2.4 instead of 1.5 * 2.0 = 3.0
        self.assertAlmostEqual(surr_clipped.item(), 2.4, places=3)
        self.assertEqual(clip_frac_high.item(), 1.0)

        # Negative advantage: when advantage < 0, ratio 1.5 -> clipped to min(1.5 * -2, 1.2 * -2) = -3.0
        neg_adv = torch.tensor([-2.0])
        surr_neg, _ = compute_surrogate_loss(log_p_high, old_log_p, neg_adv, clip_eps=0.2)
        self.assertAlmostEqual(surr_neg.item(), -3.0, places=3)

    def test_kl_penalty_properties(self):
        # Identical distributions -> KL must be exactly 0
        log_p = torch.randn(10)
        kl_zero = compute_kl_penalty(log_p, log_p)
        self.assertAlmostEqual(kl_zero.item(), 0.0, places=5)

        # Divergent distributions -> KL must be strictly positive
        ref_log_p = log_p + torch.randn(10) * 0.5
        kl_pos = compute_kl_penalty(log_p, ref_log_p)
        self.assertGreater(kl_pos.item(), 0.0)

    def test_trainer_optimization_step_mock_model(self):
        # Minimal mock causal LM head
        class SmallLM(nn.Module):
            def __init__(self, vocab_size=20, d_model=16):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                class Out:
                    pass
                h = self.embed(input_ids)
                out = Out()
                out.logits = self.head(h)
                return out

        policy_model = SmallLM()
        ref_model = SmallLM()
        ref_model.load_state_dict(policy_model.state_dict())

        config = GRPOTrainerConfig(
            clip_eps=0.2,
            kl_coeff=0.01,
            learning_rate=1e-3,
            device="cpu",
        )
        trainer = GRPOTrainer(
            policy_model=policy_model,
            ref_model=ref_model,
            config=config,
        )

        # Create dummy group sample with token ids
        prompt_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        samples = []
        for i, r in enumerate([0.2, 0.4, 0.8, 1.0]):
            comp_ids = torch.tensor([4 + i, 5, 6], dtype=torch.long)
            v = VerificationResult(
                is_safe=(r > 0.5),
                refusal_score=r,
                benign_score=r,
                format_score=1.0,
                jailbreak_score=1.0,
                language_score=1.0,
                total_reward=r,
            )
            samples.append(
                CompletionSample(
                    completion_text=f"comp_{i}",
                    full_text=f"prompt comp_{i}",
                    verification=v,
                    reward=r,
                    input_ids=prompt_ids,
                    completion_ids=comp_ids,
                    is_safe=v.is_safe,
                )
            )

        group_sample = PromptGroupSample(
            prompt="test prompt",
            language="English",
            prompt_kind="unsafe",
            scaffold="baseline",
            samples=samples,
            rewards=torch.tensor([0.2, 0.4, 0.8, 1.0]),
        )

        metrics = trainer.train_step(group_sample)
        self.assertIn("total_loss", metrics.to_dict())
        self.assertIn("surrogate_loss", metrics.to_dict())
        self.assertIn("kl_divergence", metrics.to_dict())
        self.assertAlmostEqual(metrics.mean_reward, 0.6, places=4)


if __name__ == "__main__":
    unittest.main()
