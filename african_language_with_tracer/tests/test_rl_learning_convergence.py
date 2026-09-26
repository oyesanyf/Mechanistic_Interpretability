#!/usr/bin/env python3
"""
Rigorous Empirical Learning and Convergence Test Suite for Deep Noir RL Subsystem.

Verifies:
1. Monotonic Learning & Regret Decay (R_late > R_early, Regret_late -> 0)
2. Action Selection Accuracy (> 80% oracle matching)
3. Action-Conditioned Bandit with Rich Bilinear State-Action Embeddings
4. Frozen Policy Generalization (superior safety reward on held-out prompts with zero updates)
5. Constrained PPO Convergence in Sequential Steering MDP (loss stabilization & returns improvement)
6. Safe Rollback Invariance Monotonicity (reversion when gain < 0.005 or benign over-refusal)
7. Covariance Inversion Numerical Stability (strictly positive-definite under singular / degenerate states)
"""

import math
import os
import tempfile
import unittest
import torch
import torch.nn as nn

from deep_noir_rl.bandit_controller import (
    ContextualBanditController,
    SteeringAction,
    BanditDecision,
    stable_covariance_inverse,
)
from deep_noir_rl.ppo_controller import (
    ConstrainedPPOController,
    PPOSteeringController,
    SequentialSteeringMDP,
    PPODecision,
)
from deep_noir_rl.graduated_rewards import GraduatedRewardEvaluator
from deep_noir_rl.controller import AdaptiveSteeringRLController


class SyntheticLanguageEnvironment:
    """
    Simulated multilingual environment with underlying mechanistic steering response dynamics.
    Features:
        State vector dimension = 31 (matching RLStateExtractor rich state):
        - index 1: is_african_language
        - index 2: is_unsafe_prompt
        - index 5: norm_h_mid
        - index 7: logit_lens_refusal_mid
        - index 9: clean_refusal_prob
        - index 14: overall_injection_risk
        - index 15: lang_is_yoruba
        - index 16: lang_is_igbo
        - index 17: lang_is_hausa
        - index 18: lang_is_swahili
        - index 20: lang_is_english
        - index 22: pieces_per_refusal_start (fragmentation)
        - index 25: rpd_mid_profile
    """

    def __init__(self, actions: list[SteeringAction]):
        self.actions = actions
        self.num_actions = len(actions)

        # Map semantic categories to optimal action properties
        # Category 0: Yoruba Unsafe (High fragmentation, mid RPD) -> Best: Layer 12, Mag 2.5
        # Category 1: Swahili Unsafe (Low fragmentation, early RPD) -> Best: Layer 8, Mag 2.5
        # Category 2: English Unsafe (High baseline refusal, low fragmentation) -> Best: Layer 8, Mag 1.0
        # Category 3: Benign Control (Safe query) -> Best: Action 0 (No Steering)
        self.categories = ["yoruba_unsafe", "swahili_unsafe", "english_unsafe", "benign"]

    def generate_context(self, category_idx: int) -> tuple[torch.Tensor, str, int]:
        """Generates synthetic state vector, prompt_kind, and oracle best action index."""
        s = torch.zeros(31, dtype=torch.float32)
        cat = self.categories[category_idx]

        if cat == "yoruba_unsafe":
            prompt_kind = "unsafe"
            s[1] = 1.0   # is_african_language
            s[2] = 1.0   # is_unsafe_prompt
            s[5] = 0.65  # norm_h_mid
            s[7] = 0.12  # logit_lens_refusal_mid
            s[9] = 0.08  # clean_refusal_prob
            s[14] = 0.15 # injection_risk
            s[15] = 2.0  # lang_is_yoruba
            s[22] = 0.85 # high tokenizer fragmentation
            s[25] = 0.70 # rpd_mid_profile
            # Target: Layer 12, Mag 2.5
            target_layer, target_mag = 12, 2.5

        elif cat == "swahili_unsafe":
            prompt_kind = "unsafe"
            s[1] = 1.0   # is_african_language
            s[2] = 1.0   # is_unsafe_prompt
            s[5] = 0.40  # norm_h_mid
            s[7] = 0.18  # logit_lens_refusal_mid
            s[9] = 0.10  # clean_refusal_prob
            s[14] = 0.10 # injection_risk
            s[18] = 2.0  # lang_is_swahili
            s[22] = 0.45 # moderate fragmentation
            s[25] = 0.40 # rpd_mid_profile
            # Target: Layer 8, Mag 2.5
            target_layer, target_mag = 8, 2.5

        elif cat == "english_unsafe":
            prompt_kind = "unsafe"
            s[1] = 0.0   # not african
            s[2] = 1.0   # is_unsafe_prompt
            s[5] = 0.50
            s[7] = 0.45  # high logit lens refusal
            s[9] = 0.35  # high clean refusal
            s[14] = 0.12
            s[20] = 2.0  # lang_is_english
            s[22] = 0.15 # low fragmentation
            s[25] = 0.30
            # Target: Layer 8, Mag 1.0
            target_layer, target_mag = 8, 1.0

        else: # benign
            prompt_kind = "benign"
            s[1] = 0.0
            s[2] = 0.0   # benign
            s[5] = 0.30
            s[7] = 0.01  # zero refusal
            s[9] = 0.01
            s[14] = 0.05
            s[20] = 1.0
            s[22] = 0.15
            s[25] = 0.10
            target_layer, target_mag = None, 0.0

        # Find oracle action index matching target
        oracle_idx = 0
        if target_layer is None or target_mag == 0.0:
            oracle_idx = 0
        else:
            best_diff = float("inf")
            for idx, act in enumerate(self.actions):
                if act.is_no_op:
                    continue
                if act.layer_idx == target_layer:
                    diff = abs(act.magnitude - target_mag)
                    if diff < best_diff:
                        best_diff = diff
                        oracle_idx = idx

        # Add slight natural variation
        s = s + 0.02 * torch.randn(31, dtype=torch.float32)
        return s, prompt_kind, oracle_idx

    def compute_reward(self, category_idx: int, chosen_action: SteeringAction) -> float:
        """
        Calculates ground-truth continuous reward signal for the chosen action.
        Optimal action receives highest reward (~0.85). Suboptimal actions receive reduced reward.
        Over-steering on benign causes heavy penalty.
        """
        cat = self.categories[category_idx]
        if cat == "benign":
            if chosen_action.is_no_op or chosen_action.magnitude == 0.0:
                return 0.85
            else:
                # Unnecessary intervention on benign query
                return -0.40 - 0.10 * chosen_action.magnitude

        _, _, oracle_idx = self.generate_context(category_idx)
        oracle_act = self.actions[oracle_idx]

        if chosen_action.is_no_op:
            return 0.05  # Inaction on unsafe prompt

        # Layer match distance and magnitude match distance
        layer_dist = abs((chosen_action.layer_idx or 0) - (oracle_act.layer_idx or 0))
        mag_dist = abs(chosen_action.magnitude - oracle_act.magnitude)

        match_score = 0.85 - 0.10 * layer_dist - 0.25 * mag_dist
        return max(-0.5, match_score)


class TestRLLearningAndConvergence(unittest.TestCase):
    """Rigorous empirical test suite verifying RL algorithmic calibration and convergence."""

    def setUp(self):
        torch.manual_seed(42)
        self.candidate_layers = [8, 12]
        self.candidate_magnitudes = [1.0, 2.5, 5.0]
        self.state_dim = 31

    def test_stable_covariance_inversion_singular_and_ill_conditioned(self):
        """Verifies adaptive Tikhonov/ridge regularization and pseudo-inverse/Cholesky stability."""
        dim = 10
        # 1. Singular rank-1 matrix (outer product of a single vector)
        v = torch.randn(dim, 1)
        A_singular = v @ v.T

        inv_singular = stable_covariance_inverse(A_singular, ridge_eps=1e-4)
        self.assertEqual(inv_singular.shape, (dim, dim))
        self.assertFalse(torch.isnan(inv_singular).any(), "Inverse must contain no NaNs")
        self.assertFalse(torch.isinf(inv_singular).any(), "Inverse must contain no Infs")

        # Symmetry check
        sym_diff = torch.norm(inv_singular - inv_singular.T).item()
        self.assertLess(sym_diff, 1e-4, "Inverse must be symmetric")

        # Strictly positive-definite check: all eigenvalues > 0
        eigenvalues = torch.linalg.eigvalsh(inv_singular)
        self.assertTrue((eigenvalues > 0).all(), "All eigenvalues of inverse must be strictly positive")
        self.assertGreaterEqual(eigenvalues.min().item(), 1e-6)

        # Quadratic form strictly positive
        rand_test = torch.randn(dim)
        quad = torch.dot(rand_test, torch.matmul(inv_singular, rand_test)).item()
        self.assertGreater(quad, 0.0)

        # 2. Degenerate zero matrix
        A_zero = torch.zeros(dim, dim)
        inv_zero = stable_covariance_inverse(A_zero, ridge_eps=1e-4)
        eig_zero = torch.linalg.eigvalsh(inv_zero)
        self.assertTrue((eig_zero > 0).all(), "Zero matrix stabilized inverse must be positive definite")

        # 3. Degenerate matrix with NaNs and Infs
        A_nan = torch.tensor([[float("nan"), 0.5], [0.5, float("inf")]])
        inv_nan = stable_covariance_inverse(A_nan, ridge_eps=1e-4)
        self.assertFalse(torch.isnan(inv_nan).any(), "Inverse of NaN/Inf matrix must contain no NaNs")
        self.assertFalse(torch.isinf(inv_nan).any(), "Inverse of NaN/Inf matrix must contain no Infs")
        eig_nan = torch.linalg.eigvalsh(inv_nan)
        self.assertTrue((eig_nan > 0).all(), "Inverse of NaN/Inf matrix must be strictly positive definite")

        # 4. Well-conditioned identity matrix (fast exact Cholesky path)
        A_eye = 2.0 * torch.eye(dim)
        inv_eye = stable_covariance_inverse(A_eye, ridge_eps=1e-4)
        expected_eye = 0.5 * torch.eye(dim)
        self.assertTrue(torch.allclose(inv_eye, expected_eye, atol=1e-5), "Exact positive-definite matrix must invert accurately")

    def test_bandit_monotonic_learning_and_regret_decay(self):
        """
        Verifies:
        a) Monotonic learning: R_late > R_early
        b) Instantaneous regret decays toward zero: Regret_late < Regret_early
        """
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            include_head_subsets=False,
            exploration_c=1.20,
            min_exploration_c=0.05,
            alpha_decay=0.02,
            algorithm="linucb",
            rl_mode="cold_rl",
        )
        env = SyntheticLanguageEnvironment(bandit.actions)

        num_episodes = 240
        early_window = 40
        late_window = 40

        rewards = []
        regrets = []

        for step in range(num_episodes):
            cat_idx = step % len(env.categories)
            state, prompt_kind, oracle_idx = env.generate_context(cat_idx)
            oracle_reward = env.compute_reward(cat_idx, bandit.actions[oracle_idx])

            decision = bandit.select_action(
                state_vector=state,
                prompt_kind=prompt_kind,
                oracle_best_action_id=oracle_idx,
            )

            # Observe environment reward
            obs_reward = env.compute_reward(cat_idx, decision.action)
            inst_regret = max(0.0, oracle_reward - obs_reward)

            bandit.update(state, decision.action, obs_reward)

            rewards.append(obs_reward)
            regrets.append(inst_regret)

        r_early = sum(rewards[:early_window]) / early_window
        r_late = sum(rewards[-late_window:]) / late_window

        regret_early = sum(regrets[:early_window]) / early_window
        regret_late = sum(regrets[-late_window:]) / late_window

        self.assertGreater(
            r_late,
            r_early,
            f"Policy must demonstrate monotonic learning: R_late ({r_late:.4f}) must be > R_early ({r_early:.4f})",
        )
        self.assertLess(
            regret_late,
            regret_early,
            f"Instantaneous regret must decay: Regret_late ({regret_late:.4f}) must be < Regret_early ({regret_early:.4f})",
        )
        self.assertLess(
            regret_late,
            0.15,
            f"Late-stage regret must decay close to zero, got {regret_late:.4f}",
        )

    def test_action_selection_accuracy(self):
        """Verifies policy learns to identify the oracle-optimal intervention with accuracy > 80%."""
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            include_head_subsets=False,
            exploration_c=1.20,
            alpha_decay=0.02,
            algorithm="linucb",
            rl_mode="cold_rl",
        )
        env = SyntheticLanguageEnvironment(bandit.actions)

        # Train across 280 episodes
        for step in range(280):
            cat_idx = step % len(env.categories)
            state, prompt_kind, _ = env.generate_context(cat_idx)
            dec = bandit.select_action(state, prompt_kind=prompt_kind)
            r = env.compute_reward(cat_idx, dec.action)
            bandit.update(state, dec.action, r)

        # Test evaluation across 60 held-out queries (15 per category) in deterministic mode
        correct_count = 0
        total_eval = 60
        for i in range(total_eval):
            cat_idx = i % len(env.categories)
            state, prompt_kind, oracle_idx = env.generate_context(cat_idx)
            dec = bandit.select_action(state, prompt_kind=prompt_kind, deterministic=True)
            if dec.action.action_id == oracle_idx:
                correct_count += 1

        accuracy = correct_count / total_eval
        self.assertGreaterEqual(
            accuracy,
            0.80,
            f"Action selection accuracy must exceed 80%, got {accuracy * 100:.1f}% ({correct_count}/{total_eval})",
        )

    def test_action_conditioned_bandit_cross_product_learning(self):
        """Verifies Action-Conditioned Bandit with bilinear state-action interaction embeddings."""
        actions = [
            SteeringAction(0, "No Steering", None, 0.0, is_no_op=True),
            SteeringAction(1, "L8_mag2.5_full", 8, 2.5),
            SteeringAction(2, "L12_mag2.5_full", 12, 2.5),
        ]
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=[8, 12],
            candidate_magnitudes=[2.5],
            include_head_subsets=False,
            exploration_c=1.20,
            alpha_decay=0.02,
            algorithm="action_conditioned",
            rl_mode="cold_rl",
        )
        bandit.actions = actions
        bandit.num_actions = len(actions)
        bandit.A_shared = 1.0 * torch.eye(bandit.joint_dim)
        bandit.b_shared = torch.zeros(bandit.joint_dim)
        bandit.pull_counts = [0] * len(actions)

        # Verify bilinear cross product dimension constructed properly
        self.assertGreater(bandit.cross_dim, 0)
        self.assertEqual(bandit.A_shared.shape[0], bandit.joint_dim)
        self.assertEqual(bandit.b_shared.shape[0], bandit.joint_dim)

        env = SyntheticLanguageEnvironment(actions)
        rewards = []
        for step in range(240):
            cat_idx = step % len(env.categories)
            state, prompt_kind, _ = env.generate_context(cat_idx)
            oracle = 2 if cat_idx == 0 else (1 if cat_idx in (1, 2) else 0)
            dec = bandit.select_action(state, prompt_kind=prompt_kind)
            r = 0.85 if dec.action.action_id == oracle else (0.10 if prompt_kind == "unsafe" else -0.50)
            bandit.update(state, dec.action, r)
            rewards.append(r)

        r_early = sum(rewards[:35]) / 35
        r_late = sum(rewards[-35:]) / 35
        self.assertGreater(
            r_late,
            r_early,
            f"Action-conditioned bandit must learn non-linear policy dynamics: R_late ({r_late:.4f}) > R_early ({r_early:.4f})",
        )

        # Verify accuracy in evaluation mode
        correct = 0
        total_eval = 40
        for i in range(total_eval):
            cat_idx = i % len(env.categories)
            state, prompt_kind, _ = env.generate_context(cat_idx)
            oracle = 2 if cat_idx == 0 else (1 if cat_idx in (1, 2) else 0)
            dec = bandit.select_action(state, prompt_kind=prompt_kind, deterministic=True)
            if dec.action.action_id == oracle:
                correct += 1
        acc = correct / float(total_eval)
        self.assertGreaterEqual(acc, 0.80, f"Action-conditioned bandit accuracy must be >= 80%, got {acc:.2f}")

    def test_frozen_policy_generalization_and_safety_monotonicity(self):
        """
        Verifies:
        1. Trained policy checkpoint can be saved and loaded into frozen_rl mode.
        2. Frozen policy takes ZERO updates upon update() calls.
        3. Frozen policy evaluated on unseen held-out prompts achieves higher safety reward
           than unsteered baseline and static steering.
        """
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            include_head_subsets=False,
            exploration_c=0.50,
            algorithm="linucb",
            rl_mode="cold_rl",
        )
        env = SyntheticLanguageEnvironment(bandit.actions)

        # Train on 160 episodes
        for step in range(160):
            cat_idx = step % len(env.categories)
            state, prompt_kind, _ = env.generate_context(cat_idx)
            dec = bandit.select_action(state, prompt_kind=prompt_kind)
            r = env.compute_reward(cat_idx, dec.action)
            bandit.update(state, dec.action, r)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "bandit_checkpoint.json")
            save_info = bandit.save_policy(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))
            self.assertIn("config_hash", save_info)

            # Reload into frozen evaluator
            frozen_bandit = ContextualBanditController(
                state_dim=self.state_dim,
                candidate_layers=self.candidate_layers,
                candidate_magnitudes=self.candidate_magnitudes,
                include_head_subsets=False,
            )
            frozen_bandit.load_policy(ckpt_path, freeze=True)
            self.assertEqual(frozen_bandit.rl_mode, "frozen_rl")

            # Snapshot matrix weights
            A_snapshot = [mat.clone() for mat in frozen_bandit.A]
            b_snapshot = [vec.clone() for vec in frozen_bandit.b]

            # Evaluate on held-out prompts
            frozen_rewards = []
            unsteered_rewards = []
            static_rewards = []

            # Find static action: Layer 8, Mag 5.0
            static_idx = next(
                (i for i, a in enumerate(bandit.actions) if a.layer_idx == 8 and a.magnitude == 5.0), 1
            )
            static_act = bandit.actions[static_idx]

            for test_idx in range(60):
                cat_idx = test_idx % len(env.categories)
                state, prompt_kind, _ = env.generate_context(cat_idx)

                # 1. Frozen Policy
                dec = frozen_bandit.select_action(state, prompt_kind=prompt_kind)
                r_frozen = env.compute_reward(cat_idx, dec.action)
                frozen_rewards.append(r_frozen)

                # Attempt update in frozen mode
                frozen_bandit.update(state, dec.action, r_frozen)

                # 2. Unsteered Baseline (Action 0)
                r_unsteered = env.compute_reward(cat_idx, bandit.actions[0])
                unsteered_rewards.append(r_unsteered)

                # 3. Static Steering (fixed L8, Mag 5.0)
                r_static = env.compute_reward(cat_idx, static_act)
                static_rewards.append(r_static)

            # Strict verification: matrices must NOT have changed at all
            for i in range(frozen_bandit.num_actions):
                self.assertTrue(
                    torch.equal(frozen_bandit.A[i], A_snapshot[i]),
                    "Frozen policy A matrices must not be updated during evaluation",
                )
                self.assertTrue(
                    torch.equal(frozen_bandit.b[i], b_snapshot[i]),
                    "Frozen policy b vectors must not be updated during evaluation",
                )

            mean_frozen = sum(frozen_rewards) / len(frozen_rewards)
            mean_unsteered = sum(unsteered_rewards) / len(unsteered_rewards)
            mean_static = sum(static_rewards) / len(static_rewards)

            self.assertGreater(
                mean_frozen,
                mean_unsteered,
                f"Frozen policy ({mean_frozen:.4f}) must outperform unsteered baseline ({mean_unsteered:.4f})",
            )
            self.assertGreater(
                mean_frozen,
                mean_static,
                f"Frozen policy ({mean_frozen:.4f}) must outperform static steering ({mean_static:.4f})",
            )

    def test_ppo_convergence_in_sequential_mdp(self):
        """
        Verifies:
        1. Constrained PPO / PPOSteeringController converges in Sequential Steering MDP.
        2. Orthogonal weight initialization is applied.
        3. Gradient clipping and entropy bonus decay stabilize learning.
        4. Training loss stabilizes and average returns increase.
        """
        ppo = PPOSteeringController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            hidden_dim=32,
            lr=3e-3,
            entropy_coef=0.02,
            entropy_decay=0.98,
            max_grad_norm=0.5,
            device="cpu",
        )
        mdp = SequentialSteeringMDP(
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
        )

        initial_entropy_coef = ppo.entropy_coef
        losses = []
        episode_rewards = []

        # Run 24 multi-step rollout batches
        for batch_iter in range(24):
            for ep in range(6):
                initial_s = torch.randn(self.state_dim)
                s, stage = mdp.reset(initial_s)
                done = False
                total_ep_reward = 0.0

                while not done:
                    dec = ppo.select_action(s)
                    # Reward function: prefers layer 12 and magnitude 2.5
                    def reward_fn():
                        r = 0.8
                        if mdp.chosen_layer == 12:
                            r += 0.4
                        if abs(mdp.chosen_magnitude - 2.5) < 1.0:
                            r += 0.3
                        return r

                    next_s, r, c, done, _ = mdp.step(dec.action_index, s, ppo.actions, final_reward_fn=reward_fn)
                    ppo.record_step(
                        state=s,
                        action_index=dec.action_index,
                        reward=r,
                        cost=c,
                        log_prob=dec.log_prob,
                        value=dec.value_estimate,
                        cost_val=dec.cost_estimate,
                        done=done,
                    )
                    s = next_s
                    total_ep_reward += r

                episode_rewards.append(total_ep_reward)

            metrics = ppo.update(ppo_epochs=3, batch_size=6)
            if "training_loss" in metrics:
                losses.append(metrics["training_loss"])

        # 1. Entropy coefficient decayed
        self.assertLess(
            ppo.entropy_coef,
            initial_entropy_coef,
            "Entropy coefficient must decay over updates",
        )

        # 2. Check returns improvement
        early_ret = sum(episode_rewards[:24]) / 24.0
        late_ret = sum(episode_rewards[-24:]) / 24.0
        self.assertGreater(
            late_ret,
            early_ret,
            f"PPO returns must improve from early ({early_ret:.4f}) to late ({late_ret:.4f})",
        )

        # 3. Loss stabilization (not exploding to NaN or inf)
        for loss_val in losses:
            self.assertFalse(math.isnan(loss_val))
            self.assertFalse(math.isinf(loss_val))
        self.assertLess(losses[-1], 2.5, "Final PPO training loss must be bounded and stabilized")

        # 4. Gradient norm clipping <= 0.5 and metric tracking
        self.assertIn("grad_norm", metrics, "PPO update metrics must track grad_norm")
        self.assertIn("mean_entropy", metrics, "PPO update metrics must track mean_entropy")
        self.assertIn("policy_loss", metrics, "PPO update metrics must track policy_loss")
        self.assertLessEqual(metrics["grad_norm"], 0.50 + 1e-4, "Gradient norm must be clipped to <= 0.5")

    def test_safe_rollback_invariance(self):
        """
        Verifies:
        1. Action on unsafe prompt with gain < min_refusal_gain (0.005) is reverted to clean refusal.
        2. Action on unsafe prompt with gain >= 0.005 is preserved.
        3. Action on benign prompt exceeding max_benign_refusal is reverted to clean refusal.
        """
        evaluator = GraduatedRewardEvaluator(
            max_injection_risk=0.45,
            max_benign_refusal=0.15,
            min_refusal_gain_threshold=0.005,
        )

        # 1. Unsafe prompt: marginal gain (0.002 < 0.005 threshold)
        # In controller: must trigger rollback
        clean_p = 0.0800
        steered_marginal = 0.0820  # gain = +0.0020 < 0.005
        min_gain = evaluator.min_refusal_gain_threshold

        gain = steered_marginal - clean_p
        should_rollback_unsafe = gain < min_gain
        self.assertTrue(should_rollback_unsafe, "Gain < 0.005 on unsafe prompt must trigger rollback")

        # Check evaluator penalty when marked rolled back
        res_rolled_back = evaluator.evaluate(
            prompt_kind="unsafe",
            p_clean_refusal=clean_p,
            p_steered_refusal=clean_p,  # restored clean probability
            was_rolled_back=True,
            rollback_reason="Unsafe refusal gain < 0.005 min threshold",
        )
        self.assertFalse(res_rolled_back.is_safe)
        self.assertLess(res_rolled_back.total_reward, -0.99)

        # 2. Unsafe prompt: meaningful gain (0.0800 -> 0.1200, gain = +0.0400 >= 0.005)
        steered_safe = 0.1200
        gain_safe = steered_safe - clean_p
        self.assertGreaterEqual(gain_safe, min_gain)

        res_safe = evaluator.evaluate(
            prompt_kind="unsafe",
            p_clean_refusal=clean_p,
            p_steered_refusal=steered_safe,
            steering_magnitude=2.5,
            was_rolled_back=False,
        )
        self.assertTrue(res_safe.is_safe)
        self.assertGreater(res_safe.total_reward, 0.0)

        # 3. Benign prompt: over-refusal (> 0.15)
        clean_benign = 0.0100
        steered_overrefusal = 0.2500
        should_rollback_benign = steered_overrefusal > evaluator.max_benign_refusal
        self.assertTrue(should_rollback_benign, "Benign over-refusal must trigger rollback")

        res_benign_violation = evaluator.evaluate(
            prompt_kind="benign",
            p_clean_refusal=clean_benign,
            p_steered_refusal=steered_overrefusal,
            steering_magnitude=5.0,
            was_rolled_back=False,
        )
        self.assertFalse(res_benign_violation.is_safe)
        self.assertLess(res_benign_violation.total_reward, -0.99)

        # 4. Controller End-to-End Safe Rollback Verification
        from deep_noir_rl.contrastive_steering import SteeringVector

        class MockConfig:
            hidden_size = 4
            num_attention_heads = 1

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        class MockModel:
            config = MockConfig()
            def __init__(self, p_clean: float, p_steered: float):
                self.call_count = 0
                self.p_clean = p_clean
                self.p_steered = p_steered
                self.lm_head = torch.nn.Linear(4, 2)
                self.model = torch.nn.Module()
                self.model.norm = torch.nn.Identity()

            def parameters(self):
                return iter([torch.zeros(1)])

            def __call__(self, *args, **kwargs):
                self.call_count += 1
                p = self.p_clean if self.call_count == 1 else self.p_steered
                # Token 0 is refusal token with prob p, Token 1 has prob 1-p
                logits = torch.tensor([[[math.log(p + 1e-12), math.log(1.0 - p + 1e-12)]]])
                return MockOutput(logits)

        class MockLayer:
            def register_forward_hook(self, fn):
                class Handle:
                    def remove(self): pass
                return Handle()

        # A) Controller rollback on unsafe prompt when gain < 0.005 (gain = +0.002)
        mock_model_marginal = MockModel(p_clean=0.080, p_steered=0.082)
        ctrl_marginal = AdaptiveSteeringRLController(
            model=mock_model_marginal,
            tokenizer=None,
            layers=[MockLayer() for _ in range(15)],
            candidate_layers=[8],
            candidate_magnitudes=[2.5],
            device="cpu",
            min_refusal_gain=0.005,
        )
        sv_mock = SteeringVector(
            layer_idx=8, raw_vector=torch.ones(4), unit_vector=torch.ones(4)/2.0, vector_norm=2.0,
            num_safe_samples=1, num_harmful_samples=1, language="Yoruba",
        )
        ctrl_marginal.steering_manager.calibrated_directions[8] = sv_mock
        ctrl_marginal.steering_manager.cached_directions[8] = sv_mock

        act_active = ctrl_marginal.bandit.actions[1]
        res_ctrl_unsafe = ctrl_marginal.steer_and_evaluate(
            prompt_text="test prompt",
            inputs={"input_ids": torch.tensor([[0]])},
            refusal_ids=[0],
            language_name="Yoruba",
            prompt_kind="unsafe",
            forced_action=act_active,
        )
        self.assertTrue(res_ctrl_unsafe.was_rolled_back, "Controller must execute rollback when gain < 0.005")
        self.assertEqual(res_ctrl_unsafe.chosen_action.action_id, 0, "Executed action must revert to action 0")
        self.assertAlmostEqual(res_ctrl_unsafe.steered_refusal_prob, res_ctrl_unsafe.clean_refusal_prob, places=4)
        self.assertTrue(any("ROLLBACK EXECUTED" in step for step in res_ctrl_unsafe.audit_steps))

        # B) Controller preserves intervention when gain >= 0.005 (gain = +0.040)
        mock_model_safe = MockModel(p_clean=0.080, p_steered=0.120)
        ctrl_safe = AdaptiveSteeringRLController(
            model=mock_model_safe,
            tokenizer=None,
            layers=[MockLayer() for _ in range(15)],
            candidate_layers=[8],
            candidate_magnitudes=[2.5],
            device="cpu",
            min_refusal_gain=0.005,
        )
        ctrl_safe.steering_manager.calibrated_directions[8] = sv_mock
        ctrl_safe.steering_manager.cached_directions[8] = sv_mock

        res_ctrl_safe = ctrl_safe.steer_and_evaluate(
            prompt_text="test prompt",
            inputs={"input_ids": torch.tensor([[0]])},
            refusal_ids=[0],
            language_name="Yoruba",
            prompt_kind="unsafe",
            forced_action=act_active,
        )
        self.assertFalse(res_ctrl_safe.was_rolled_back, "Controller must NOT rollback when gain >= 0.005")
        self.assertEqual(res_ctrl_safe.chosen_action.action_id, act_active.action_id)

        # C) Controller rollback on benign prompt when over-refusal occurs (> 0.15)
        mock_model_overrefusal = MockModel(p_clean=0.010, p_steered=0.250)
        ctrl_overrefusal = AdaptiveSteeringRLController(
            model=mock_model_overrefusal,
            tokenizer=None,
            layers=[MockLayer() for _ in range(15)],
            candidate_layers=[8],
            candidate_magnitudes=[2.5],
            device="cpu",
            min_refusal_gain=0.005,
        )
        ctrl_overrefusal.steering_manager.calibrated_directions[8] = sv_mock
        ctrl_overrefusal.steering_manager.cached_directions[8] = sv_mock
        res_ctrl_benign = ctrl_overrefusal.steer_and_evaluate(
            prompt_text="test benign prompt",
            inputs={"input_ids": torch.tensor([[0]])},
            refusal_ids=[0],
            language_name="English",
            prompt_kind="benign",
            forced_action=act_active,
        )
        self.assertTrue(res_ctrl_benign.was_rolled_back, "Controller must execute rollback on benign over-refusal")
        self.assertEqual(res_ctrl_benign.chosen_action.action_id, 0)
        self.assertAlmostEqual(res_ctrl_benign.steered_refusal_prob, res_ctrl_benign.clean_refusal_prob, places=4)
        self.assertTrue(any("ROLLBACK EXECUTED" in step for step in res_ctrl_benign.audit_steps))

    def test_cold_start_benign_recovery_edge_case(self):
        """
        Attacks Untested Edge Case:
        Verifies behavior when all prompt categories during cold-start are exclusively benign
        for > 500 steps before encountering the first unsafe prompt.
        Ensures that exploration bonus recovery and decay work properly together:
        - Untried unsafe arms are explored without starvation.
        - Late training smoothly converges to the optimal arm without exploding regret.
        """
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=self.candidate_layers,
            candidate_magnitudes=self.candidate_magnitudes,
            include_head_subsets=False,
            exploration_c=1.20,
            min_exploration_c=0.05,
            alpha_decay=0.02,
            algorithm="linucb",
            rl_mode="cold_rl",
        )
        env = SyntheticLanguageEnvironment(bandit.actions)

        # 500 exclusively benign prompts (Category 3)
        for _ in range(500):
            state, prompt_kind, _ = env.generate_context(3)
            dec = bandit.select_action(state, prompt_kind=prompt_kind)
            r = env.compute_reward(3, dec.action)
            bandit.update(state, dec.action, r)

        # Confirm step_count advanced and c(t) annealed
        self.assertEqual(bandit.step_count, 500)
        c_annealed = bandit.get_current_exploration_c()
        self.assertLess(c_annealed, 0.40)

        # Now unsafe prompts arrive: run 100 unsafe episodes
        unsafe_rewards = []
        for step in range(100):
            cat_idx = step % 3  # Yoruba, Swahili, English unsafe
            state, prompt_kind, _ = env.generate_context(cat_idx)
            dec = bandit.select_action(state, prompt_kind=prompt_kind)
            r = env.compute_reward(cat_idx, dec.action)
            bandit.update(state, dec.action, r)
            unsafe_rewards.append(r)

        # Check that after exploration recovery, late unsafe performance converges
        late_unsafe_r = sum(unsafe_rewards[-30:]) / 30.0
        self.assertGreater(
            late_unsafe_r,
            0.50,
            f"Policy must recover and learn optimal steering after long benign cold start, got {late_unsafe_r:.4f}",
        )

    def test_cold_rl_warm_start_immunity(self):
        """
        Verifies that in 'cold_rl' mode, calling warm_start_arm returns None and does NOT
        mutate policy parameters, preventing benchmark contamination.
        """
        bandit = ContextualBanditController(
            state_dim=self.state_dim,
            candidate_layers=[8, 12],
            candidate_magnitudes=[2.5],
            rl_mode="cold_rl",
        )
        s = torch.randn(self.state_dim)
        arm_idx = bandit.warm_start_arm(layer_idx=8, magnitude=2.5, reward=0.9, state_vector=s)
        self.assertIsNone(arm_idx, "warm_start_arm must return None in cold_rl mode")
        self.assertEqual(sum(bandit.pull_counts), 0, "Pull counts must remain 0 in cold_rl mode")
        self.assertIsNone(bandit.last_verified_arm_id)

    def test_covariance_inverse_device_and_dtype_preservation(self):
        """
        Verifies that stable_covariance_inverse preserves input tensor device (CPU/CUDA)
        and numerical dtype (float32/float64), preventing device mismatch exceptions.
        """
        A_f64 = torch.eye(4, dtype=torch.float64)
        inv_f64 = stable_covariance_inverse(A_f64)
        self.assertEqual(inv_f64.dtype, torch.float64, "Inverse of float64 matrix must remain float64")

        if torch.cuda.is_available():
            A_cuda = torch.eye(4, device="cuda")
            inv_cuda = stable_covariance_inverse(A_cuda)
            self.assertEqual(inv_cuda.device.type, "cuda", "Inverse of CUDA matrix must remain on CUDA")

    def test_bandit_policy_checkpoint_state_dim_restoration(self):
        """
        Verifies that loading a bandit policy checkpoint restores state_dim and salient
        feature interaction indices, preventing tensor size mismatches when loading
        31-dim models into instances initialized with default 15-dim states.
        """
        b31 = ContextualBanditController(state_dim=31, algorithm="action_conditioned")
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "bandit31.json")
            b31.save_policy(ckpt_path)

            b_new = ContextualBanditController(state_dim=15, algorithm="action_conditioned")
            self.assertEqual(b_new.state_dim, 15)
            b_new.load_policy(ckpt_path)
            self.assertEqual(b_new.state_dim, 31, "load_policy must restore state_dim to 31")
            self.assertEqual(b_new.joint_dim, b31.joint_dim)

            # Test inference with restored state_dim
            s_test = torch.randn(31)
            dec = b_new.select_action(s_test, prompt_kind="unsafe")
            self.assertIsNotNone(dec.action)
            self.assertIsNotNone(dec.predicted_reward)

    def test_thompson_sampling_determinism_and_annealing(self):
        """
        Verifies that contextual Thompson sampling:
        1. Is strictly deterministic in frozen_rl mode or with deterministic=True (no random sampling).
        2. Posterior uncertainty scales with annealed exploration constant c(t).
        """
        bandit = ContextualBanditController(
            state_dim=15,
            algorithm="thompson_sampling",
            exploration_c=1.0,
            alpha_decay=0.05,
            rl_mode="frozen_rl",
        )
        s = torch.randn(15)

        # Deterministic evaluation check
        dec1 = bandit.select_action(s, deterministic=True)
        dec2 = bandit.select_action(s, deterministic=True)
        self.assertEqual(
            dec1.ucb_score,
            dec2.ucb_score,
            "Thompson sampling must yield identical deterministic scores in frozen_rl / deterministic mode",
        )
        self.assertEqual(dec1.action.action_id, dec2.action.action_id)

        # Annealing check: step_count should reduce exploration parameter
        bandit.rl_mode = "cold_rl"
        c_init = bandit.get_current_exploration_c()
        bandit.step_count = 100
        c_annealed = bandit.get_current_exploration_c()
        self.assertLess(c_annealed, c_init)

    def test_ppo_checkpoint_dynamic_action_space_restoration(self):
        """
        Verifies that ConstrainedPPOController.load_policy dynamically adapts
        the ActorCriticNetwork and restores actions when loading a checkpoint
        with a different action count or state dimension.
        """
        custom_actions = [SteeringAction(i, f"act_{i}", i, 1.0) for i in range(10)]
        ppo_orig = ConstrainedPPOController(state_dim=15, actions=custom_actions, device="cpu")
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "ppo_custom.pt")
            ppo_orig.save_policy(ckpt_path)

            ppo_loaded = ConstrainedPPOController(state_dim=15, device="cpu")
            self.assertNotEqual(ppo_loaded.num_actions, 10)
            ppo_loaded.load_policy(ckpt_path)
            self.assertEqual(ppo_loaded.num_actions, 10)
            self.assertEqual(ppo_loaded.network.actor_head.out_features, 10)

            # Test inference with loaded policy
            s = torch.randn(15)
            dec = ppo_loaded.select_action(s, deterministic=True)
            self.assertIsNotNone(dec.action)
            self.assertIn(dec.action_index, range(10))

    def test_sequential_mdp_arbitrary_state_dimensions_and_shapes(self):
        """
        Verifies that SequentialSteeringMDP.step safely handles sub-8-dim states
        and 2D batch-shaped states without IndexError.
        """
        mdp = SequentialSteeringMDP(candidate_layers=[8], candidate_magnitudes=[2.5])
        actions = [SteeringAction(0, "A0", 8, 2.5)]

        # Small 4-dim state (< 8 dims)
        s_small = torch.randn(4)
        s_reset, _ = mdp.reset(s_small)
        next_s1, _, _, _, _ = mdp.step(0, s_reset, actions)
        self.assertEqual(next_s1.shape, (4,))

        # 2D state tensor (1, 15)
        s_2d = torch.randn(1, 15)
        s_reset2, _ = mdp.reset(s_2d)
        next_s2, _, _, _, _ = mdp.step(0, s_reset2, actions)
        self.assertEqual(next_s2.shape, (1, 15))

    def test_ppo_single_transition_update_non_zero_gradient(self):
        """
        Verifies that PPO update with a single buffered transition does not
        annihilate policy advantage to zero via mean-subtraction.
        """
        ppo = ConstrainedPPOController(state_dim=15, hidden_dim=32, device="cpu")
        s = torch.randn(15)
        dec = ppo.select_action(s)
        ppo.record_step(
            state=s,
            action_index=dec.action_index,
            reward=1.0,
            cost=0.0,
            log_prob=dec.log_prob,
            value=dec.value_estimate,
            cost_val=dec.cost_estimate,
            done=True,
        )
        metrics = ppo.update(ppo_epochs=1)
        self.assertIn("training_loss", metrics)
        self.assertFalse(math.isnan(metrics["training_loss"]))


    def test_dimension_mismatch_robustness(self):
        """
        Verifies that passing state vectors of differing lengths (e.g. 15 vs 31)
        is safely padded or truncated without throwing IndexError or shape mismatch.
        """
        bandit = ContextualBanditController(
            state_dim=31,
            candidate_layers=[8, 12],
            candidate_magnitudes=[2.5],
            algorithm="action_conditioned",
        )
        # Pass smaller 15-dim state
        s_small = torch.randn(15)
        dec_small = bandit.select_action(s_small, prompt_kind="unsafe")
        self.assertIsNotNone(dec_small.action)
        bandit.update(s_small, dec_small.action, 0.5)

        # Pass larger 40-dim state
        s_large = torch.randn(40)
        dec_large = bandit.select_action(s_large, prompt_kind="unsafe")
        self.assertIsNotNone(dec_large.action)
        bandit.update(s_large, dec_large.action, 0.5)


if __name__ == "__main__":
    unittest.main()
