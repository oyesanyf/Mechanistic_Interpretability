#!/usr/bin/env python3
"""
Unit and integration tests for the 7 targeted RL additions:
1. Frozen test mode (--rl_eval_mode frozen_test) with no policy updates and no Part B warm starts.
2. Separate modes: cold_rl, partb_prior_rl, frozen_rl.
3. Temporal learning summary (early vs late reward/regret, cumulative regret, % eps-optimal).
4. Direct counts of RL advantage (exceeds, matches, below raw gain).
5. Oracle / Counterfactual benchmark aggregation.
6. Hardened policy checkpointing and reload with state dimension synchronization (15 vs 31).
7. Dedicated behavioral-success metric for steered generation.
"""

import json
import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn

from deep_noir_rl.bandit_controller import (
    ContextualBanditController,
    SteeringAction,
)
from deep_noir_rl.contrastive_steering import SteeringVector
from deep_noir_rl.controller import (
    AdaptiveSteeringRLController,
    AdaptiveSteeringResult,
)
from deep_noir_rl.evaluator import (
    DeepNoirRLEvaluator,
    PolicyEvaluationSummary,
)
from deep_noir_rl.graduated_rewards import GraduatedRewardBreakdown
from african_safety_full_research_auditor_with_circuit_tracer import (
    CombinedPromptResult,
    CombinedSummary,
    summarize_results,
    parse_args,
    GenerationEval,
    save_summary,
    save_prompt_details,
)


class MockConfig:
    hidden_size = 16
    num_attention_heads = 2
    num_hidden_layers = 4


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Identity()

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.layers = nn.ModuleList([MockLayer() for _ in range(4)])
        self.model = nn.Module()
        self.model.layers = self.layers
        self.lm_head = nn.Linear(16, 50, bias=False)

    def forward(self, input_ids=None, **kwargs):
        batch_size, seq_len = (input_ids.shape if input_ids is not None else (1, 4))
        h = torch.randn(batch_size, seq_len, 16)
        logits = self.lm_head(h)
        class Output:
            pass
        out = Output()
        out.logits = logits
        out.hidden_states = [h for _ in range(5)]
        return out

    def generate(self, input_ids=None, max_new_tokens=24, **kwargs):
        batch_size, seq_len = (input_ids.shape if input_ids is not None else (1, 4))
        new_tokens = torch.full((batch_size, max_new_tokens), 10, dtype=torch.long)
        base = input_ids if input_ids is not None else torch.zeros((batch_size, 4), dtype=torch.long)
        return torch.cat([base, new_tokens], dim=-1)


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": torch.tensor([[10, 20, 30, 40]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}

    def encode(self, text, add_special_tokens=False):
        return [10, 20]

    def decode(self, token_ids, skip_special_tokens=True):
        return "I am sorry, I cannot fulfill this request."


class TestTargetedRLAdditions(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        self.layers = list(self.model.layers)
        self.device = "cpu"

    def _setup_cached_directions(self, ctrl):
        cal_vec = torch.randn(16)
        for l in ctrl.candidate_layers:
            ctrl.steering_manager.calibrated_directions[l] = SteeringVector(
                layer_idx=l,
                raw_vector=cal_vec,
                unit_vector=cal_vec / torch.norm(cal_vec),
                vector_norm=1.0,
                num_safe_samples=1,
                num_harmful_samples=1,
                language="English",
            )

    def test_frozen_test_and_modes(self):
        """Req 1 & 2: In frozen_test or frozen_rl, policy updates are disallowed and Part B is isolated."""
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            eval_mode="frozen_test",
            rl_mode="frozen_rl",
            rich_state=False,
        )
        self._setup_cached_directions(ctrl)
        self.assertEqual(ctrl.eval_mode, "frozen_test")
        self.assertEqual(ctrl.rl_mode, "frozen_rl")

        # Mock an awakening result
        class MockAwakening:
            target_layer = 1
            safety_awakening_gain = 0.25
            mutation_l2 = 2.5
            mutation_vector = torch.randn(16)

        initial_pulls = list(ctrl.bandit.pull_counts)

        inputs = self.tokenizer("test unsafe prompt")
        res = ctrl.steer_and_evaluate(
            prompt_text="test unsafe prompt",
            inputs=inputs,
            refusal_ids=[10, 20],
            language_name="English",
            prompt_kind="unsafe",
            update_policy=True,  # Even if caller passes True, frozen_test should disallow
            awakening_results=[MockAwakening()],
            best_awakening=MockAwakening(),
        )

        # Pull counts should not have increased because eval_mode is frozen_test
        current_pulls = list(ctrl.bandit.pull_counts)
        self.assertEqual(current_pulls, initial_pulls, "Arm pull count updated during frozen_test!")

    def test_cold_rl_isolation(self):
        """Req 2: cold_rl does not warm-start from Part B awakening."""
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            eval_mode="train",
            rl_mode="cold_rl",
            rich_state=False,
        )
        self._setup_cached_directions(ctrl)

        class MockAwakening:
            target_layer = 1
            safety_awakening_gain = 0.50
            mutation_l2 = 2.5
            mutation_vector = torch.randn(16)

        inputs = self.tokenizer("test unsafe prompt")
        res = ctrl.steer_and_evaluate(
            prompt_text="test unsafe prompt",
            inputs=inputs,
            refusal_ids=[10, 20],
            language_name="English",
            prompt_kind="unsafe",
            update_policy=True,
            awakening_results=[MockAwakening()],
            best_awakening=MockAwakening(),
        )
        # In cold_rl, prompt_awakening_vectors must be empty
        self.assertEqual(len(ctrl.steering_manager.prompt_awakening_vectors), 0)

    def test_temporal_learning_summary(self):
        """Req 3: Temporal learning metrics (early vs late reward, regret, eps-optimal)."""
        bandit = ContextualBanditController(
            state_dim=15,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            algorithm="linucb",
        )
        # Record 10 steps of mock trajectory
        for step in range(10):
            reward = 0.1 * step
            bandit.trajectory.append({
                "step": step,
                "observed_reward": reward,
                "instantaneous_regret": max(0.0, 1.0 - reward),
                "is_epsilon_optimal": (1.0 - reward) <= 0.05,
                "action_id": step % len(bandit.actions),
            })
        summary = bandit.get_temporal_learning_summary(epsilon=0.05)
        self.assertIn("early_mean_reward", summary)
        self.assertIn("late_mean_reward", summary)
        self.assertGreater(summary["late_mean_reward"], summary["early_mean_reward"])
        self.assertLess(summary["late_mean_regret"], summary["early_mean_regret"])
        self.assertIn("pct_epsilon_optimal", summary)
        self.assertIn("action_frequency_shifts", summary)

    def test_rl_advantage_counts_and_summary(self):
        """Req 4: Direct counts and percentages of rl_exceeds_raw_gain, matches, below."""
        mock_results = []
        # Item 1: RL exceeds Part B by > 1e-5
        r1 = CombinedPromptResult(
            language="Yoruba", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=0, category="weapons", prompt_text="p1",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_1", raw_intervention_gain=0.10, rl_selected_gain=0.20,
            rl_gain_over_non_rl=0.10, rl_reward=0.8, rl_steered_prob=0.30, rl_is_safe=True,
            rl_was_rolled_back=False,
            unsteered_behavior_label="unclear", unsteered_behavior_refusal=False,
            steered_behavior_label="refusal", steered_behavior_refusal=True,
        )
        # Item 2: RL matches Part B within 1e-5
        r2 = CombinedPromptResult(
            language="Yoruba", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=1, category="drugs", prompt_text="p2",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_1", raw_intervention_gain=0.15, rl_selected_gain=0.15,
            rl_gain_over_non_rl=0.00, rl_reward=0.7, rl_steered_prob=0.25, rl_is_safe=True,
            rl_was_rolled_back=False,
            unsteered_behavior_label="unclear", unsteered_behavior_refusal=False,
            steered_behavior_label="unclear", steered_behavior_refusal=False,
        )
        # Item 3: RL below Part B by < -1e-5
        r3 = CombinedPromptResult(
            language="Yoruba", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=2, category="cyber", prompt_text="p3",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_0", raw_intervention_gain=0.25, rl_selected_gain=0.05,
            rl_gain_over_non_rl=-0.20, rl_reward=0.2, rl_steered_prob=0.15, rl_is_safe=True,
            rl_was_rolled_back=True,
            unsteered_behavior_label="unclear", unsteered_behavior_refusal=False,
            steered_behavior_label="unclear", steered_behavior_refusal=False,
        )
        mock_results = [r1, r2, r3]
        summaries = summarize_results(mock_results, [0, 1])
        self.assertEqual(len(summaries), 1)
        s = summaries[0]

        # Verify counts and percentages
        self.assertEqual(s.rl_exceeds_raw_gain_count, 1)
        self.assertEqual(s.rl_matches_raw_gain_count, 1)
        self.assertEqual(s.rl_below_raw_gain_count, 1)
        self.assertAlmostEqual(s.rl_exceeds_raw_gain_pct, 33.333, places=1)
        self.assertAlmostEqual(s.rl_matches_raw_gain_pct, 33.333, places=1)
        self.assertAlmostEqual(s.rl_below_raw_gain_pct, 33.333, places=1)

        # Req 7: Verify behavioral refusal rates
        self.assertAlmostEqual(s.unsteered_behavioral_refusal_rate, 0.0)
        self.assertAlmostEqual(s.steered_behavioral_refusal_rate, 33.333, places=1)
        self.assertAlmostEqual(s.behavioral_refusal_gain, 33.333, places=1)

    def test_oracle_counterfactual_aggregation(self):
        """Req 5: Oracle and counterfactual benchmark aggregation."""
        r1 = CombinedPromptResult(
            language="Swahili", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=0, category="weapons", prompt_text="p1",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_1",
            has_counterfactual_eval=True,
            oracle_action_name="Action_1",
            oracle_reward=0.9,
            oracle_instantaneous_regret=0.02,
            oracle_is_epsilon_optimal=True,
            oracle_action_match=True,
        )
        r2 = CombinedPromptResult(
            language="Swahili", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=1, category="drugs", prompt_text="p2",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_2",
            has_counterfactual_eval=True,
            oracle_action_name="Action_3",
            oracle_reward=0.8,
            oracle_instantaneous_regret=0.10,
            oracle_is_epsilon_optimal=False,
            oracle_action_match=False,
        )
        summaries = summarize_results([r1, r2], [0, 1])
        s = summaries[0]
        self.assertEqual(s.n_counterfactual_evals, 2)
        self.assertAlmostEqual(s.mean_oracle_reward, 0.85)
        self.assertAlmostEqual(s.mean_oracle_regret, 0.06)
        self.assertAlmostEqual(s.oracle_action_accuracy, 50.0)
        self.assertAlmostEqual(s.oracle_pct_epsilon_optimal, 50.0)

    def test_policy_checkpoint_save_load_dimension_sync(self):
        """Req 6: Policy checkpoint serialization and state dimension synchronization (15 vs 31)."""
        # Create a controller with rich_state=True (31 dimensions)
        ctrl_31 = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            rich_state=True,
        )
        self.assertEqual(ctrl_31.state_extractor.state_dim, 31)

        # Update bandit with a mock state
        s_vec = torch.randn(31)
        act = ctrl_31.bandit.actions[1]
        ctrl_31.bandit.update(s_vec, act, reward=0.85)
        ctrl_31.cumulative_regret = 0.42

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_policy.json")
            save_meta = ctrl_31.save_policy(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            with open(ckpt_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["state_dim"], 31)

            # Create a new controller initialized with rich_state=False (15 dimensions)
            ctrl_15 = AdaptiveSteeringRLController(
                model=self.model,
                tokenizer=self.tokenizer,
                layers=self.layers,
                device=self.device,
                candidate_layers=[1, 2],
                candidate_magnitudes=[1.0, 2.5],
                rich_state=False,
            )
            self.assertEqual(ctrl_15.state_extractor.state_dim, 15)

            # Load the 31-dim checkpoint
            load_meta = ctrl_15.load_policy(ckpt_path, freeze=True)

            # Verify state dimension synchronization!
            self.assertEqual(ctrl_15.state_extractor.state_dim, 31)
            self.assertTrue(ctrl_15.state_extractor.rich_state)
            self.assertEqual(ctrl_15.bandit.state_dim, 31)
            self.assertEqual(ctrl_15.eval_mode, "frozen_test")
            self.assertEqual(ctrl_15.rl_mode, "frozen_rl")
            self.assertAlmostEqual(ctrl_15.cumulative_regret, 0.42)

    def test_cli_eval_mode_args(self):
        """Req 1 & 2: CLI arguments --rl_eval_mode and --eval_mode parsing."""
        from african_safety_full_research_auditor_with_circuit_tracer import parse_args
        test_argv = ["script.py", "--rl_eval_mode", "frozen_test", "--rl_mode", "cold_rl"]
        sys_argv_backup = sys.argv
        try:
            sys.argv = test_argv
            parsed = parse_args()
            self.assertEqual(parsed.rl_eval_mode, "frozen_test")
            self.assertEqual(parsed.eval_mode, "frozen_test")
            self.assertEqual(parsed.rl_mode, "cold_rl")
        finally:
            sys.argv = sys_argv_backup

    def test_evaluator_temporal_metrics_and_oracle_summarization(self):
        """Req 3, 5, 7: Evaluator temporal learning dynamics, oracle regret calculation, and behavioral refusal."""
        evaluator = DeepNoirRLEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
        )

        act1 = SteeringAction(action_id=1, name="Action_1", layer_idx=1, magnitude=1.0)
        act2 = SteeringAction(action_id=2, name="Action_2", layer_idx=2, magnitude=2.5)

        rb1 = GraduatedRewardBreakdown(total_reward=0.8, s_acc=0.8, s_inj=0.0, s_cap=0.0, s_cost=0.0, is_safe=True)
        rb2 = GraduatedRewardBreakdown(total_reward=0.2, s_acc=0.2, s_inj=0.0, s_cap=0.0, s_cost=0.0, is_safe=False)
        rb3 = GraduatedRewardBreakdown(total_reward=0.9, s_acc=0.9, s_inj=0.0, s_cap=0.0, s_cost=0.0, is_safe=True)

        r1 = AdaptiveSteeringResult(
            prompt_text="p1", language="Yoruba", prompt_kind="unsafe", scaffold="baseline",
            clean_refusal_prob=0.1, steered_refusal_prob=0.8, refusal_gain=0.7,
            chosen_action=act1, reward_breakdown=rb1, was_rolled_back=False,
            instantaneous_regret=0.02, oracle_action_name="Action_1", oracle_reward=0.82,
            is_behavior_refusal=True,
        )
        r2 = AdaptiveSteeringResult(
            prompt_text="p2", language="Yoruba", prompt_kind="unsafe", scaffold="baseline",
            clean_refusal_prob=0.1, steered_refusal_prob=0.2, refusal_gain=0.1,
            chosen_action=act2, reward_breakdown=rb2, was_rolled_back=True,
            instantaneous_regret=0.60, oracle_action_name="Action_1", oracle_reward=0.80,
            is_behavior_refusal=False,
        )
        r3 = AdaptiveSteeringResult(
            prompt_text="p3", language="Yoruba", prompt_kind="benign", scaffold="baseline",
            clean_refusal_prob=0.05, steered_refusal_prob=0.05, refusal_gain=0.0,
            chosen_action=act1, reward_breakdown=rb3, was_rolled_back=False,
            instantaneous_regret=0.01, oracle_action_name="Action_1", oracle_reward=0.91,
            is_behavior_refusal=False,
        )

        results = [r1, r2, r3]
        temp_metrics = evaluator.compute_temporal_learning_metrics(results, epsilon=0.05)
        self.assertIn("late_rollback_rate", temp_metrics)
        self.assertIn("early_rollback_rate", temp_metrics)
        self.assertIn("action_frequency_shifts", temp_metrics)
        self.assertAlmostEqual(temp_metrics["cumulative_regret"], 0.63, places=2)

        summary = evaluator._summarize_policy_results(
            "bandit", results, elapsed=1.5, compute_forward_passes=3, unsteered_refusal_rate=10.0
        )
        # Ensure mean_oracle_regret is computed over oracle regrets (0.02 + 0.60 + 0.01) / 3 = 0.21, NOT early_mean_regret!
        expected_oracle_regret = round((0.02 + 0.60 + 0.01) / 3.0, 5)
        self.assertAlmostEqual(summary.mean_oracle_regret, expected_oracle_regret, places=4)
        self.assertAlmostEqual(summary.unsteered_behavioral_refusal_rate, 10.0)
        self.assertAlmostEqual(summary.steered_behavioral_refusal_rate, 50.0)  # 1 of 2 unsafe
        self.assertAlmostEqual(summary.behavioral_refusal_gain, 40.0)

    def test_validation_mode_disallows_updates(self):
        """Req 1: Validation mode disallows policy updates."""
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            eval_mode="validation",
            rl_mode="partb_prior_rl",
            rich_state=False,
        )
        self._setup_cached_directions(ctrl)
        initial_pulls = list(ctrl.bandit.pull_counts)
        inputs = self.tokenizer("test unsafe prompt")
        ctrl.steer_and_evaluate(
            prompt_text="test unsafe prompt",
            inputs=inputs,
            refusal_ids=[10, 20],
            language_name="English",
            prompt_kind="unsafe",
            update_policy=True,
        )
        current_pulls = list(ctrl.bandit.pull_counts)
        self.assertEqual(current_pulls, initial_pulls, "Arm pull count updated during validation mode!")

    def test_zulu_refusal_behavior_detection(self):
        """Req 7: Zulu refusal markers trigger behavior refusal classification."""
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
            eval_mode="train",
            rl_mode="partb_prior_rl",
            behavior_reward=True,
        )
        self._setup_cached_directions(ctrl)

        orig_decode = self.tokenizer.decode
        self.tokenizer.decode = lambda tokens, **kwargs: "Ngiyaxolisa, angikwazi ukukusiza ngalokhu."
        try:
            inputs = self.tokenizer("test unsafe prompt")
            res = ctrl.steer_and_evaluate(
                prompt_text="test unsafe prompt",
                inputs=inputs,
                refusal_ids=[10, 20],
                language_name="Zulu",
                prompt_kind="unsafe",
                update_policy=False,
            )
            self.assertEqual(res.steered_behavior_label, "refusal")
            self.assertTrue(res.is_behavior_refusal)
        finally:
            self.tokenizer.decode = orig_decode

    def test_load_policy_candidate_path_without_extension(self):
        """Req 6: Policy checkpoint loading resolves candidate paths without .json."""
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=[1, 2],
            candidate_magnitudes=[1.0, 2.5],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "saved_policy.json")
            ctrl.save_policy(save_path)

            ctrl2 = AdaptiveSteeringRLController(
                model=self.model,
                tokenizer=self.tokenizer,
                layers=self.layers,
                device=self.device,
                candidate_layers=[1, 2],
                candidate_magnitudes=[1.0, 2.5],
            )
            load_target = os.path.join(tmp_dir, "saved_policy")
            base_p, ext_p = os.path.splitext(load_target)
            cand_paths = []
            if ext_p:
                cand_paths.append(f"{base_p}_English{ext_p}")
            cand_paths.append(f"{load_target}_English.json")
            cand_paths.append(f"{load_target}_English")
            if not load_target.endswith(".json"):
                cand_paths.append(f"{load_target}.json")
            cand_paths.append(load_target)

            resolved = None
            for cp in cand_paths:
                if os.path.exists(cp):
                    ctrl2.load_policy(cp, freeze=True)
                    resolved = cp
                    break
            self.assertIsNotNone(resolved)
            self.assertTrue(resolved.endswith(".json"))

    def test_combined_summary_and_csv_export(self):
        """Req 1, 2, 3: Mode propagation in CombinedSummary and CSV export column integrity."""
        from pathlib import Path
        import csv
        r = CombinedPromptResult(
            language="Zulu", resource="low", family="Niger-Congo", scaffold="baseline",
            seed=0, prompt_kind="unsafe", prompt_id=0, category="weapons", prompt_text="p1",
            refusal_token_ids=[1], refusal_token_texts=["No"], refusal_pieces_per_start=1.0,
            peak_rpd=0.5, peak_rpd_layer=1, gini_rpd=0.2, fragility_signal_strength="medium",
            fragility_label="medium_fragility", mean_clean_refusal_prob=0.1, max_clean_refusal_prob=0.1,
            peak_entropy_increase=0.1, peak_english_refusal_increase=0.0, layer_results=[],
            awakening_results=[], best_awakening=None, generation_eval=GenerationEval(),
            warning_flags=[], audit_trace=None,
            rl_action_name="Action_1", raw_intervention_gain=0.10, rl_selected_gain=0.20,
            rl_gain_over_non_rl=0.10, rl_reward=0.8, rl_steered_prob=0.30, rl_is_safe=True,
            rl_was_rolled_back=False,
            unsteered_behavior_label="unclear", unsteered_behavior_refusal=False,
            steered_behavior_label="refusal", steered_behavior_refusal=True,
            has_counterfactual_eval=True, oracle_action_name="Action_1", oracle_reward=0.85,
            oracle_instantaneous_regret=0.05, oracle_is_epsilon_optimal=True, oracle_action_match=True,
            rl_mode="cold_rl", rl_eval_mode="frozen_test",
        )
        summaries = summarize_results([r], [0, 1])
        s = summaries[0]
        self.assertEqual(s.rl_mode, "cold_rl")
        self.assertEqual(s.rl_eval_mode, "frozen_test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_csv = Path(tmp_dir) / "summary.csv"
            save_summary(summaries, summary_csv)
            self.assertTrue(summary_csv.exists())

            with summary_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                row = next(reader)
                self.assertIn("early_mean_regret", header)
                self.assertIn("late_mean_regret", header)
                self.assertIn("cumulative_regret", header)
                self.assertIn("pct_epsilon_optimal", header)
                self.assertIn("rl_mode", header)
                self.assertIn("rl_eval_mode", header)
                self.assertIn("action_frequency_shifts", header)
                idx_rl_mode = header.index("rl_mode")
                idx_rl_eval_mode = header.index("rl_eval_mode")
                self.assertEqual(row[idx_rl_mode], "cold_rl")
                self.assertEqual(row[idx_rl_eval_mode], "frozen_test")

            prompt_csv = Path(tmp_dir) / "prompts.csv"
            save_prompt_details([r], prompt_csv)
            self.assertTrue(prompt_csv.exists())
            with prompt_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                row = next(reader)
                self.assertIn("rl_mode", header)
                self.assertIn("rl_eval_mode", header)
                self.assertEqual(row[header.index("rl_mode")], "cold_rl")
                self.assertEqual(row[header.index("rl_eval_mode")], "frozen_test")


if __name__ == "__main__":
    unittest.main()
