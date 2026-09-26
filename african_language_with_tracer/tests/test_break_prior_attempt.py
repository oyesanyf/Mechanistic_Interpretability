import unittest
import torch
from deep_noir_rl.contrastive_steering import ContrastiveSteeringManager, SteeringVector
from deep_noir_rl.bandit_controller import ContextualBanditController, SteeringAction


class TestBreakPriorAttempt(unittest.TestCase):
    def test_break_contrastive_cache_corruption(self):
        """
        Break Test 2: Proves that register_awakening_direction permanently overwrites
        and destroys calibrated contrastive steering directions.
        """
        class MockConfig:
            hidden_size = 4
            num_attention_heads = 1
        class MockLayer:
            def register_forward_hook(self, fn):
                class Handle:
                    def remove(self): pass
                return Handle()
        class MockModel:
            config = MockConfig()
            def parameters(self):
                return iter([torch.zeros(1)])
        mgr = ContrastiveSteeringManager(model=MockModel(), tokenizer=None, layers=[MockLayer() for _ in range(15)], device="cpu")
        # Simulate calibration: calibrated contrastive direction at layer 8
        calibrated_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mgr.cached_directions[8] = SteeringVector(
            layer_idx=8,
            raw_vector=calibrated_vec,
            unit_vector=calibrated_vec / torch.norm(calibrated_vec),
            vector_norm=1.0,
            num_safe_samples=1,
            num_harmful_samples=1,
            language="Yoruba",
        )

        # Prompt 0 arrives: Part B finds awakening mutation vector at layer 8
        prompt0_mutation = torch.tensor([0.0, 5.0, 0.0, 0.0])
        mgr.register_awakening_direction(layer_idx=8, vector=prompt0_mutation, gain=0.0074, language="Yoruba")

        # End of Prompt 0: controller calls clear_prompt_awakening_vectors()
        mgr.clear_prompt_awakening_vectors()

        # Prompt 1 arrives: Prompt 1 has NO awakening at layer 8.
        # It should fall back to the calibrated contrastive direction (calibrated_vec).
        with mgr.apply_steering(layer_idx=8, magnitude=1.0) as hook:
            # Check what direction is being used
            sv = mgr.prompt_awakening_vectors.get(8) or mgr.cached_directions.get(8)
            # If cached_directions[8] was overwritten with prompt0_mutation, this assert will fail!
            self.assertTrue(
                torch.equal(sv.raw_vector, calibrated_vec),
                f"Cache corruption! Expected calibrated vector {calibrated_vec.tolist()}, but got {sv.raw_vector.tolist()} from Prompt 0!"
            )

    def test_break_norm_mismatch_scaling(self):
        """
        Break Test 1: Proves that if Part B finds a mutation vector with norm 4.7,
        the prior attempt's abs(mag - norm) < 0.05 check fails and scales the vector.
        """
        class MockConfig:
            hidden_size = 4
            num_attention_heads = 1
        captured_delta = []
        class MockLayer:
            def register_forward_hook(self, fn):
                class Handle:
                    def remove(self): pass
                self.fn = fn
                return Handle()
        mock_layer = MockLayer()
        class MockModel:
            config = MockConfig()
            def parameters(self):
                return iter([torch.zeros(1)])
        mgr = ContrastiveSteeringManager(model=MockModel(), tokenizer=None, layers=[mock_layer]*15, device="cpu")
        # Part B optimizes and norm converges to 4.7
        mutation_4_7 = torch.tensor([0.0, 4.7, 0.0, 0.0])
        mgr.register_awakening_direction(layer_idx=8, vector=mutation_4_7, gain=0.0074, language="Yoruba")

        # Bandit selects arm L8_mag5.0_full (magnitude = 5.0) representing the verified arm
        with mgr.apply_steering(layer_idx=8, magnitude=5.0):
            # Trigger hook with dummy residual activation
            act = torch.zeros(1, 1, 4)
            out = mock_layer.fn(mock_layer, (act,), act)
            injected_delta = out - act
            injected_norm = torch.norm(injected_delta).item()

        # In prior attempt, injected_norm is 5.0 (re-scaled), NOT 4.7 (the verified mutation vector)!
        self.assertAlmostEqual(
            injected_norm,
            4.7,
            places=2,
            msg=f"Expected verified vector norm 4.7 to be preserved, but got re-scaled norm {injected_norm:.4f}!"
        )

    def test_break_ghost_state_bandit_leakage(self):
        """
        Break Test 3: Proves that last_verified_arm_id leaks from prompt 0 to prompt 1
        when prompt 1 has NO verified awakening.
        """
        bandit = ContextualBanditController(
            state_dim=4,
            candidate_layers=[8, 12],
            candidate_magnitudes=[5.0, 12.0],
            include_head_subsets=False,
            exploration_c=1.25,
        )
        s0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # Prompt 0: warm-start arm for layer 8
        warm_idx = bandit.warm_start_arm(layer_idx=8, magnitude=5.0, reward=0.5, state_vector=s0)
        self.assertIsNotNone(warm_idx)

        # Prompt 1 arrives: verified_gain = 0.0, verified_arm_id = None
        s1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        dec = bandit.select_action(
            state_vector=s1,
            prompt_kind="unsafe",
            preferred_layer=None,
            verified_arm_id=None,
            verified_gain=0.0,
        )
        # In prior attempt, dec treats warm_idx as verified because of last_verified_arm_id!
        # If ghost state leaks, last_verified_arm_id is still active.
        self.assertIsNone(
            bandit.last_verified_arm_id,
            "last_verified_arm_id must not leak into subsequent prompts where verified_gain is 0.0!"
        )

    def test_break_unverified_pulled_arm_overrides_verified(self):
        """
        Break Test 4: Proves that if an unverified arm was pulled once before,
        it receives full c=1.25 exploration bonus and can overtake a verified arm on unsafe prompts.
        """
        bandit = ContextualBanditController(
            state_dim=4,
            candidate_layers=[8, 12],
            candidate_magnitudes=[5.0, 12.0],
            include_head_subsets=False,
            exploration_c=1.25,
        )
        # Arm 1: L8_mag5.0_full (verified)
        # Arm 2: L8_mag12.0_full (unverified, but was pulled on a previous prompt)
        s0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        bandit.update(s0, bandit.actions[2], reward=0.0) # Arm 2 pulled once, zero reward
        self.assertEqual(bandit.pull_counts[2], 1)

        # Warm start arm 1 as verified with modest reward 0.20
        warm_idx = bandit.warm_start_arm(layer_idx=8, magnitude=5.0, reward=0.20, state_vector=s0)
        self.assertEqual(warm_idx, 1)

        s_eval = s0.clone()
        dec = bandit.select_action(
            state_vector=s_eval,
            prompt_kind="unsafe",
            preferred_layer=8,
            verified_arm_id=1,
            verified_gain=0.005,
        )
        # On an unsafe prompt with verified arm available, safety-constrained exploration
        # MUST pick the verified arm, NOT an unverified high-magnitude arm that was pulled once.
        self.assertEqual(
            dec.action.name,
            "L8_mag5.0_full",
            f"Bandit must pick verified arm L8_mag5.0_full, but unverified pulled arm got chosen: {dec.action.name}"
        )

    def test_break_counterfactual_zero_gain_bug(self):
        """
        Break Test 5: Proves that in evaluate_counterfactual_actions, candidate steering actions
        calculate real refusal gain (steered_refusal - clean_refusal > 0) rather than passing
        ref_prob for both clean and steered refusal (which previously made gain identically 0).
        """
        from deep_noir_rl.controller import AdaptiveSteeringRLController

        class MockConfig:
            hidden_size = 4
            num_attention_heads = 1
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        class MockModel:
            config = MockConfig()
            def __init__(self):
                self.call_count = 0
            def parameters(self):
                return iter([torch.zeros(1)])
            def __call__(self, *args, **kwargs):
                self.call_count += 1
                # Return distinct logits: baseline has refusal prob ~0.05, steered has ~0.80
                if self.call_count == 1:
                    # Clean forward pass
                    logits = torch.tensor([[[0.0, 0.0, -3.0, 3.0]]])  # token 2 (refusal) low
                else:
                    # Steered forward pass
                    logits = torch.tensor([[[0.0, 0.0, 3.0, -3.0]]])  # token 2 (refusal) high
                return MockOutput(logits)

        mock_model = MockModel()
        mock_model.lm_head = torch.nn.Linear(4, 4)
        mock_model.model = torch.nn.Module()
        mock_model.model.norm = torch.nn.Identity()

        class MockLayer:
            def register_forward_hook(self, fn):
                class Handle:
                    def remove(self): pass
                return Handle()

        ctrl = AdaptiveSteeringRLController(
            model=mock_model,
            tokenizer=None,
            layers=[MockLayer() for _ in range(15)],
            candidate_layers=[8],
            candidate_magnitudes=[2.5],
            device="cpu",
        )
        ctrl.candidate_layers = [8]
        # Calibrate a dummy vector so apply_steering works
        ctrl.steering_manager.cached_directions[8] = SteeringVector(
            layer_idx=8,
            raw_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            unit_vector=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            vector_norm=1.0,
            num_safe_samples=1,
            num_harmful_samples=1,
        )

        dummy_inp = {"input_ids": torch.tensor([[1, 2]])}
        res = ctrl.evaluate_counterfactual_actions(
            prompt_text="test prompt",
            inputs=dummy_inp,
            refusal_ids=[2],
            language_name="English",
            prompt_kind="unsafe",
            epsilon=0.05,
        )
        evals = res["candidate_evaluations"]
        # Find steered action (action_id != 0)
        steered_acts = [e for aid, e in evals.items() if aid != 0]
        self.assertGreater(len(steered_acts), 0)
        for sa in steered_acts:
            self.assertGreater(
                sa["refusal_gain"],
                0.10,
                f"Refusal gain in counterfactual eval was {sa['refusal_gain']}; prior attempt had 0.0 due to passing ref_prob as clean refusal!"
            )
        self.assertIn("action_selection_accuracy", res)

    def test_break_cross_validated_directions_exists(self):
        """
        Break Test 6: Proves that compute_cross_validated_directions exists on
        ContrastiveSteeringManager and computes consensus directions across folds.
        """
        class MockConfig:
            hidden_size = 4
            num_attention_heads = 1
        class MockLayer:
            def register_forward_hook(self, fn):
                class Handle:
                    def remove(self): pass
                return Handle()
        class MockModel:
            config = MockConfig()
            def parameters(self):
                return iter([torch.zeros(1)])
        mgr = ContrastiveSteeringManager(model=MockModel(), tokenizer=None, layers=[MockLayer() for _ in range(15)], device="cpu")
        self.assertTrue(
            hasattr(mgr, "compute_cross_validated_directions"),
            "ContrastiveSteeringManager must implement compute_cross_validated_directions (Requirement 20)!"
        )
        self.assertTrue(
            hasattr(mgr, "compute_matched_pair_directions"),
            "ContrastiveSteeringManager must implement compute_matched_pair_directions (Requirement 20)!"
        )

    def test_break_bandit_load_policy_actions_desync(self):
        """
        Break Test 7: Proves that load_policy properly restores expanded actions
        when loading a checkpoint saved with expanded_action_space=True.
        """
        import tempfile
        bandit_exp = ContextualBanditController(
            state_dim=4,
            candidate_layers=[8, 12],
            candidate_magnitudes=[1.0, 5.0],
            expanded_action_space=True,
        )
        exp_action_count = len(bandit_exp.actions)
        self.assertGreater(exp_action_count, 5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            bandit_exp.save_policy(tmp_path)

            # Fresh bandit initialized with default expanded_action_space=False
            bandit_loaded = ContextualBanditController(
                state_dim=4,
                candidate_layers=[8, 12],
                candidate_magnitudes=[1.0, 5.0],
                expanded_action_space=False,
            )
            default_action_count = len(bandit_loaded.actions)
            self.assertNotEqual(default_action_count, exp_action_count)

            bandit_loaded.load_policy(tmp_path)
            self.assertEqual(
                len(bandit_loaded.actions),
                exp_action_count,
                "Bandit load_policy must reconstruct actions matching the checkpoint!"
            )
            self.assertEqual(len(bandit_loaded.A), len(bandit_loaded.actions))
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_break_ablation_fallback_covers_all_heads(self):
        """
        Break Test 8: Proves that _attribute_heads_ablation_fallback searches
        across all heads, not capping at 4 heads.
        """
        from deep_noir_rl.gradient_attribution import GradientActivationAttributor
        class MockConfig:
            hidden_size = 32
            num_attention_heads = 8
        class MockOutput:
            def __init__(self):
                self.logits = torch.zeros(1, 1, 10)
        class MockModel:
            config = MockConfig()
            def __call__(self, *args, **kwargs):
                return MockOutput()
            def parameters(self):
                return iter([torch.zeros(1)])
        class MockLayer:
            def __init__(self):
                class MockAttn:
                    def register_forward_hook(self, fn):
                        class Handle:
                            def remove(self): pass
                        return Handle()
                self.self_attn = MockAttn()

        attributor = GradientActivationAttributor(model=MockModel(), layers=[MockLayer()], device="cpu")
        self.assertEqual(attributor.num_heads, 8)
        # Call fallback on layer 0 requesting top 8
        scores = attributor._attribute_heads_ablation_fallback(
            inputs={"input_ids": torch.tensor([[1]])},
            refusal_ids=[2],
            layer_indices=[0],
            top_k=8,
        )
        # Must evaluate all 8 heads
        self.assertEqual(len(scores), 8, f"Expected 8 head scores evaluated, but got {len(scores)}!")


if __name__ == "__main__":
    unittest.main()

