import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from african_safety_full_research_auditor_with_circuit_tracer import (
    build_prompt,
    calibration_prompts_for_language,
    token_ids_for_starts,
    ensure_tokenizer_padding,
    SAFETY_INTENT_CATEGORIES,
    BENIGN_INTENT_CATEGORIES,
    LANGUAGES,
    SafetyAwakener,
    AwakeningResult,
)
from deep_noir_rl.controller import AdaptiveSteeringRLController
from deep_noir_rl.bandit_controller import SteeringAction
from deep_noir_rl.graduated_rewards import GraduatedRewardEvaluator


class TestPartBToCPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        ensure_tokenizer_padding(cls.tokenizer)
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_id,
            torch_dtype=torch.float16 if cls.device == "cuda" else torch.float32,
        ).to(cls.device)
        cls.model.eval()
        cls.layers = cls.model.model.layers
        cls.target_layers = [8, 12]

    def test_calibration_prompts_distinct(self):
        """Verifies that safe_cal (benign) and harmful_cal (unsafe) are distinct and yield non-zero vectors."""
        yoruba = next(l for l in LANGUAGES if l["name"] == "Yoruba")
        safe_cal = calibration_prompts_for_language(yoruba, 4, "baseline", prompt_kind="benign")
        harmful_cal = [build_prompt(yoruba, cat, i, "unsafe", "baseline") for i, cat in enumerate(SAFETY_INTENT_CATEGORIES[:4])]

        self.assertNotEqual(safe_cal, harmful_cal)
        # Check that safe_cal uses benign intent and harmful_cal uses unsafe intent
        self.assertIn("benign", safe_cal[0].lower())
        self.assertIn("unsafe", harmful_cal[0].lower())

        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            candidate_layers=self.target_layers,
        )
        vecs = ctrl.calibrate(safe_cal, harmful_cal, run_full_deep_noir=False)
        self.assertIn(8, vecs)
        self.assertGreater(vecs[8].vector_norm, 1e-3, "Contrastive vector norm must be non-zero after distinct calibration.")

    def test_part_b_vector_integration_into_part_c(self):
        """
        Concrete test:
        1. Run Part B on Yoruba prompt to find a positive refusal awakening direction at layer 8.
        2. Feed Part B result into Part C.
        3. Verify Part C achieves approximately the same refusal gain when applying steering at layer 8.
        4. Verify that No Steering receives an inaction penalty when verified gain was available.
        """
        yoruba = next(l for l in LANGUAGES if l["name"] == "Yoruba")
        refusal_starts = yoruba["refusal_starts"] + yoruba["safe_starts"]
        refusal_ids, _, _ = token_ids_for_starts(self.tokenizer, refusal_starts)

        prompt_text = build_prompt(yoruba, SAFETY_INTENT_CATEGORIES[0], 0, "unsafe", "baseline")
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # 1. Run Part B SafetyAwakener at Layer 8
        d_model = self.model.config.hidden_size
        awakener = SafetyAwakener(self.model, self.layers, layer_idx=8, device=self.device, d_model=d_model, mutation_scale=5.0, max_norm=5.0)
        aw_res = awakener.optimize(inputs=inputs, refusal_ids=refusal_ids, steps=6, lr=0.1, l1_lambda=1e-4, l2_lambda=1e-4, topk=5)

        self.assertGreater(aw_res.safety_awakening_gain, 0.0, "Part B must find a positive awakening gain.")
        self.assertGreater(aw_res.mutation_l2, 0.0)

        # 2. Initialize Part C controller
        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            policy_type="bandit",
            candidate_layers=self.target_layers,
            candidate_magnitudes=[5.0],
            exploration_c=0.25,
        )

        # 3. User's concrete diagnostic test: Forced action bypass
        forced_act = SteeringAction(action_id=1, name="L8_mag5.0_full", layer_idx=8, magnitude=5.0, is_no_op=False)
        res_forced = ctrl.steer_and_evaluate(
            prompt_text=prompt_text,
            inputs=inputs,
            refusal_ids=refusal_ids,
            language_name="Yoruba",
            prompt_kind="unsafe",
            scaffold_name="baseline",
            awakening_results=[aw_res],
            best_awakening=aw_res,
            forced_action=forced_act,
        )
        self.assertEqual(res_forced.chosen_action.name, "L8_mag5.0_full")
        self.assertAlmostEqual(res_forced.refusal_gain, aw_res.safety_awakening_gain, delta=0.002,
                               msg=f"Forced action Part C gain ({res_forced.refusal_gain:+.6f}) must match Part B gain ({aw_res.safety_awakening_gain:+.6f})")

        # 4. Autonomous RL Action Selection with Part B integration
        ctrl2 = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            policy_type="bandit",
            candidate_layers=self.target_layers,
            candidate_magnitudes=[5.0],
            exploration_c=0.25,
        )
        rl_res = ctrl2.steer_and_evaluate(
            prompt_text=prompt_text,
            inputs=inputs,
            refusal_ids=refusal_ids,
            language_name="Yoruba",
            prompt_kind="unsafe",
            scaffold_name="baseline",
            awakening_results=[aw_res],
            best_awakening=aw_res,
        )

        # RL controller must exploit Part B intervention (NOT choose No Steering!)
        self.assertNotEqual(rl_res.chosen_action.name, "No Steering", "Bandit must not choose No Steering when verified beneficial awakening is available.")
        self.assertEqual(rl_res.chosen_action.name, "L8_mag5.0_full")
        self.assertAlmostEqual(rl_res.refusal_gain, aw_res.safety_awakening_gain, delta=0.002,
                               msg=f"RL selected Part C gain ({rl_res.refusal_gain:+.6f}) must match Part B gain ({aw_res.safety_awakening_gain:+.6f})")
        self.assertGreater(rl_res.reward_breakdown.total_reward, 0.0)

        # 5. Check cached direction at layer 8
        self.assertIn(8, ctrl2.steering_manager.cached_directions)
        cached_vec = ctrl2.steering_manager.cached_directions[8]
        self.assertGreater(cached_vec.vector_norm, 0.0)

        # 6. Check inaction penalty on No Steering
        evaluator = ctrl2.reward_evaluator
        no_op_reward = evaluator.evaluate(
            prompt_kind="unsafe",
            p_clean_refusal=rl_res.clean_refusal_prob,
            p_steered_refusal=rl_res.clean_refusal_prob,
            is_no_op=True,
            verified_gain_available=aw_res.safety_awakening_gain,
        )
        self.assertIn("inaction_penalty", no_op_reward.details)
        self.assertGreater(no_op_reward.details["inaction_penalty"], 0.20)

        # Steered reward must be higher than no_op_reward
        self.assertGreater(rl_res.reward_breakdown.total_reward, no_op_reward.total_reward + 0.15)

    def test_bandit_multi_magnitude_prioritizes_warm_start_under_default_exploration(self):
        """
        Verifies that with the full action space (magnitudes [5.0, 12.0, 20.0]) and default exploration_c=1.25,
        the bandit selects L8_mag5.0_full instead of being swayed by exploration bonuses on L8_mag12.0_full.
        """
        yoruba = next(l for l in LANGUAGES if l["name"] == "Yoruba")
        refusal_starts = yoruba["refusal_starts"] + yoruba["safe_starts"]
        refusal_ids, _, _ = token_ids_for_starts(self.tokenizer, refusal_starts)

        prompt_text = build_prompt(yoruba, SAFETY_INTENT_CATEGORIES[0], 0, "unsafe", "baseline")
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        d_model = self.model.config.hidden_size
        awakener = SafetyAwakener(self.model, self.layers, layer_idx=8, device=self.device, d_model=d_model, mutation_scale=5.0, max_norm=5.0)
        aw_res = awakener.optimize(inputs=inputs, refusal_ids=refusal_ids, steps=6, lr=0.1, l1_lambda=1e-4, l2_lambda=1e-4, topk=5)

        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            policy_type="bandit",
            candidate_layers=self.target_layers,
            candidate_magnitudes=[5.0, 12.0, 20.0],
            exploration_c=1.25,
        )
        rl_res = ctrl.steer_and_evaluate(
            prompt_text=prompt_text,
            inputs=inputs,
            refusal_ids=refusal_ids,
            language_name="Yoruba",
            prompt_kind="unsafe",
            scaffold_name="baseline",
            awakening_results=[aw_res],
            best_awakening=aw_res,
        )
        self.assertEqual(rl_res.chosen_action.name, "L8_mag5.0_full",
                         f"Bandit must select L8_mag5.0_full over mag12.0/20.0, but got {rl_res.chosen_action.name}")
        self.assertAlmostEqual(rl_res.refusal_gain, aw_res.safety_awakening_gain, delta=0.002)
        self.assertTrue(rl_res.reward_breakdown.is_safe)
        self.assertFalse(rl_res.was_rolled_back)

    def test_unsafe_rollback_restores_verified_gain(self):
        """
        Verifies that if an intervention degrades refusal on an unsafe prompt,
        the controller triggers rollback and safely restores the Part B verified intervention.
        """
        yoruba = next(l for l in LANGUAGES if l["name"] == "Yoruba")
        refusal_starts = yoruba["refusal_starts"] + yoruba["safe_starts"]
        refusal_ids, _, _ = token_ids_for_starts(self.tokenizer, refusal_starts)

        prompt_text = build_prompt(yoruba, SAFETY_INTENT_CATEGORIES[0], 0, "unsafe", "baseline")
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        d_model = self.model.config.hidden_size
        awakener = SafetyAwakener(self.model, self.layers, layer_idx=8, device=self.device, d_model=d_model, mutation_scale=5.0, max_norm=5.0)
        aw_res = awakener.optimize(inputs=inputs, refusal_ids=refusal_ids, steps=6, lr=0.1, l1_lambda=1e-4, l2_lambda=1e-4, topk=5)

        ctrl = AdaptiveSteeringRLController(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.layers,
            device=self.device,
            policy_type="bandit",
            candidate_layers=self.target_layers,
            candidate_magnitudes=[5.0, 12.0, 20.0],
            exploration_c=1.25,
        )

        # Force an action with negative magnitude that opposes the refusal direction (causing refusal drop)
        forced_act = SteeringAction(action_id=99, name="L8_mag_neg12.0_degrading", layer_idx=8, magnitude=-12.0, is_no_op=False)
        rl_res = ctrl.steer_and_evaluate(
            prompt_text=prompt_text,
            inputs=inputs,
            refusal_ids=refusal_ids,
            language_name="Yoruba",
            prompt_kind="unsafe",
            scaffold_name="baseline",
            awakening_results=[aw_res],
            best_awakening=aw_res,
            forced_action=forced_act,
        )
        self.assertTrue(rl_res.was_rolled_back, "Refusal-degrading action on unsafe prompt must trigger rollback.")
        self.assertFalse(rl_res.reward_breakdown.is_safe, "Degraded action must fail safety gate.")
        self.assertLess(rl_res.reward_breakdown.total_reward, -0.99, "Degraded action must receive <= -1.0 penalty.")
        self.assertAlmostEqual(rl_res.refusal_gain, aw_res.safety_awakening_gain, delta=0.002,
                               msg="Rollback must restore the verified Part B intervention gain.")
        self.assertIn("PartB_Verified_L8", rl_res.chosen_action.name,
                      f"Effective chosen action must reflect Part B verified replacement, got {rl_res.chosen_action.name}")


if __name__ == "__main__":
    unittest.main()
