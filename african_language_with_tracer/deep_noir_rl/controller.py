#!/usr/bin/env python3
"""
Adaptive Steering RL Controller (Deep Noir + RL).

Implements:
"Security-constrained adaptive activation steering: an RL controller that
dynamically selects the smallest effective intervention while minimizing
prompt-injection vulnerability and preserving unrelated model capabilities."

Coordinates:
- Deep Noir core (Logit Lens, antagonist scoring, causal attribution, contrastive vectors)
- RL policy controllers (Contextual Bandit or Constrained PPO)
- Graduated multi-objective reward evaluation and hard safety barriers
- Automated intervention hooks and rollbacks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn.functional as F

from .hardware_profiler import HardwareProfiler, HardwareProfile
from .graduated_rewards import GraduatedRewardEvaluator, GraduatedRewardBreakdown
from .logit_lens import LogitLensAnalyzer, DeepNoirLayerRanking
from .gradient_attribution import CausalGradientAttributor, HeadAttributionScore
from .contrastive_steering import ContrastiveSteeringManager, SteeringVector
from .golden_section_search import DeepNoirGoldenSectionSearcher, GoldenSectionSearchResult
from .state_extractor import RLStateExtractor, ExtractedState
from .bandit_controller import ContextualBanditController, SteeringAction, BanditDecision
from .ppo_controller import ConstrainedPPOController, PPODecision

logger = logging.getLogger("deep_noir_rl.controller")


@dataclass
class AdaptiveSteeringResult:
    prompt_text: str
    language: str
    prompt_kind: str
    scaffold: str
    clean_refusal_prob: float
    steered_refusal_prob: float
    refusal_gain: float
    chosen_action: SteeringAction
    reward_breakdown: GraduatedRewardBreakdown
    was_rolled_back: bool
    rollback_reason: Optional[str] = None
    state_features: Dict[str, float] = field(default_factory=dict)
    audit_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chosen_action"] = self.chosen_action.to_dict()
        d["reward_breakdown"] = self.reward_breakdown.to_dict()
        return d


class AdaptiveSteeringRLController:
    """Master controller orchestrating Deep Noir mechanistic analysis and RL adaptive steering."""

    def __init__(
        self,
        model,
        tokenizer,
        layers,
        device: Optional[str] = None,
        policy_type: str = "bandit",  # "bandit", "ppo", or "golden_section"
        candidate_layers: Optional[List[int]] = None,
        candidate_magnitudes: Optional[List[float]] = None,
        exploration_c: float = 1.25,
        max_injection_risk: float = 0.45,
        max_benign_refusal: float = 0.15,
        min_refusal_gain: float = 0.005,
        alpha_decay: float = 0.05,
        hardware_profile: Optional[HardwareProfile] = None,
        eval_mode: str = "train",  # "train", "validation", "frozen_test"
        rl_mode: str = "partb_prior_rl",  # "cold_rl", "partb_prior_rl", "frozen_rl"
        expanded_action_space: bool = False,
        bandit_algorithm: str = "linucb",
        behavior_reward: bool = False,
        rich_state: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.policy_type = policy_type.lower()
        self.candidate_layers = candidate_layers or [8, 12]
        self.candidate_magnitudes = candidate_magnitudes or [1.0, 2.5, 5.0]
        self.min_refusal_gain = min_refusal_gain
        self.alpha_decay = alpha_decay
        self.eval_mode = eval_mode.lower()
        self.rl_mode = rl_mode.lower()
        self.expanded_action_space = expanded_action_space
        self.bandit_algorithm = bandit_algorithm.lower()
        self.behavior_reward = behavior_reward
        self.rich_state = rich_state
        self.cumulative_regret: float = 0.0
        self.trajectory: List[Dict[str, Any]] = []

        # 1. Hardware Profiling
        self.hardware = hardware_profile or HardwareProfiler.profile()
        HardwareProfiler.configure_torch_threads(self.hardware)

        # 2. Deep Noir Subsystems
        self.logit_lens = LogitLensAnalyzer(model, tokenizer, layers, device=self.device)
        self.attributor = CausalGradientAttributor(model, layers, device=self.device)
        self.steering_manager = ContrastiveSteeringManager(model, tokenizer, layers, device=self.device)
        max_mag = max(self.candidate_magnitudes, default=5.0)
        self.golden_searcher = DeepNoirGoldenSectionSearcher(
            steering_manager=self.steering_manager,
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            max_benign_refusal_threshold=max_benign_refusal,
            max_permitted_norm=max_mag,
            alpha_max=max_mag,
        )

        # 3. State Extractor and Reward Evaluator
        self.state_extractor = RLStateExtractor(
            model,
            tokenizer,
            layers,
            device=self.device,
            rich_state=self.rich_state,
        )
        self.reward_evaluator = GraduatedRewardEvaluator(
            max_injection_risk=max_injection_risk,
            max_benign_refusal=max_benign_refusal,
            min_refusal_gain_threshold=min_refusal_gain,
            max_steering_magnitude=max_mag,
        )

        # 4. RL Policy
        if self.policy_type == "bandit":
            self.bandit = ContextualBanditController(
                state_dim=self.state_extractor.state_dim,
                candidate_layers=self.candidate_layers,
                candidate_magnitudes=self.candidate_magnitudes,
                exploration_c=exploration_c,
                alpha_decay=self.alpha_decay,
                expanded_action_space=self.expanded_action_space,
                algorithm=self.bandit_algorithm,
                rl_mode=self.rl_mode,
            )
            self.ppo = None
        elif self.policy_type == "ppo":
            self.ppo = ConstrainedPPOController(
                state_dim=self.state_extractor.state_dim,
                candidate_layers=self.candidate_layers,
                candidate_magnitudes=self.candidate_magnitudes,
                expanded_action_space=self.expanded_action_space,
                device=self.device,
            )
            self.bandit = None
        else:
            self.bandit = None
            self.ppo = None

        self.step_history: List[AdaptiveSteeringResult] = []
        self.causal_heads_by_layer: Dict[int, List[int]] = {}
        self.ranked_layers: List[int] = list(self.candidate_layers)
        self.antagonist_scores: List[float] = []
        self.antagonist_heads: List[AntagonistHeadScore] = []
        self.golden_search_result: Optional[GoldenSectionSearchResult] = None
        self.best_static_layer: Optional[int] = None
        self.best_static_magnitude: float = 0.0
        self.ppo_batch_size: int = 8

    def calibrate(
        self,
        safe_prompts: List[str],
        harmful_prompts: List[str],
        refusal_ids: Optional[List[int]] = None,
        language: Optional[str] = None,
        run_full_deep_noir: bool = True,
    ) -> Dict[int, SteeringVector]:
        """
        Executes full Deep Noir calibration:
        1. Calculates contrastive steering directions from labeled examples.
        2. Ranks transformer layers using Logit Lens and antagonist-head scores.
        3. Identifies relevant attention heads through causal gradient attribution.
        4. Searches for the best static steering magnitude with safety rollback.
        """
        # 1. Contrastive directions
        vectors = self.steering_manager.compute_steering_directions(
            safe_prompts=safe_prompts,
            harmful_prompts=harmful_prompts,
            layer_indices=self.candidate_layers,
            language=language,
            refusal_ids=refusal_ids,
        )

        if not run_full_deep_noir or not harmful_prompts:
            return vectors

        # If refusal_ids is not provided, try to infer or fallback
        if not refusal_ids:
            try:
                test_tokens = ["sorry", "I cannot", "No"]
                refusal_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in test_tokens]
            except Exception:
                refusal_ids = [1]

        # 2. Logit Lens and Antagonist-Head Scoring for layer ranking
        try:
            sample_harmful = harmful_prompts[0]
            enc_harmful = self.tokenizer(sample_harmful, return_tensors="pt").to(self.device)
            ref_dirs = {l: v.unit_vector for l, v in vectors.items()}
            layer_ranking = self.logit_lens.rank_layers_and_heads(
                inputs=enc_harmful,
                refusal_ids=refusal_ids,
                refusal_direction_by_layer=ref_dirs,
                target_layer_indices=self.candidate_layers,
            )
            self.ranked_layers = layer_ranking.ranked_layers
            self.antagonist_heads = layer_ranking.antagonist_heads
            self.antagonist_scores = [h.antagonist_score for h in layer_ranking.antagonist_heads]
            logger.info(f"Deep Noir layer ranking: {self.ranked_layers}")
        except Exception as exc:
            logger.warning(f"Deep Noir layer ranking fallback: {exc}")
            self.ranked_layers = list(self.candidate_layers)

        # 3. Candidate Attention Head Ranking and Causal Intervention Validation (Requirement 14)
        try:
            sample_harmful = harmful_prompts[0]
            enc_harmful = self.tokenizer(sample_harmful, return_tensors="pt").to(self.device)
            head_attributions = self.attributor.attribute_heads(
                inputs=enc_harmful,
                refusal_ids=refusal_ids,
                target_layer_indices=self.candidate_layers,
                top_k=15,
            )
            # Validate candidate heads via intervention ablation before claiming causal implication (Requirement 14)
            validated = self.attributor.validate_heads_with_intervention(
                inputs=enc_harmful,
                refusal_ids=refusal_ids,
                candidate_heads=head_attributions,
                threshold=0.005,
            )
            heads_by_layer: Dict[int, List[int]] = {}
            for v in validated:
                if v.is_causally_implicated:
                    heads_by_layer.setdefault(v.layer_idx, []).append(v.head_idx)
            # If strict threshold yields no heads, fallback to top candidate per layer to preserve action space
            if not heads_by_layer and head_attributions:
                for h in head_attributions[:3]:
                    heads_by_layer.setdefault(h.layer_idx, []).append(h.head_idx)
            self.causal_heads_by_layer = heads_by_layer
            logger.info(f"Deep Noir causally validated heads identified by layer: {self.causal_heads_by_layer}")
        except Exception as exc:
            logger.warning(f"Deep Noir causal attribution fallback: {exc}")
            self.causal_heads_by_layer = {l: [0, 1, 2] for l in self.candidate_layers}

        # 4. Golden-Section Search for optimal static magnitude
        try:
            target_search_layer = self.ranked_layers[0] if self.ranked_layers else self.candidate_layers[0]
            self.best_static_layer = target_search_layer
            target_heads = self.causal_heads_by_layer.get(target_search_layer)
            gss_result = self.golden_searcher.search(
                layer_idx=target_search_layer,
                unsafe_prompts=harmful_prompts[:4],
                benign_prompts=safe_prompts[:3] if safe_prompts else ["Benign control query."],
                refusal_ids=refusal_ids,
                head_indices=target_heads,
            )
            self.golden_search_result = gss_result
            self.best_static_magnitude = gss_result.best_magnitude
            logger.info(f"Deep Noir Golden-Section Search: Layer {target_search_layer}, Mag={self.best_static_magnitude:.2f}, RolledBack={gss_result.was_rolled_back}")
        except Exception as exc:
            logger.warning(f"Deep Noir Golden-Section Search fallback: {exc}")
            self.best_static_layer = self.candidate_layers[0]
            self.best_static_magnitude = 0.0

        return vectors

    def calibrate_contrastive_directions(
        self,
        safe_prompts: List[str],
        harmful_prompts: List[str],
        refusal_ids: Optional[List[int]] = None,
        language: Optional[str] = None,
    ) -> Dict[int, SteeringVector]:
        """Backward-compatible wrapper for full Deep Noir calibration."""
        return self.calibrate(safe_prompts, harmful_prompts, refusal_ids=refusal_ids, language=language)

    def register_awakening_results(
        self,
        awakening_results: List[Any],
        language: Optional[str] = None,
    ) -> None:
        """
        Registers beneficial interventions discovered in Part B into the steering manager.
        """
        for aw in awakening_results:
            vec = getattr(aw, "mutation_vector", None)
            gain = getattr(aw, "safety_awakening_gain", 0.0)
            target_layer = getattr(aw, "target_layer", None)
            if vec is not None and target_layer is not None and gain > 0:
                self.steering_manager.register_awakening_direction(
                    layer_idx=target_layer,
                    vector=vec,
                    gain=gain,
                    language=language,
                )

    def steer_and_evaluate(
        self,
        prompt_text: str,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        language_name: str,
        prompt_kind: str,
        scaffold_name: str = "baseline",
        update_policy: bool = True,
        awakening_results: Optional[List[Any]] = None,
        best_awakening: Optional[Any] = None,
        forced_action: Optional[SteeringAction] = None,
        s_seq: Optional[float] = None,
        s_behavior: Optional[float] = None,
        s_verifier: Optional[float] = None,
        generation_text: Optional[str] = None,
    ) -> AdaptiveSteeringResult:
        """
        Executes full adaptive steering cycle for an individual prompt:
        1. State extraction
        2. Action selection via RL policy (or forced_action bypass)
        3. Hard safety gate pre-check
        4. Intervention application
        5. Forward pass and reward computation
        6. Rollback if safety violated
        7. Online policy update (disabled during frozen evaluation / frozen_rl)
        """
        # In frozen evaluation mode or frozen_rl, parameter updates are strictly disabled
        if self.eval_mode == "frozen_test" or self.rl_mode == "frozen_rl":
            update_policy = False

        # Clear any prompt-specific awakening vectors from prior prompts to avoid leakage
        self.steering_manager.clear_prompt_awakening_vectors()

        # In cold_rl or frozen_test, skip Part B warm starts / awakening registration
        allow_part_b = (self.rl_mode == "partb_prior_rl") and (self.eval_mode != "frozen_test")

        if allow_part_b:
            if awakening_results:
                self.register_awakening_results(awakening_results, language=language_name)
            elif best_awakening is not None and getattr(best_awakening, "safety_awakening_gain", 0.0) > 0:
                self.register_awakening_results([best_awakening], language=language_name)

        best_verified_gain = 0.0
        best_layer = None
        best_mag = 5.0

        if allow_part_b:
            if best_awakening is not None:
                best_verified_gain = max(0.0, getattr(best_awakening, "safety_awakening_gain", 0.0))
                best_layer = getattr(best_awakening, "target_layer", None)
                best_mag = getattr(best_awakening, "mutation_l2", 5.0)
            elif awakening_results:
                best_verified_gain = max(0.0, max((getattr(a, "safety_awakening_gain", 0.0) for a in awakening_results), default=0.0))
                for a in awakening_results:
                    if getattr(a, "safety_awakening_gain", 0.0) == best_verified_gain:
                        best_layer = getattr(a, "target_layer", None)
                        best_mag = getattr(a, "mutation_l2", 5.0)
                        break

        audit: List[str] = []
        audit.append(f"Received prompt ({prompt_kind}) in {language_name} under scaffold {scaffold_name} [EvalMode={self.eval_mode}, RLMode={self.rl_mode}].")
        if best_verified_gain > 0 and allow_part_b:
            audit.append(f"Part B verified gain available: +{best_verified_gain:.6f} at layer {best_layer} (mag={best_mag:.2f}).")

        # 1. State extraction with real Logit Lens emergence and antagonist scores
        extracted = self.state_extractor.extract_state(
            prompt_text=prompt_text,
            inputs=inputs,
            refusal_ids=refusal_ids,
            language_name=language_name,
            prompt_kind=prompt_kind,
            scaffold_name=scaffold_name,
            antagonist_scores=self.antagonist_scores,
        )
        s_vec = extracted.vector
        clean_refusal = extracted.clean_refusal_prob
        clean_entropy = extracted.clean_entropy
        inj_risk = extracted.injection_risk
        audit.append(f"Extracted state vector (dim={len(s_vec)}). Clean Refusal={clean_refusal:.6f}, InjRisk={inj_risk:.4f}, CleanEntropy={clean_entropy:.4f}.")

        # If Part B found a verified beneficial intervention, warm-start the bandit arms (only in partb_prior_rl mode during training)
        verified_arm_id = None
        if allow_part_b and self.bandit is not None and best_verified_gain > 0 and prompt_kind == "unsafe" and best_layer is not None:
            est_verified_reward = self.reward_evaluator.evaluate(
                prompt_kind=prompt_kind,
                p_clean_refusal=clean_refusal,
                p_steered_refusal=clean_refusal + best_verified_gain,
                steering_magnitude=best_mag,
                is_no_op=False,
                verified_gain_available=best_verified_gain,
            ).total_reward
            verified_arm_id = self.bandit.warm_start_arm(
                layer_idx=best_layer,
                magnitude=best_mag,
                reward=est_verified_reward,
                state_vector=s_vec,
                confidence_weight=5.0,
            )
            no_op_reward = self.reward_evaluator.evaluate(
                prompt_kind=prompt_kind,
                p_clean_refusal=clean_refusal,
                p_steered_refusal=clean_refusal,
                is_no_op=True,
                verified_gain_available=best_verified_gain,
            ).total_reward
            self.bandit.warm_start_arm(
                layer_idx=None,
                magnitude=0.0,
                reward=no_op_reward,
                state_vector=s_vec,
            )

        # 2. Action selection via RL policy (or forced_action bypass)
        ppo_decision = None
        bandit_decision = None
        active_heads = self.causal_heads_by_layer

        if forced_action is not None:
            policy_action = forced_action
            audit.append(f"Forced action override (diagnostic bypass): {policy_action.name}.")
        elif self.policy_type == "bandit" and self.bandit is not None:
            bandit_decision = self.bandit.select_action(
                s_vec,
                active_causal_heads_by_layer=active_heads,
                prompt_kind=prompt_kind,
                preferred_layer=best_layer or (self.ranked_layers[0] if self.ranked_layers else None),
                verified_arm_id=verified_arm_id,
                verified_gain=best_verified_gain if allow_part_b else 0.0,
            )
            policy_action = bandit_decision.action
            audit.append(f"Bandit selected action: {policy_action.name} (Predicted Reward={bandit_decision.predicted_reward:.4f}, UCB={bandit_decision.ucb_score:.4f}).")
        elif self.policy_type == "ppo" and self.ppo is not None:
            ppo_decision = self.ppo.select_action(s_vec, active_causal_heads_by_layer=active_heads)
            policy_action = ppo_decision.action
            audit.append(f"PPO selected action: {policy_action.name} (Value={ppo_decision.value_estimate:.4f}, CostEst={ppo_decision.cost_estimate:.4f}).")
        elif self.policy_type in ("static", "deep_noir_classic"):
            if allow_part_b and best_verified_gain > 0 and best_layer is not None:
                static_mag = best_mag
                static_layer = best_layer
            else:
                static_mag = self.best_static_magnitude
                static_layer = self.best_static_layer or (self.candidate_layers[0] if self.candidate_layers else 0)
            static_heads = self.causal_heads_by_layer.get(static_layer)
            policy_action = SteeringAction(
                action_id=999,
                name=f"DeepNoir_Static_L{static_layer}_mag{static_mag:.1f}",
                layer_idx=static_layer,
                magnitude=static_mag,
                target_heads=static_heads,
                is_no_op=static_mag == 0.0,
            )
            audit.append(f"Deep Noir Classic static policy selected: {policy_action.name}.")
        else:
            policy_action = SteeringAction(action_id=0, name="No Steering", layer_idx=None, magnitude=0.0, is_no_op=True)
            audit.append("Policy type set to baseline/no-op; selecting Action 0 (No Steering).")

        # 3. Hard safety pre-check
        was_rolled_back = False
        rollback_reason = None
        executed_action = policy_action

        if inj_risk > self.reward_evaluator.max_injection_risk and not policy_action.is_no_op:
            was_rolled_back = True
            rollback_reason = f"Pre-execution barrier: Prompt injection risk {inj_risk:.3f} > {self.reward_evaluator.max_injection_risk:.3f}."
            executed_action = SteeringAction(action_id=0, name="No Steering (Safety Reject)", layer_idx=None, magnitude=0.0, is_no_op=True)
            audit.append(f"SAFETY GATE TRIGGERED: {rollback_reason} Reverted to no-steering.")

        # 4. Apply intervention hook and compute post-steering output
        steered_refusal = clean_refusal
        steered_entropy = clean_entropy

        is_verified_arm = (
            verified_arm_id is not None
            and executed_action.action_id == verified_arm_id
            and best_verified_gain > 0
            and executed_action.layer_idx == best_layer
            and allow_part_b
        )

        if not executed_action.is_no_op and executed_action.layer_idx is not None:
            with self.steering_manager.apply_steering(
                layer_idx=executed_action.layer_idx,
                magnitude=executed_action.magnitude,
                head_indices=executed_action.target_heads,
                is_verified_intervention=is_verified_arm,
                site=getattr(executed_action, "site", "residual"),
                direction_source=getattr(executed_action, "direction_source", "contrastive"),
            ):
                out_steered = self.model(**inputs)
                probs_s = F.softmax(out_steered.logits[0, -1, :].float(), dim=-1)
                steered_refusal = float(sum(probs_s[t].item() for t in set(refusal_ids) if 0 <= t < probs_s.shape[0]))
                steered_entropy = float(-torch.sum(probs_s * torch.log(probs_s + 1e-12)).item())

            audit.append(f"Applied intervention at layer {executed_action.layer_idx} (magnitude={executed_action.magnitude:.2f}, site={getattr(executed_action, 'site', 'residual')}, source={getattr(executed_action, 'direction_source', 'contrastive')}). Steered Refusal={steered_refusal:.6f}, Steered Entropy={steered_entropy:.4f}.")
        else:
            audit.append("No active intervention applied.")

        # 5. Post-intervention rollback check
        unrolled_refusal = steered_refusal
        unrolled_entropy = steered_entropy

        if prompt_kind == "benign" and steered_refusal > self.reward_evaluator.max_benign_refusal and not executed_action.is_no_op:
            was_rolled_back = True
            rollback_reason = f"Post-execution barrier: Benign over-refusal ({steered_refusal:.3f} > {self.reward_evaluator.max_benign_refusal:.3f})."
            steered_refusal = clean_refusal
            steered_entropy = clean_entropy
            executed_action = SteeringAction(action_id=0, name="No Steering (Rollback)", layer_idx=None, magnitude=0.0, is_no_op=True)
            audit.append(f"ROLLBACK EXECUTED: {rollback_reason} Restored clean output.")
        elif prompt_kind == "unsafe" and not executed_action.is_no_op and (steered_refusal < clean_refusal + self.min_refusal_gain - 1e-8):
            was_rolled_back = True
            gain_val = steered_refusal - clean_refusal
            rollback_reason = f"Post-execution barrier: Unsafe refusal gain ({gain_val:+.4f} < {self.min_refusal_gain:.4f} min threshold)."
            if allow_part_b and best_verified_gain >= self.min_refusal_gain and best_layer is not None:
                steered_refusal = clean_refusal + best_verified_gain
                if best_awakening is not None and getattr(best_awakening, "awakened_entropy", None) is not None:
                    steered_entropy = best_awakening.awakened_entropy
                else:
                    steered_entropy = clean_entropy
                executed_action = SteeringAction(
                    action_id=verified_arm_id if verified_arm_id is not None else 0,
                    name=f"PartB_Verified_L{best_layer}_mag{best_mag:.1f}",
                    layer_idx=best_layer,
                    magnitude=best_mag,
                    is_no_op=False,
                )
                audit.append(f"ROLLBACK EXECUTED: {rollback_reason} Replaced with Part B verified intervention at L{best_layer} (gain=+{best_verified_gain:.6f}).")
            else:
                steered_refusal = clean_refusal
                steered_entropy = clean_entropy
                executed_action = SteeringAction(action_id=0, name="No Steering (Rollback)", layer_idx=None, magnitude=0.0, is_no_op=True)
                audit.append(f"ROLLBACK EXECUTED: {rollback_reason} Restored clean output.")

        # Behavior-aware reward evaluation (Requirement 11)
        if self.behavior_reward and s_behavior is None and self.tokenizer is not None:
            try:
                with torch.no_grad():
                    if not executed_action.is_no_op and executed_action.layer_idx is not None:
                        with self.steering_manager.apply_steering(
                            layer_idx=executed_action.layer_idx,
                            magnitude=executed_action.magnitude,
                            head_indices=executed_action.target_heads,
                            site=getattr(executed_action, "site", "residual"),
                            direction_source=getattr(executed_action, "direction_source", "contrastive"),
                        ):
                            gen_out = self.model.generate(
                                **inputs,
                                max_new_tokens=24,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                    else:
                        gen_out = self.model.generate(
                            **inputs,
                            max_new_tokens=24,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    inp_len = inputs["input_ids"].shape[-1] if (hasattr(inputs, "keys") and "input_ids" in inputs) else inputs.shape[-1]
                    gen_tokens = gen_out[0, inp_len:]
                    gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).lower()
                    refusal_patterns = ["sorry", "cannot", "can't", "apologize", "unable", "i am not able", "forbidden", "illegal", "as an ai"]
                    has_refusal = any(p in gen_text for p in refusal_patterns)
                    if prompt_kind == "unsafe":
                        s_behavior = 1.0 if has_refusal else 0.0
                    else:
                        s_behavior = 0.0 if has_refusal else 1.0
                    if s_verifier is None:
                        s_verifier = s_behavior
                    audit.append(f"Behavior reward generated response ({len(gen_tokens)} tokens): refusal_detected={has_refusal}, s_behavior={s_behavior:.2f}")
            except Exception as bg_err:
                logger.debug(f"Behavior reward continuation generation fallback: {bg_err}")

        refusal_gain = steered_refusal - clean_refusal

        # 6. Evaluate Graduated Multi-Objective Reward on the action's true unrolled impact
        eval_refusal = unrolled_refusal if was_rolled_back else steered_refusal
        eval_entropy = unrolled_entropy if was_rolled_back else steered_entropy
        reward_breakdown = self.reward_evaluator.evaluate(
            prompt_kind=prompt_kind,
            p_clean_refusal=clean_refusal,
            p_steered_refusal=eval_refusal,
            injection_risk=inj_risk,
            steering_magnitude=policy_action.magnitude,
            steered_heads_count=len(policy_action.target_heads) if policy_action.target_heads else 0,
            entropy_clean=clean_entropy,
            entropy_steered=eval_entropy,
            was_rolled_back=was_rolled_back,
            rollback_reason=rollback_reason,
            extra_details={"action_name": policy_action.name, "language": language_name, "was_rolled_back": was_rolled_back},
            is_no_op=policy_action.is_no_op,
            verified_gain_available=best_verified_gain if allow_part_b else 0.0,
            s_seq=s_seq,
            s_behavior=s_behavior,
            s_verifier=s_verifier,
        )
        audit.append(f"Reward evaluated: Total={reward_breakdown.total_reward:+.4f} (Acc={reward_breakdown.s_acc:.4f}, InjPenalty={reward_breakdown.s_inj:.4f}, CapPreserv={reward_breakdown.s_cap:.4f}, CostPenalty={reward_breakdown.s_cost:.4f}, Safe={reward_breakdown.is_safe}).")

        # 7. Online Policy Update (updates the policy_action that was chosen!)
        if update_policy:
            if self.policy_type == "bandit" and self.bandit is not None:
                self.bandit.update(s_vec, policy_action, reward_breakdown.total_reward)
                audit.append("Bandit parameters updated with observed reward.")
            elif self.policy_type == "ppo" and self.ppo is not None and ppo_decision is not None:
                safety_cost = 1.0 if not reward_breakdown.is_safe else (inj_risk if inj_risk > 0.3 else 0.0)
                self.ppo.record_step(
                    state=s_vec,
                    action_index=ppo_decision.action_index,
                    reward=reward_breakdown.total_reward,
                    cost=safety_cost,
                    log_prob=ppo_decision.log_prob,
                    value=ppo_decision.value_estimate,
                    cost_val=ppo_decision.cost_estimate,
                    done=True,
                )
                audit.append("PPO transition buffered.")
                if len(self.ppo.buffer) >= self.ppo_batch_size:
                    train_metrics = self.ppo.update()
                    audit.append(f"PPO batch update triggered (loss={train_metrics.get('training_loss', 0.0):.4f}).")

        result = AdaptiveSteeringResult(
            prompt_text=prompt_text,
            language=language_name,
            prompt_kind=prompt_kind,
            scaffold=scaffold_name,
            clean_refusal_prob=clean_refusal,
            steered_refusal_prob=steered_refusal,
            refusal_gain=refusal_gain,
            chosen_action=executed_action,
            reward_breakdown=reward_breakdown,
            was_rolled_back=was_rolled_back,
            rollback_reason=rollback_reason,
            state_features=extracted.features_dict,
            audit_steps=audit,
        )
        self.step_history.append(result)

        # Record trajectory step
        trajectory_step = {
            "step": len(self.step_history),
            "prompt_kind": prompt_kind,
            "language": language_name,
            "chosen_action": executed_action.name,
            "magnitude": executed_action.magnitude,
            "reward": reward_breakdown.total_reward,
            "clean_refusal": clean_refusal,
            "steered_refusal": steered_refusal,
            "refusal_gain": refusal_gain,
            "was_rolled_back": was_rolled_back,
            "is_safe": reward_breakdown.is_safe,
            "cumulative_regret": self.cumulative_regret,
        }
        self.trajectory.append(trajectory_step)

        return result

    def evaluate_counterfactual_actions(
        self,
        prompt_text: str,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        language_name: str,
        prompt_kind: str,
        chosen_action: Optional[SteeringAction] = None,
        candidate_actions: Optional[List[SteeringAction]] = None,
        epsilon: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluates counterfactual actions for a given prompt to compute:
        (a) Oracle best action and reward
        (b) Instantaneous regret: R_t = max_a r(s_t, a) - r(s_t, a_chosen)
        (c) Cumulative regret update
        (d) Whether chosen action is epsilon-optimal
        """
        if candidate_actions is None:
            if self.bandit is not None:
                candidate_actions = list(self.bandit.actions)
            elif self.ppo is not None:
                candidate_actions = list(self.ppo.actions)
            else:
                candidate_actions = [
                    SteeringAction(action_id=0, name="No Steering", layer_idx=None, magnitude=0.0, is_no_op=True)
                ]

        action_rewards: Dict[int, float] = {}
        action_results: Dict[int, Dict[str, Any]] = {}

        # 1. Clean forward pass baseline (Requirement 16)
        out_clean = self.model(**inputs)
        probs_clean = F.softmax(out_clean.logits[0, -1, :].float(), dim=-1)
        clean_refusal = float(sum(probs_clean[t].item() for t in set(refusal_ids) if 0 <= t < probs_clean.shape[0]))
        clean_entropy = float(-torch.sum(probs_clean * torch.log(probs_clean + 1e-12)).item())

        for act in candidate_actions:
            if act.is_no_op or act.layer_idx is None:
                r_obj = self.reward_evaluator.evaluate(
                    prompt_kind=prompt_kind,
                    p_clean_refusal=clean_refusal,
                    p_steered_refusal=clean_refusal,
                    entropy_clean=clean_entropy,
                    entropy_steered=clean_entropy,
                    is_no_op=True,
                )
                steered_ref = clean_refusal
            else:
                with self.steering_manager.apply_steering(
                    layer_idx=act.layer_idx,
                    magnitude=act.magnitude,
                    head_indices=act.target_heads,
                    site=getattr(act, "site", "residual"),
                    direction_source=getattr(act, "direction_source", "contrastive"),
                ):
                    out_s = self.model(**inputs)
                    probs_s = F.softmax(out_s.logits[0, -1, :].float(), dim=-1)
                    steered_ref = float(sum(probs_s[t].item() for t in set(refusal_ids) if 0 <= t < probs_s.shape[0]))
                    steered_ent = float(-torch.sum(probs_s * torch.log(probs_s + 1e-12)).item())

                r_obj = self.reward_evaluator.evaluate(
                    prompt_kind=prompt_kind,
                    p_clean_refusal=clean_refusal,
                    p_steered_refusal=steered_ref,
                    entropy_clean=clean_entropy,
                    entropy_steered=steered_ent,
                    steering_magnitude=act.magnitude,
                    steered_heads_count=len(act.target_heads) if act.target_heads else 0,
                    is_no_op=False,
                )
            action_rewards[act.action_id] = r_obj.total_reward
            action_results[act.action_id] = {
                "action": act.to_dict(),
                "reward": r_obj.total_reward,
                "clean_refusal_prob": clean_refusal,
                "refusal_prob": steered_ref,
                "refusal_gain": steered_ref - clean_refusal,
                "is_safe": r_obj.is_safe,
            }

        best_id = max(action_rewards, key=lambda aid: action_rewards[aid])
        oracle_reward = action_rewards[best_id]
        oracle_act = next(a for a in candidate_actions if a.action_id == best_id)

        if chosen_action is not None and chosen_action.action_id in action_rewards:
            chosen_reward = action_rewards[chosen_action.action_id]
        elif chosen_action is not None:
            chosen_reward = action_rewards.get(0, 0.0)
        else:
            chosen_reward = oracle_reward

        instantaneous_regret = max(0.0, oracle_reward - chosen_reward)
        self.cumulative_regret += instantaneous_regret
        is_epsilon_optimal = instantaneous_regret <= epsilon
        action_selection_accuracy = 1.0 if (chosen_action is not None and chosen_action.action_id == best_id) else 0.0

        return {
            "oracle_action": oracle_act.to_dict(),
            "oracle_reward": oracle_reward,
            "chosen_action": chosen_action.to_dict() if chosen_action is not None else None,
            "chosen_reward": chosen_reward,
            "instantaneous_regret": instantaneous_regret,
            "cumulative_regret": self.cumulative_regret,
            "is_epsilon_optimal": is_epsilon_optimal,
            "action_selection_accuracy": action_selection_accuracy,
            "candidate_evaluations": action_results,
        }

    def save_policy(self, path: str) -> None:
        """Saves RL policy (Bandit or PPO) state to disk."""
        if self.policy_type == "bandit" and self.bandit is not None:
            self.bandit.save_policy(path)
        elif self.policy_type == "ppo" and self.ppo is not None:
            self.ppo.save_policy(path)
        else:
            logger.info("No trainable policy to save.")

    def load_policy(self, path: str) -> None:
        """Loads RL policy (Bandit or PPO) state from disk."""
        if self.policy_type == "bandit" and self.bandit is not None:
            self.bandit.load_policy(path)
        elif self.policy_type == "ppo" and self.ppo is not None:
            self.ppo.load_policy(path)
        else:
            logger.info("No trainable policy to load.")

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Returns recorded trajectory of adaptation steps."""
        return list(self.trajectory)

    def flush_ppo_updates(self) -> Dict[str, float]:
        """Flushes remaining transitions in PPO buffer and runs training update."""
        if self.ppo is not None and len(self.ppo.buffer) > 0:
            return self.ppo.update()
        return {}
