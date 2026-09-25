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
        hardware_profile: Optional[HardwareProfile] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.policy_type = policy_type.lower()
        self.candidate_layers = candidate_layers or [12, 16, 20]
        self.candidate_magnitudes = candidate_magnitudes or [5.0, 12.0, 20.0]

        # 1. Hardware Profiling
        self.hardware = hardware_profile or HardwareProfiler.profile()
        HardwareProfiler.configure_torch_threads(self.hardware)

        # 2. Deep Noir Subsystems
        self.logit_lens = LogitLensAnalyzer(model, tokenizer, layers, device=self.device)
        self.attributor = CausalGradientAttributor(model, layers, device=self.device)
        self.steering_manager = ContrastiveSteeringManager(model, tokenizer, layers, device=self.device)
        self.golden_searcher = DeepNoirGoldenSectionSearcher(
            steering_manager=self.steering_manager,
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            max_benign_refusal_threshold=max_benign_refusal,
        )

        # 3. State Extractor and Reward Evaluator
        self.state_extractor = RLStateExtractor(model, tokenizer, layers, device=self.device)
        self.reward_evaluator = GraduatedRewardEvaluator(
            max_injection_risk=max_injection_risk,
            max_benign_refusal=max_benign_refusal,
        )

        # 4. RL Policy
        if self.policy_type == "bandit":
            self.bandit = ContextualBanditController(
                state_dim=self.state_extractor.state_dim,
                candidate_layers=self.candidate_layers,
                candidate_magnitudes=self.candidate_magnitudes,
                exploration_c=exploration_c,
            )
            self.ppo = None
        elif self.policy_type == "ppo":
            self.ppo = ConstrainedPPOController(
                state_dim=self.state_extractor.state_dim,
                candidate_layers=self.candidate_layers,
                candidate_magnitudes=self.candidate_magnitudes,
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

        # 3. Causal Gradient Attribution to identify high-leverage attention heads
        try:
            sample_harmful = harmful_prompts[0]
            enc_harmful = self.tokenizer(sample_harmful, return_tensors="pt").to(self.device)
            head_attributions = self.attributor.attribute_heads(
                inputs=enc_harmful,
                refusal_ids=refusal_ids,
                target_layer_indices=self.candidate_layers,
                top_k=15,
            )
            heads_by_layer: Dict[int, List[int]] = {}
            for h in head_attributions:
                heads_by_layer.setdefault(h.layer_idx, []).append(h.head_idx)
            self.causal_heads_by_layer = heads_by_layer
            logger.info(f"Deep Noir causal heads identified by layer: {self.causal_heads_by_layer}")
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

    def steer_and_evaluate(
        self,
        prompt_text: str,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        language_name: str,
        prompt_kind: str,
        scaffold_name: str = "baseline",
        update_policy: bool = True,
    ) -> AdaptiveSteeringResult:
        """
        Executes full adaptive steering cycle for an individual prompt:
        1. State extraction
        2. Action selection via RL policy
        3. Hard safety gate pre-check
        4. Intervention application
        5. Forward pass and reward computation
        6. Rollback if safety violated
        7. Online policy update
        """
        audit: List[str] = []
        audit.append(f"Received prompt ({prompt_kind}) in {language_name} under scaffold {scaffold_name}.")

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

        # 2. Action selection via RL policy
        ppo_decision = None
        bandit_decision = None
        active_heads = self.causal_heads_by_layer

        if self.policy_type == "bandit" and self.bandit is not None:
            bandit_decision = self.bandit.select_action(s_vec, active_causal_heads_by_layer=active_heads)
            policy_action = bandit_decision.action
            audit.append(f"Bandit selected action: {policy_action.name} (Predicted Reward={bandit_decision.predicted_reward:.4f}, UCB={bandit_decision.ucb_score:.4f}).")
        elif self.policy_type == "ppo" and self.ppo is not None:
            ppo_decision = self.ppo.select_action(s_vec, active_causal_heads_by_layer=active_heads)
            policy_action = ppo_decision.action
            audit.append(f"PPO selected action: {policy_action.name} (Value={ppo_decision.value_estimate:.4f}, CostEst={ppo_decision.cost_estimate:.4f}).")
        elif self.policy_type in ("static", "deep_noir_classic"):
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

        if not executed_action.is_no_op and executed_action.layer_idx is not None:
            with self.steering_manager.apply_steering(
                layer_idx=executed_action.layer_idx,
                magnitude=executed_action.magnitude,
                head_indices=executed_action.target_heads,
            ):
                out_steered = self.model(**inputs)
                probs_s = F.softmax(out_steered.logits[0, -1, :].float(), dim=-1)
                steered_refusal = float(sum(probs_s[t].item() for t in set(refusal_ids) if 0 <= t < probs_s.shape[0]))
                steered_entropy = float(-torch.sum(probs_s * torch.log(probs_s + 1e-12)).item())

            audit.append(f"Applied intervention at layer {executed_action.layer_idx} (magnitude={executed_action.magnitude:.2f}). Steered Refusal={steered_refusal:.6f}, Steered Entropy={steered_entropy:.4f}.")
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
            audit.append(f"ROLLBACK EXECUTED: {rollback_reason} Restored clean output.")

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
        return result

    def flush_ppo_updates(self) -> Dict[str, float]:
        """Flushes remaining transitions in PPO buffer and runs training update."""
        if self.ppo is not None and len(self.ppo.buffer) > 0:
            return self.ppo.update()
        return {}
