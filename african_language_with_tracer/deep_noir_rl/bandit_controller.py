#!/usr/bin/env python3
"""
Contextual Bandit Controller for Adaptive Activation Steering.

Methodology:
Implements LinUCB (Linear Upper Confidence Bound) contextual bandit.
Selects per-input steering actions:
    Action = (layer, attention_heads, steering_magnitude) OR "no steering" (alpha=0).

Balances exploration and exploitation using confidence intervals:
    UCB_a(s) = theta_a^T s + c(t) * sqrt(s^T A_a^{-1} s)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple, Any

import torch

logger = logging.getLogger("deep_noir_rl.bandit_controller")


@dataclass
class SteeringAction:
    action_id: int
    name: str
    layer_idx: Optional[int]
    magnitude: float
    target_heads: Optional[List[int]] = None
    is_no_op: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BanditDecision:
    action: SteeringAction
    predicted_reward: float
    uncertainty: float
    ucb_score: float
    all_scores: Dict[int, float] = field(default_factory=dict)


class ContextualBanditController:
    """LinUCB Contextual Bandit for adaptive per-input steering control."""

    def __init__(
        self,
        state_dim: int = 15,
        candidate_layers: Optional[List[int]] = None,
        candidate_magnitudes: Optional[List[float]] = None,
        include_head_subsets: bool = True,
        exploration_c: float = 0.25,
        min_exploration_c: float = 0.10,
        decay_rate: float = 0.995,
        ridge_lambda: float = 1.0,
    ):
        self.state_dim = state_dim
        self.candidate_layers = candidate_layers or [12, 16, 20]
        self.candidate_magnitudes = candidate_magnitudes or [5.0, 12.0, 20.0]
        self.include_head_subsets = include_head_subsets
        self.exploration_c = exploration_c
        self.min_exploration_c = min_exploration_c
        self.decay_rate = decay_rate
        self.ridge_lambda = ridge_lambda

        self.step_count = 0
        self.actions = self._build_action_space()
        self.num_actions = len(self.actions)

        # LinUCB state matrices per arm: A_a in R^{d x d}, b_a in R^d
        self.A: List[torch.Tensor] = [
            self.ridge_lambda * torch.eye(state_dim, dtype=torch.float32)
            for _ in range(self.num_actions)
        ]
        self.b: List[torch.Tensor] = [
            torch.zeros(state_dim, dtype=torch.float32)
            for _ in range(self.num_actions)
        ]
        self.pull_counts: List[int] = [0] * self.num_actions
        self.last_verified_arm_id: Optional[int] = None
        self.last_verified_reward: Optional[float] = None

    def _build_action_space(self) -> List[SteeringAction]:
        """Constructs discrete action space including no-steering and parameter sweeps."""
        actions: List[SteeringAction] = []
        action_id = 0

        # Action 0: "No steering"
        actions.append(SteeringAction(
            action_id=action_id,
            name="No Steering",
            layer_idx=None,
            magnitude=0.0,
            target_heads=None,
            is_no_op=True,
        ))
        action_id += 1

        # Parameter grid: layer x magnitude x (all heads vs top heads)
        for layer in self.candidate_layers:
            for mag in self.candidate_magnitudes:
                # Option 1: Full residual stream
                actions.append(SteeringAction(
                    action_id=action_id,
                    name=f"L{layer}_mag{mag:.1f}_full",
                    layer_idx=layer,
                    magnitude=mag,
                    target_heads=None,
                    is_no_op=False,
                ))
                action_id += 1

                # Option 2: Targeted causal head subset
                if self.include_head_subsets:
                    actions.append(SteeringAction(
                        action_id=action_id,
                        name=f"L{layer}_mag{mag:.1f}_heads",
                        layer_idx=layer,
                        magnitude=mag,
                        target_heads=[0, 1, 2],  # Resolved dynamically or default top heads
                        is_no_op=False,
                    ))
                    action_id += 1

        logger.info(f"Initialized Contextual Bandit with {len(actions)} discrete actions.")
        return actions

    def get_current_exploration_c(self) -> float:
        """Decays exploration constant over time."""
        c = self.exploration_c * (self.decay_rate ** self.step_count)
        return max(self.min_exploration_c, c)

    def warm_start_arm(
        self,
        layer_idx: Optional[int],
        magnitude: float,
        reward: float,
        state_vector: Optional[torch.Tensor] = None,
        is_heads: bool = False,
        confidence_weight: float = 5.0,
    ) -> Optional[int]:
        """
        Warm-starts / seeds a bandit arm with prior verified evidence (e.g. from Part B awakening).
        Uses confidence_weight to ensure verified evidence is reflected in prior ridge parameters.
        """
        matching_idx = None
        if layer_idx is None or magnitude == 0.0:
            matching_idx = 0
        else:
            best_mag_diff = float("inf")
            for a_idx, action in enumerate(self.actions):
                if action.is_no_op:
                    continue
                if action.layer_idx == layer_idx:
                    if is_heads and action.target_heads is None:
                        continue
                    if not is_heads and action.target_heads is not None:
                        continue
                    diff = abs(action.magnitude - magnitude)
                    if diff < best_mag_diff:
                        best_mag_diff = diff
                        matching_idx = a_idx

        if matching_idx is not None:
            if state_vector is not None:
                s = state_vector.detach().cpu().float()
                if s.dim() == 2:
                    s = s.squeeze(0)
                s_norm = s / (torch.norm(s) + 1e-8)
                self.A[matching_idx] += confidence_weight * torch.outer(s_norm, s_norm)
                self.b[matching_idx] += (confidence_weight * reward) * s_norm
            else:
                self.b[matching_idx] += (confidence_weight * reward) * (1.0 / math.sqrt(self.state_dim))
            self.pull_counts[matching_idx] += int(confidence_weight)
            self.last_verified_arm_id = matching_idx
            self.last_verified_reward = reward
            logger.info(f"Warm-started bandit arm {matching_idx} ({self.actions[matching_idx].name}) with prior reward {reward:+.4f} (weight={confidence_weight}).")
            return matching_idx
        return None

    def select_action(
        self,
        state_vector: torch.Tensor,
        active_causal_heads_by_layer: Optional[Dict[int, List[int]]] = None,
        prompt_kind: Optional[str] = None,
        preferred_layer: Optional[int] = None,
        verified_arm_id: Optional[int] = None,
        verified_gain: float = 0.0,
    ) -> BanditDecision:
        """Selects action via LinUCB decision rule with safety-constrained exploration."""
        s = state_vector.detach().cpu().float()
        if s.dim() == 2:
            s = s.squeeze(0)
        s_norm = s / (torch.norm(s) + 1e-8)

        # An arm is verified on THIS prompt IF AND ONLY IF verified_arm_id is provided and verified_gain > 0.
        # If verified_gain <= 0 or verified_arm_id is None, no verified arm exists for this prompt.
        has_verified_arm = (verified_arm_id is not None and verified_gain > 0)
        if not has_verified_arm:
            self.last_verified_arm_id = None
            self.last_verified_reward = None

        c = self.get_current_exploration_c()
        scores: Dict[int, float] = {}
        best_action_id = 0
        best_key = (-float("inf"), -1, -1, -1, -1)
        best_pred = 0.0
        best_uncert = 0.0

        for a_idx in range(self.num_actions):
            action = self.actions[a_idx]
            A_inv = torch.linalg.pinv(self.A[a_idx])
            theta = torch.matmul(A_inv, self.b[a_idx])

            pred = torch.dot(theta, s_norm).item()
            var = torch.dot(s_norm, torch.matmul(A_inv, s_norm)).item()
            uncertainty = math.sqrt(max(0.0, var))

            if prompt_kind == "unsafe":
                if action.is_no_op:
                    # Clean baseline is known; no epistemic exploration bonus for inaction on unsafe prompts
                    score = pred
                elif has_verified_arm:
                    if a_idx == verified_arm_id:
                        # Arm with verified intervention from Part B: proven safe efficacy
                        score = pred + c * uncertainty
                    else:
                        # Safety-constrained exploration: unverified arms on unsafe prompts
                        # must not blindly gamble with high exploration bonuses over a verified safe arm.
                        # Exploration constant is strictly bounded to c_eff = min(c, 0.10) for all unverified arms.
                        c_eff = min(c, 0.10)
                        mag_cost = 0.15 * (0.7 * min(1.0, action.magnitude / 30.0))
                        score = pred + c_eff * uncertainty - mag_cost
                else:
                    # General unsafe prompt without verified arm: penalize magnitude cost on unpulled arms
                    mag_cost = 0.15 * (0.7 * min(1.0, action.magnitude / 30.0)) if self.pull_counts[a_idx] == 0 else 0.0
                    score = pred + c * uncertainty - mag_cost
            else:
                score = pred + c * uncertainty
            scores[a_idx] = score

            # Tie-breaking key: score, verified arm priority, active intervention over inaction, preferred layer, full residual over heads
            is_verified = 1 if (has_verified_arm and a_idx == verified_arm_id) else 0
            is_active = 0 if (prompt_kind == "unsafe" and action.is_no_op) else 1
            is_pref = 1 if (preferred_layer is not None and action.layer_idx == preferred_layer) else 0
            is_full = 1 if (action.target_heads is None and not action.is_no_op) else 0
            key = (round(score, 5), is_verified, is_active, is_pref, is_full)

            if key > best_key:
                best_key = key
                best_action_id = a_idx
                best_pred = pred
                best_uncert = uncertainty

        chosen_action = self.actions[best_action_id]

        # Dynamically inject detected causal heads if action targets heads
        if not chosen_action.is_no_op and chosen_action.target_heads is not None:
            if active_causal_heads_by_layer and chosen_action.layer_idx in active_causal_heads_by_layer:
                chosen_heads = active_causal_heads_by_layer[chosen_action.layer_idx]
                chosen_action = SteeringAction(
                    action_id=chosen_action.action_id,
                    name=chosen_action.name,
                    layer_idx=chosen_action.layer_idx,
                    magnitude=chosen_action.magnitude,
                    target_heads=chosen_heads,
                    is_no_op=False,
                )

        return BanditDecision(
            action=chosen_action,
            predicted_reward=best_pred,
            uncertainty=best_uncert,
            ucb_score=best_key[0],
            all_scores=scores,
        )

    def update(
        self,
        state_vector: torch.Tensor,
        action: SteeringAction,
        reward: float,
    ) -> None:
        """Updates LinUCB online ridge parameters with observed reward."""
        s = state_vector.detach().cpu().float()
        if s.dim() == 2:
            s = s.squeeze(0)
        s_norm = s / (torch.norm(s) + 1e-8)

        a_idx = action.action_id
        if 0 <= a_idx < self.num_actions:
            self.A[a_idx] += torch.outer(s_norm, s_norm)
            self.b[a_idx] += reward * s_norm
            self.pull_counts[a_idx] += 1
            self.step_count += 1
