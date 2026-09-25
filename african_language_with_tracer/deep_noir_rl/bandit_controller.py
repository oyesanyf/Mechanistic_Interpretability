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
        exploration_c: float = 1.25,
        min_exploration_c: float = 0.20,
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

    def select_action(
        self,
        state_vector: torch.Tensor,
        active_causal_heads_by_layer: Optional[Dict[int, List[int]]] = None,
    ) -> BanditDecision:
        """Selects action via LinUCB decision rule."""
        s = state_vector.detach().cpu().float()
        if s.dim() == 2:
            s = s.squeeze(0)

        c = self.get_current_exploration_c()
        scores: Dict[int, float] = {}
        best_action_id = 0
        best_score = -float("inf")
        best_pred = 0.0
        best_uncert = 0.0

        for a_idx in range(self.num_actions):
            A_inv = torch.linalg.pinv(self.A[a_idx])
            theta = torch.matmul(A_inv, self.b[a_idx])

            pred = torch.dot(theta, s).item()
            var = torch.dot(s, torch.matmul(A_inv, s)).item()
            uncertainty = math.sqrt(max(0.0, var))

            score = pred + c * uncertainty
            scores[a_idx] = score

            if score > best_score:
                best_score = score
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
            ucb_score=best_score,
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

        a_idx = action.action_id
        if 0 <= a_idx < self.num_actions:
            self.A[a_idx] += torch.outer(s, s)
            self.b[a_idx] += reward * s
            self.pull_counts[a_idx] += 1
            self.step_count += 1
