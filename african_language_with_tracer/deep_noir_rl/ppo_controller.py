#!/usr/bin/env python3
"""
Constrained PPO (Proximal Policy Optimization) Controller for Adaptive Steering.

Methodology:
1. Actor-Critic Architecture:
   - Policy network pi_theta(a | s) over discrete steering actions
   - Value network V_phi(s) for task reward
   - Cost critic V_psi(s) for safety constraint evaluation (injection risk & capability loss)
2. Adaptive Lagrangian Multiplier lambda:
   Enforces E[Cost] <= d_limit by dynamically adjusting penalty weights:
       lambda <- max(0, lambda + eta * (mean_cost - d_limit))
3. Clipped Surrogate Objective:
   Maximizes reward advantages while penalizing constraint violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .bandit_controller import SteeringAction

logger = logging.getLogger("deep_noir_rl.ppo_controller")


@dataclass
class PPODecision:
    action: SteeringAction
    action_index: int
    log_prob: float
    value_estimate: float
    cost_estimate: float
    action_probabilities: List[float] = field(default_factory=list)


@dataclass
class PPOTypedTransition:
    state: torch.Tensor
    action_index: int
    reward: float
    cost: float
    log_prob: float
    value: float
    cost_val: float


class ActorCriticNetwork(nn.Module):
    """Joint Actor-Critic neural network with dual reward and cost value heads."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.reward_critic_head = nn.Linear(hidden_dim, 1)
        self.cost_critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        feat = self.shared(state)
        action_logits = self.actor_head(feat)
        reward_val = self.reward_critic_head(feat)
        cost_val = self.cost_critic_head(feat)
        return action_logits, reward_val, cost_val


class ConstrainedPPOController:
    """Constrained PPO controller enforcing safety barriers via adaptive Lagrangian multipliers."""

    def __init__(
        self,
        state_dim: int = 15,
        actions: Optional[List[SteeringAction]] = None,
        candidate_layers: Optional[List[int]] = None,
        candidate_magnitudes: Optional[List[float]] = None,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.20,
        max_safety_cost_limit: float = 0.35,
        lagrangian_lr: float = 0.05,
        initial_lagrangian: float = 1.0,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if actions is not None:
            self.actions = actions
        else:
            from .bandit_controller import ContextualBanditController
            dummy_bandit = ContextualBanditController(
                state_dim=state_dim,
                candidate_layers=candidate_layers or [12, 16, 20],
                candidate_magnitudes=candidate_magnitudes or [5.0, 12.0, 20.0],
            )
            self.actions = dummy_bandit.actions

        self.num_actions = len(self.actions)
        self.network = ActorCriticNetwork(state_dim, self.num_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.d_limit = max_safety_cost_limit
        self.lagrangian_lr = lagrangian_lr
        self.lagrangian_lambda = initial_lagrangian

        self.buffer: List[PPOTypedTransition] = []

    def select_action(
        self,
        state_vector: torch.Tensor,
        deterministic: bool = False,
        active_causal_heads_by_layer: Optional[Dict[int, List[int]]] = None,
    ) -> PPODecision:
        """Selects action and records value and safety cost estimates."""
        s = state_vector.to(self.device).float()
        if s.dim() == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            logits, r_val, c_val = self.network(s)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                a_idx = int(torch.argmax(probs, dim=-1).item())
            else:
                a_idx = int(dist.sample().item())

            log_prob = float(dist.log_prob(torch.tensor(a_idx, device=self.device)).item())
            reward_est = float(r_val.item())
            cost_est = float(c_val.item())
            probs_list = [float(p.item()) for p in probs[0]]

        chosen_action = self.actions[a_idx]
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

        return PPODecision(
            action=chosen_action,
            action_index=a_idx,
            log_prob=log_prob,
            value_estimate=reward_est,
            cost_estimate=cost_est,
            action_probabilities=probs_list,
        )

    def record_step(
        self,
        state: torch.Tensor,
        action_index: int,
        reward: float,
        cost: float,
        log_prob: float,
        value: float,
        cost_val: float,
    ) -> None:
        """Appends step transition to rollout buffer."""
        self.buffer.append(PPOTypedTransition(
            state=state.detach().cpu(),
            action_index=action_index,
            reward=reward,
            cost=cost,
            log_prob=log_prob,
            value=value,
            cost_val=cost_val,
        ))

    def update(self, ppo_epochs: int = 4, batch_size: int = 16) -> Dict[str, float]:
        """Runs Constrained PPO updates with adaptive Lagrangian multiplier."""
        if not self.buffer:
            return {}

        states = torch.stack([t.state for t in self.buffer]).to(self.device).float()
        actions = torch.tensor([t.action_index for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], device=self.device, dtype=torch.float32)
        rewards = [t.reward for t in self.buffer]
        costs = [t.cost for t in self.buffer]

        # Compute discounted returns
        reward_returns = []
        discounted_r = 0.0
        for r in reversed(rewards):
            discounted_r = r + self.gamma * discounted_r
            reward_returns.insert(0, discounted_r)
        reward_returns_t = torch.tensor(reward_returns, device=self.device, dtype=torch.float32)

        cost_returns = []
        discounted_c = 0.0
        for c in reversed(costs):
            discounted_c = c + self.gamma * discounted_c
            cost_returns.insert(0, discounted_c)
        cost_returns_t = torch.tensor(cost_returns, device=self.device, dtype=torch.float32)

        mean_cost = sum(costs) / max(1, len(costs))

        # Update Lagrangian multiplier: lambda <- max(0, lambda + eta * (mean_cost - d_limit))
        cost_violation = mean_cost - self.d_limit
        self.lagrangian_lambda = max(0.0, self.lagrangian_lambda + self.lagrangian_lr * cost_violation)

        total_loss_accum = 0.0
        for _ in range(ppo_epochs):
            logits, r_vals, c_vals = self.network(states)
            dist = Categorical(F.softmax(logits, dim=-1))
            new_log_probs = dist.log_prob(actions)

            # Ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Advantages
            adv_reward = reward_returns_t.view(-1) - r_vals.view(-1).detach()
            adv_cost = cost_returns_t.view(-1) - c_vals.view(-1).detach()

            # Net constrained advantage
            net_adv = adv_reward - self.lagrangian_lambda * adv_cost
            if net_adv.numel() > 1:
                adv_std = net_adv.std()
                if not torch.isnan(adv_std) and adv_std > 1e-6:
                    net_adv = (net_adv - net_adv.mean()) / (adv_std + 1e-8)
                else:
                    net_adv = net_adv - net_adv.mean()
            else:
                net_adv = net_adv - net_adv.mean()

            # Clipped surrogate objective
            surr1 = ratios * net_adv
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * net_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value losses
            v_loss_reward = F.mse_loss(r_vals.view(-1), reward_returns_t.view(-1))
            v_loss_cost = F.mse_loss(c_vals.view(-1), cost_returns_t.view(-1))

            loss = policy_loss + 0.5 * v_loss_reward + 0.5 * v_loss_cost

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss_accum += float(loss.item())

        self.buffer.clear()

        return {
            "mean_reward": round(sum(rewards) / max(1, len(rewards)), 5),
            "mean_cost": round(mean_cost, 5),
            "lagrangian_lambda": round(self.lagrangian_lambda, 5),
            "cost_violation": round(cost_violation, 5),
            "training_loss": round(total_loss_accum / max(1, ppo_epochs), 5),
        }
