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
3. Independent Prompt Handling (Requirement 5):
   Independent prompts are one-step contextual bandits: gamma = 0.0 (or done=True boundary per prompt),
   eliminating artificial temporal discounting across unrelated multilingual prompts.
4. Sequential Steering MDP Formulation (Requirement 6):
   Sequential trajectory formulation:
       Stage 0 (Site Selection) -> Observe intermediate state ->
       Stage 1 (Direction Selection) -> Observe state ->
       Stage 2 (Magnitude & Head Mode) -> Terminal multi-objective reward.
5. Reproducible Policy Checkpointing (Requirement 23):
   Save and load policy parameters for frozen reload evaluation.
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    done: bool = True  # Explicit episodic boundary for independent prompts (Requirement 5)


class ActorCriticNetwork(nn.Module):
    """Joint Actor-Critic neural network with dual reward and cost value heads."""

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

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
        gamma: float = 0.0,  # Requirement 5: Default gamma=0.0 for independent prompt contextual optimization
        clip_eps: float = 0.20,
        max_safety_cost_limit: float = 0.35,
        lagrangian_lr: float = 0.05,
        initial_lagrangian: float = 1.0,
        device: Optional[str] = None,
        mode: str = "contextual",  # "contextual" (one-step) or "sequential" (multi-stage MDP)
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode.lower()

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
        self.trajectory_history: List[Dict[str, Any]] = []

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
                    site=chosen_action.site,
                    direction_source=chosen_action.direction_source,
                    head_mode="heads",
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
        done: bool = True,  # Explicit episode boundary (Requirement 5)
    ) -> None:
        """Appends step transition to rollout buffer with episode boundary tracking."""
        self.buffer.append(PPOTypedTransition(
            state=state.detach().cpu(),
            action_index=action_index,
            reward=reward,
            cost=cost,
            log_prob=log_prob,
            value=value,
            cost_val=cost_val,
            done=done,
        ))

    def update(self, ppo_epochs: int = 4, batch_size: int = 16) -> Dict[str, float]:
        """
        Runs Constrained PPO updates with adaptive Lagrangian multiplier.
        Requirement 5: Correctly computes returns without artificial temporal coupling across independent prompts.
        """
        if not self.buffer:
            return {}

        states = torch.stack([t.state for t in self.buffer]).to(self.device).float()
        actions = torch.tensor([t.action_index for t in self.buffer], device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], device=self.device, dtype=torch.float32)
        rewards = [t.reward for t in self.buffer]
        costs = [t.cost for t in self.buffer]

        # Compute returns respecting episode boundaries (Requirement 5)
        if self.gamma == 0.0 or all(t.done for t in self.buffer):
            # One-step contextual decisions: immediate return with zero cross-prompt discounting
            reward_returns = list(rewards)
            cost_returns = list(costs)
        else:
            # Sequential MDP trajectory discounting
            reward_returns = []
            cost_returns = []
            discounted_r = 0.0
            discounted_c = 0.0
            for t in reversed(self.buffer):
                if t.done:
                    discounted_r = 0.0
                    discounted_c = 0.0
                discounted_r = t.reward + self.gamma * discounted_r
                discounted_c = t.cost + self.gamma * discounted_c
                reward_returns.insert(0, discounted_r)
                cost_returns.insert(0, discounted_c)

        reward_returns_t = torch.tensor(reward_returns, device=self.device, dtype=torch.float32)
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

            # Probability ratios
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

        metrics = {
            "mean_reward": round(sum(rewards) / max(1, len(rewards)), 5),
            "mean_cost": round(mean_cost, 5),
            "lagrangian_lambda": round(self.lagrangian_lambda, 5),
            "cost_violation": round(cost_violation, 5),
            "training_loss": round(total_loss_accum / max(1, ppo_epochs), 5),
        }
        self.trajectory_history.append(metrics)
        return metrics

    def save_policy(self, path: str) -> Dict[str, Any]:
        """Serializes PPO network and Lagrangian optimizer state (Requirement 23)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        state_dict_cpu = {k: v.cpu() for k, v in self.network.state_dict().items()}
        torch.save({
            "state_dict": state_dict_cpu,
            "optimizer_state": self.optimizer.state_dict(),
            "lagrangian_lambda": self.lagrangian_lambda,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "hidden_dim": self.hidden_dim,
            "gamma": self.gamma,
            "d_limit": self.d_limit,
            "actions": [a.to_dict() for a in self.actions],
        }, str(p))

        logger.info(f"Saved PPO policy checkpoint to {path}.")
        return {"path": str(p)}

    def load_policy(self, path: str) -> None:
        """Loads serialized PPO policy checkpoint (Requirement 23)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.lagrangian_lambda = checkpoint.get("lagrangian_lambda", self.lagrangian_lambda)
        self.gamma = checkpoint.get("gamma", self.gamma)
        logger.info(f"Loaded PPO policy checkpoint from {path}.")


class SequentialSteeringMDP:
    """
    Sequential Steering MDP Environment (Requirement 6).
    Decomposes the steering decision into a 3-step sequential Markov decision process:
        Step 0: Choose intervention site (layer & site: residual vs fresh_write)
        Step 1: Choose steering direction source (contrastive, native, transported, awakening)
        Step 2: Choose magnitude & attention head mode -> Execute -> Receive multi-objective reward
    Allows PPO actor/critic and cost critic to learn genuine multi-step trajectories.
    """

    def __init__(
        self,
        candidate_layers: List[int],
        candidate_magnitudes: List[float],
        candidate_sites: Optional[List[str]] = None,
        candidate_sources: Optional[List[str]] = None,
    ):
        self.candidate_layers = candidate_layers
        self.candidate_magnitudes = candidate_magnitudes
        self.sites = candidate_sites or ["residual", "fresh_write"]
        self.sources = candidate_sources or ["contrastive", "transported_english", "awakening"]

        self.current_step = 0
        self.chosen_layer: Optional[int] = None
        self.chosen_site: str = "residual"
        self.chosen_source: str = "contrastive"
        self.chosen_magnitude: float = 0.0
        self.chosen_head_mode: str = "full"

    def reset(self, initial_state: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Resets MDP for a new prompt. Returns (state, stage_idx)."""
        self.current_step = 0
        self.chosen_layer = None
        self.chosen_site = "residual"
        self.chosen_source = "contrastive"
        self.chosen_magnitude = 0.0
        self.chosen_head_mode = "full"
        return initial_state, self.current_step

    def step(
        self,
        action_idx: int,
        current_state: torch.Tensor,
        action_space: List[SteeringAction],
        final_reward_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, float, float, bool, Dict[str, Any]]:
        """
        Executes transition in sequential MDP:
        Returns (next_state, reward, safety_cost, done, info).
        """
        action = action_space[action_idx]
        self.current_step += 1

        if self.current_step == 1:
            # Stage 0 complete: site selected
            self.chosen_layer = action.layer_idx
            self.chosen_site = action.site
            # Intermediate step: small exploration cost, no terminal reward yet
            reward = 0.0
            cost = 0.0
            done = False
            next_state = current_state.clone()
            # Feature perturbation reflecting chosen site
            next_state[4] = float(action.layer_idx or 0) / 32.0
            return next_state, reward, cost, done, {"stage": "site_selected", "layer": self.chosen_layer}

        elif self.current_step == 2:
            # Stage 1 complete: direction source selected
            self.chosen_source = action.direction_source
            reward = 0.0
            cost = 0.0
            done = False
            next_state = current_state.clone()
            next_state[7] = 0.8 if "transport" in self.chosen_source else 0.5
            return next_state, reward, cost, done, {"stage": "source_selected", "source": self.chosen_source}

        else:
            # Stage 2 complete: magnitude & head mode selected -> Terminal step
            self.chosen_magnitude = action.magnitude
            self.chosen_head_mode = action.head_mode
            done = True
            reward = final_reward_fn() if callable(final_reward_fn) else 0.5
            cost = 0.1 if action.magnitude > 5.0 else 0.0
            return current_state, reward, cost, done, {
                "stage": "terminal",
                "layer": self.chosen_layer,
                "site": self.chosen_site,
                "source": self.chosen_source,
                "magnitude": self.chosen_magnitude,
            }
