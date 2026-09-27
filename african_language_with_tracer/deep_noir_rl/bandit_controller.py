#!/usr/bin/env python3
"""
Contextual Bandit Controller for Adaptive Activation Steering.

Methodology:
1. LinUCB (Linear Upper Confidence Bound) contextual bandit with per-arm linear models:
       UCB_a(s) = theta_a^T s + c(t) * sqrt(s^T A_a^{-1} s)
2. Shared Action-Conditioned Linear / Ridge Model:
       Allows evidence to generalize across related layers, magnitudes, sites, and sources.
3. Contextual Thompson Sampling:
       Posterior sampling theta_tilde_a ~ N(theta_hat_a, v^2 A_a^{-1})
4. Explicit RL Modes:
       - 'cold_rl': No Part B warm starts during evaluation
       - 'partb_prior_rl': Controlled expert prior mode
       - 'frozen_rl': Pure frozen policy inference (zero updates)
5. Rich Action Space:
       Action = (layer, site, direction_source, magnitude, head_mode) OR "no steering".
6. Trajectory Tracking & Regret Profiling:
       Saves per-decision uncertainty, predictions, rewards, and regret.
7. Reproducible Policy Checkpointing:
       Save and load policy parameters for frozen evaluation.
"""

from __future__ import annotations

import math
import json
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
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
    site: str = "residual"  # "residual" or "fresh_write"
    direction_source: str = "contrastive"  # "contrastive", "native", "transported_english", "awakening"
    head_mode: str = "full"  # "full" or "heads"
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
    instantaneous_regret: Optional[float] = None
    cumulative_regret: Optional[float] = None
    chosen_arm_index: Optional[int] = None


def stable_covariance_inverse(A: torch.Tensor, ridge_eps: float = 1e-4) -> torch.Tensor:
    """
    Computes strictly positive-definite matrix inverse of covariance matrix A.
    Applies adaptive Tikhonov/ridge regularization and pseudo-inverse / Cholesky fallbacks
    ensuring numerical stability and non-degeneracy, immune to NaNs/Infs and singular states.
    Preserves input device and dtype.
    """
    orig_device = A.device
    orig_dtype = A.dtype
    dim = A.shape[-1]
    A_clean = torch.nan_to_num(A.detach().cpu().float(), nan=0.0, posinf=1e4, neginf=-1e4)
    A_sym = 0.5 * (A_clean + A_clean.transpose(-1, -2))

    # Fast path: try exact Cholesky first (reg=0.0), then adaptive Tikhonov regularization
    for reg in [0.0, ridge_eps, max(1e-4, ridge_eps * 10.0), 0.01, 0.1, 1.0]:
        try:
            A_reg = A_sym if reg == 0.0 else (A_sym + reg * torch.eye(dim, dtype=A_sym.dtype))
            L = torch.linalg.cholesky(A_reg)
            inv = torch.cholesky_inverse(L)
            inv = 0.5 * (inv + inv.transpose(-1, -2))
            if not torch.isnan(inv).any() and not torch.isinf(inv).any():
                eigs = torch.linalg.eigvalsh(inv)
                if (eigs > 1e-7).all():
                    return inv.to(device=orig_device, dtype=orig_dtype)
        except (torch.linalg.LinAlgError, RuntimeError):
            continue

    # Fallback: eigenvalue decomposition with eigenvalue clamping for strict positive-definiteness
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        clamped_eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        inv = eigenvectors @ torch.diag(1.0 / clamped_eigenvalues) @ eigenvectors.transpose(-1, -2)
        inv = 0.5 * (inv + inv.transpose(-1, -2))
        if not torch.isnan(inv).any() and not torch.isinf(inv).any():
            return inv.to(device=orig_device, dtype=orig_dtype)
    except Exception:
        pass

    # Ultimate fallback: regularized pseudo-inverse with eigenvalue projection to guarantee positive-definiteness
    try:
        pinv = torch.linalg.pinv(A_sym + max(1e-3, ridge_eps) * torch.eye(dim, dtype=A_sym.dtype))
        eigs, vecs = torch.linalg.eigh(0.5 * (pinv + pinv.transpose(-1, -2)))
        clamped = torch.clamp(eigs, min=1e-5)
        inv = vecs @ torch.diag(clamped) @ vecs.transpose(-1, -2)
        return (0.5 * (inv + inv.transpose(-1, -2))).to(device=orig_device, dtype=orig_dtype)
    except Exception:
        return torch.eye(dim, dtype=orig_dtype, device=orig_device)


class ContextualBanditController:
    """Contextual Bandit controller for adaptive per-input steering control."""

    def __init__(
        self,
        state_dim: int = 15,
        candidate_layers: Optional[List[int]] = None,
        candidate_magnitudes: Optional[List[float]] = None,
        candidate_sites: Optional[List[str]] = None,
        candidate_direction_sources: Optional[List[str]] = None,
        include_head_subsets: bool = True,
        expanded_action_space: bool = False,
        exploration_c: float = 0.25,
        min_exploration_c: float = 0.10,
        decay_rate: float = 0.995,
        alpha_decay: float = 0.05,
        ridge_lambda: float = 1.0,
        algorithm: str = "linucb",  # "linucb", "action_conditioned", "thompson_sampling"
        rl_mode: str = "partb_prior_rl",   # "cold_rl", "partb_prior_rl", "frozen_rl"
        max_steering_magnitude: Optional[float] = None,
    ):
        self.state_dim = state_dim
        self.candidate_layers = candidate_layers or [8, 12]
        self.candidate_magnitudes = candidate_magnitudes or [1.0, 2.5, 5.0]
        self.candidate_sites = candidate_sites or ["residual"]
        self.candidate_direction_sources = candidate_direction_sources or ["contrastive"]
        self.include_head_subsets = include_head_subsets
        self.expanded_action_space = expanded_action_space
        self.exploration_c = exploration_c
        self.min_exploration_c = min_exploration_c
        self.decay_rate = decay_rate
        self.alpha_decay = alpha_decay
        self.ridge_lambda = ridge_lambda
        self.algorithm = algorithm.lower()
        self.rl_mode = rl_mode.lower()
        self.max_steering_magnitude = max_steering_magnitude or max(self.candidate_magnitudes, default=5.0)

        self.step_count = 0
        self.actions = self._build_action_space()
        self.num_actions = len(self.actions)

        # Per-arm LinUCB state matrices: A_a in R^{d x d}, b_a in R^d
        self.A: List[torch.Tensor] = [
            self.ridge_lambda * torch.eye(state_dim, dtype=torch.float32)
            for _ in range(self.num_actions)
        ]
        self.b: List[torch.Tensor] = [
            torch.zeros(state_dim, dtype=torch.float32)
            for _ in range(self.num_actions)
        ]
        self.pull_counts: List[int] = [0] * self.num_actions

        # Shared action-conditioned model representation (Requirement 7)
        # Action feature dim = 7 (normalized layer, normalized mag, is_heads, is_write_site, is_transported, is_awakening, is_no_op)
        self.action_feat_dim = 7
        # Salient state indices for bilinear / cross-product interaction terms
        # Captures language one-hot, tokenizer fragmentation, target layer RPD, and safety risks
        if self.state_dim >= 31:
            self.salient_state_indices = [1, 2, 5, 7, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]
        elif self.state_dim >= 15:
            self.salient_state_indices = [1, 2, 4, 5, 6, 7, 8, 9, 11, 14]
        else:
            self.salient_state_indices = list(range(self.state_dim))

        self.salient_action_indices = list(range(self.action_feat_dim))  # layer, mag, heads, site, trans, aw, no_op
        self.cross_dim = len(self.salient_state_indices) * len(self.salient_action_indices)
        self.joint_dim = self.state_dim + self.action_feat_dim + self.cross_dim
        self.A_shared = self.ridge_lambda * torch.eye(self.joint_dim, dtype=torch.float32)
        self.b_shared = torch.zeros(self.joint_dim, dtype=torch.float32)

        self.last_verified_arm_id: Optional[int] = None
        self.last_verified_reward: Optional[float] = None
        self.trajectory: List[Dict[str, Any]] = []
        self.cumulative_regret: float = 0.0

    def _build_action_space(self) -> List[SteeringAction]:
        """Constructs discrete action space including standard grid and expanded spaces."""
        actions: List[SteeringAction] = []
        action_id = 0

        # Action 0: "No steering"
        actions.append(SteeringAction(
            action_id=action_id,
            name="No Steering",
            layer_idx=None,
            magnitude=0.0,
            target_heads=None,
            site="residual",
            direction_source="contrastive",
            head_mode="full",
            is_no_op=True,
        ))
        action_id += 1

        if not self.expanded_action_space:
            # Standard grid: layer x magnitude x (full vs heads)
            for layer in self.candidate_layers:
                for mag in self.candidate_magnitudes:
                    actions.append(SteeringAction(
                        action_id=action_id,
                        name=f"L{layer}_mag{mag:.1f}_full",
                        layer_idx=layer,
                        magnitude=mag,
                        target_heads=None,
                        site="residual",
                        direction_source="contrastive",
                        head_mode="full",
                        is_no_op=False,
                    ))
                    action_id += 1

                    if self.include_head_subsets:
                        actions.append(SteeringAction(
                            action_id=action_id,
                            name=f"L{layer}_mag{mag:.1f}_heads",
                            layer_idx=layer,
                            magnitude=mag,
                            target_heads=[0, 1, 2],
                            site="residual",
                            direction_source="contrastive",
                            head_mode="heads",
                            is_no_op=False,
                        ))
                        action_id += 1
        else:
            # Expanded action space (Requirement 8):
            # layer x site x direction_source x magnitude x head_mode
            sites = self.candidate_sites if len(self.candidate_sites) > 1 else ["residual", "fresh_write"]
            sources = self.candidate_direction_sources if len(self.candidate_direction_sources) > 1 else ["contrastive", "transported_english"]

            for layer in self.candidate_layers:
                for site in sites:
                    for src in sources:
                        for mag in self.candidate_magnitudes:
                            src_tag = "trans" if "transport" in src else ("aw" if "awake" in src else "c")
                            site_tag = "wr" if site == "fresh_write" else "res"
                            actions.append(SteeringAction(
                                action_id=action_id,
                                name=f"L{layer}_{site_tag}_{src_tag}_mag{mag:.1f}_full",
                                layer_idx=layer,
                                magnitude=mag,
                                target_heads=None,
                                site=site,
                                direction_source=src,
                                head_mode="full",
                                is_no_op=False,
                            ))
                            action_id += 1

                            if self.include_head_subsets:
                                actions.append(SteeringAction(
                                    action_id=action_id,
                                    name=f"L{layer}_{site_tag}_{src_tag}_mag{mag:.1f}_heads",
                                    layer_idx=layer,
                                    magnitude=mag,
                                    target_heads=[0, 1, 2],
                                    site=site,
                                    direction_source=src,
                                    head_mode="heads",
                                    is_no_op=False,
                                ))
                                action_id += 1

        logger.info(f"Initialized Contextual Bandit with {len(actions)} discrete actions (expanded={self.expanded_action_space}).")
        return actions

    def compute_action_features(self, action: SteeringAction) -> torch.Tensor:
        """Constructs fixed-size normalized feature vector phi(a) for action a."""
        feats = torch.zeros(self.action_feat_dim, dtype=torch.float32)
        if action.is_no_op:
            feats[6] = 1.0
            return feats

        min_l = min(self.candidate_layers) if self.candidate_layers else 0
        max_l = max(self.candidate_layers) if self.candidate_layers else 24
        # Offset by 1.0 so minimum candidate layer has distinct non-zero encoding for cross-product interactions
        raw_l = (action.layer_idx - min_l + 1.0) / max(1.0, float(max_l - min_l + 1.0)) if action.layer_idx is not None else 0.5
        l_norm = min(1.0, max(0.05, float(raw_l)))
        m_norm = min(1.0, max(0.1, action.magnitude / max(1e-5, self.max_steering_magnitude)))

        feats[0] = float(l_norm)
        feats[1] = float(m_norm)
        feats[2] = 1.0 if action.target_heads is not None else 0.5
        feats[3] = 1.0 if action.site == "fresh_write" else 0.5
        feats[4] = 1.0 if "transport" in action.direction_source else 0.2
        feats[5] = 1.0 if "awake" in action.direction_source else (0.6 if "native" in action.direction_source else 0.2)
        feats[6] = 0.0
        return feats

    def compute_joint_features(self, state_norm: torch.Tensor, action: SteeringAction) -> torch.Tensor:
        """
        Constructs rich joint context-action representation psi(s, a).
        Forms bilinear / cross-product interaction terms between key state features
        (language one-hot, tokenizer fragmentation, target layer RPD, safety indicators)
        and action features (layer depth, normalized magnitude, steering site, head mode).
        Guarantees dimension and device invariance through padding / truncation.
        """
        if state_norm.shape[0] < self.state_dim:
            s_fixed = torch.zeros(self.state_dim, dtype=state_norm.dtype, device=state_norm.device)
            s_fixed[:state_norm.shape[0]] = state_norm
            state_norm = s_fixed
        elif state_norm.shape[0] > self.state_dim:
            state_norm = state_norm[:self.state_dim]

        phi_a = self.compute_action_features(action).to(device=state_norm.device, dtype=state_norm.dtype)
        s_part = state_norm[self.salient_state_indices]
        a_part = phi_a[self.salient_action_indices]
        cross_term = torch.outer(s_part, a_part).flatten()
        return torch.cat([state_norm, phi_a, cross_term], dim=0)

    def get_current_exploration_c(self) -> float:
        """
        Decays exploration constant over time:
            c(t) = c_0 / sqrt(1 + alpha_decay * t)
        Prevents chronic over-exploration late in training while maintaining robust initial discovery.
        """
        c = self.exploration_c / math.sqrt(1.0 + self.alpha_decay * self.step_count)
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
        Requirement 2: Strictly disabled in 'cold_rl' and 'frozen_rl' modes to prevent test leakage.
        """
        if self.rl_mode in ("cold_rl", "frozen_rl"):
            return None

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
                if s.shape[0] < self.state_dim:
                    s_pad = torch.zeros(self.state_dim, dtype=torch.float32)
                    s_pad[:s.shape[0]] = s
                    s = s_pad
                elif s.shape[0] > self.state_dim:
                    s = s[:self.state_dim]
                s_norm = s / (torch.norm(s) + 1e-8)
                self.A[matching_idx] += confidence_weight * torch.outer(s_norm, s_norm)
                self.b[matching_idx] += (confidence_weight * reward) * s_norm

                # Also update shared action-conditioned model
                z = self.compute_joint_features(s_norm, self.actions[matching_idx])
                self.A_shared += confidence_weight * torch.outer(z, z)
                self.b_shared += (confidence_weight * reward) * z
            else:
                self.b[matching_idx] += (confidence_weight * reward) * (1.0 / math.sqrt(self.state_dim))
                dummy_s = torch.zeros(self.state_dim, dtype=torch.float32)
                z = self.compute_joint_features(dummy_s, self.actions[matching_idx])
                self.A_shared += confidence_weight * torch.outer(z, z)
                self.b_shared += (confidence_weight * reward) * z

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
        oracle_best_action_id: Optional[int] = None,
        deterministic: bool = False,
    ) -> BanditDecision:
        """Selects action via LinUCB, Shared Action-Conditioned Model, or Thompson Sampling."""
        s = state_vector.detach().cpu().float()
        if s.dim() == 2:
            s = s.squeeze(0)
        if s.shape[0] < self.state_dim:
            s_pad = torch.zeros(self.state_dim, dtype=torch.float32)
            s_pad[:s.shape[0]] = s
            s = s_pad
        elif s.shape[0] > self.state_dim:
            s = s[:self.state_dim]
        s_norm = s / (torch.norm(s) + 1e-8)

        # In cold_rl or frozen_rl, ignore any verified arm injected at test time
        if self.rl_mode in ("cold_rl", "frozen_rl"):
            has_verified_arm = False
            verified_arm_id = None
            self.last_verified_arm_id = None
            self.last_verified_reward = None
        else:
            has_verified_arm = (verified_arm_id is not None and verified_gain > 0)
            if not has_verified_arm:
                self.last_verified_arm_id = None
                self.last_verified_reward = None

        if deterministic or self.rl_mode == "frozen_rl":
            c = 0.0
        else:
            c = self.get_current_exploration_c()

        scores: Dict[int, float] = {}
        best_action_id = 0
        best_key = (-float("inf"), -1, -1, -1, -1)
        best_pred = 0.0
        best_uncert = 0.0

        # Precompute shared inverse if using shared action-conditioned model
        shared_theta = None
        A_shared_inv = None
        if self.algorithm == "action_conditioned":
            A_shared_inv = stable_covariance_inverse(self.A_shared, ridge_eps=self.ridge_lambda * 1e-3)
            shared_theta = torch.matmul(A_shared_inv, self.b_shared)

        for a_idx in range(self.num_actions):
            action = self.actions[a_idx]

            if self.algorithm == "action_conditioned" and shared_theta is not None and A_shared_inv is not None:
                # Shared action-conditioned model (Requirement 7)
                z = self.compute_joint_features(s_norm, action)
                pred = torch.dot(shared_theta, z).item()
                var = torch.dot(z, torch.matmul(A_shared_inv, z)).item()
                uncertainty = math.sqrt(max(0.0, var))
            elif self.algorithm == "thompson_sampling":
                # Contextual Thompson Sampling (Requirement 7)
                A_inv = stable_covariance_inverse(self.A[a_idx], ridge_eps=self.ridge_lambda * 1e-3)
                mu = torch.matmul(A_inv, self.b[a_idx])
                pred = torch.dot(mu, s_norm).item()
                var = torch.dot(s_norm, torch.matmul(A_inv, s_norm)).item()
                uncertainty = math.sqrt(max(0.0, var))

                if deterministic or self.rl_mode == "frozen_rl" or c <= 0.0:
                    theta_sample = mu
                else:
                    # Posterior sampling scaled by annealed exploration factor c(t)
                    cov = (c ** 2) * A_inv
                    cov = 0.5 * (cov + cov.T) + 1e-5 * torch.eye(self.state_dim)
                    try:
                        dist = torch.distributions.MultivariateNormal(mu, cov)
                        theta_sample = dist.sample()
                    except Exception:
                        theta_sample = mu

                thompson_pred = torch.dot(theta_sample, s_norm).item()
            else:
                # Standard LinUCB
                A_inv = stable_covariance_inverse(self.A[a_idx], ridge_eps=self.ridge_lambda * 1e-3)
                theta = torch.matmul(A_inv, self.b[a_idx])
                pred = torch.dot(theta, s_norm).item()
                var = torch.dot(s_norm, torch.matmul(A_inv, s_norm)).item()
                uncertainty = math.sqrt(max(0.0, var))

            # Unpulled arm exploration bonus: applies to independent models (LinUCB) where arms don't share parameters
            unpulled_bonus = c if (self.algorithm not in ("action_conditioned", "thompson_sampling") and self.pull_counts[a_idx] == 0 and not action.is_no_op and prompt_kind == "unsafe" and not deterministic and self.rl_mode != "frozen_rl") else 0.0

            if self.algorithm == "thompson_sampling":
                score = thompson_pred
            elif prompt_kind == "unsafe":
                if action.is_no_op:
                    score = pred
                elif has_verified_arm:
                    if a_idx == verified_arm_id:
                        score = pred + c * uncertainty
                    else:
                        c_eff = min(c, 0.10)
                        mag_cost = 0.15 * (0.7 * min(1.0, action.magnitude / max(1e-5, self.max_steering_magnitude)))
                        score = pred + c_eff * uncertainty - mag_cost
                else:
                    score = pred + c * uncertainty + unpulled_bonus
            else:
                score = pred + c * uncertainty
            scores[a_idx] = score

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
                    site=chosen_action.site,
                    direction_source=chosen_action.direction_source,
                    head_mode="heads",
                    is_no_op=False,
                )

        inst_regret = None
        if oracle_best_action_id is not None:
            oracle_score = scores.get(oracle_best_action_id, best_pred)
            inst_regret = max(0.0, oracle_score - best_pred)
            self.cumulative_regret += inst_regret

        decision = BanditDecision(
            action=chosen_action,
            predicted_reward=best_pred,
            uncertainty=best_uncert,
            ucb_score=best_key[0],
            all_scores=scores,
            instantaneous_regret=inst_regret,
            cumulative_regret=self.cumulative_regret if oracle_best_action_id is not None else None,
            chosen_arm_index=best_action_id,
        )

        # Record trajectory (Requirement 17)
        self.trajectory.append({
            "step_index": self.step_count,
            "action_id": chosen_action.action_id,
            "action_name": chosen_action.name,
            "layer_idx": chosen_action.layer_idx,
            "magnitude": chosen_action.magnitude,
            "site": chosen_action.site,
            "direction_source": chosen_action.direction_source,
            "predicted_reward": round(best_pred, 5),
            "uncertainty": round(best_uncert, 5),
            "ucb_score": round(best_key[0], 5),
            "instantaneous_regret": round(inst_regret, 5) if inst_regret is not None else None,
            "cumulative_regret": round(self.cumulative_regret, 5) if inst_regret is not None else None,
            "rl_mode": self.rl_mode,
            "algorithm": self.algorithm,
        })

        return decision

    def update(
        self,
        state_vector: torch.Tensor,
        action: SteeringAction,
        reward: float,
    ) -> None:
        """
        Updates LinUCB online ridge parameters with observed reward.
        Requirement 1 & 2: In 'frozen_rl' mode, updates are strictly skipped.
        """
        if self.rl_mode == "frozen_rl":
            return

        s = state_vector.detach().cpu().float()
        if s.dim() == 2:
            s = s.squeeze(0)
        if s.shape[0] < self.state_dim:
            s_pad = torch.zeros(self.state_dim, dtype=torch.float32)
            s_pad[:s.shape[0]] = s
            s = s_pad
        elif s.shape[0] > self.state_dim:
            s = s[:self.state_dim]
        s_norm = s / (torch.norm(s) + 1e-8)

        a_idx = action.action_id
        if 0 <= a_idx < self.num_actions:
            self.A[a_idx] += torch.outer(s_norm, s_norm)
            self.b[a_idx] += reward * s_norm
            self.pull_counts[a_idx] += 1
            self.step_count += 1

            # Update shared action-conditioned model
            z = self.compute_joint_features(s_norm, self.actions[a_idx])
            self.A_shared += torch.outer(z, z)
            self.b_shared += reward * z

            # Update latest trajectory record with observed reward
            if self.trajectory and self.trajectory[-1]["action_id"] == a_idx:
                self.trajectory[-1]["observed_reward"] = round(reward, 5)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Returns recorded learning trajectory."""
        return list(self.trajectory)

    def save_trajectory(self, path: str) -> None:
        """Saves learning trajectory records to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.trajectory, f, indent=2)

    def save_policy(self, path: str) -> Dict[str, Any]:
        """
        Serializes bandit policy for frozen reload evaluation (Requirement 23).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        A_serial = [mat.tolist() for mat in self.A]
        b_serial = [vec.tolist() for vec in self.b]
        A_sh_serial = self.A_shared.tolist()
        b_sh_serial = self.b_shared.tolist()

        payload = {
            "algorithm": self.algorithm,
            "rl_mode": self.rl_mode,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "step_count": self.step_count,
            "pull_counts": self.pull_counts,
            "exploration_c": self.exploration_c,
            "min_exploration_c": self.min_exploration_c,
            "decay_rate": self.decay_rate,
            "alpha_decay": self.alpha_decay,
            "ridge_lambda": self.ridge_lambda,
            "max_steering_magnitude": self.max_steering_magnitude,
            "expanded_action_space": self.expanded_action_space,
            "cross_dim": self.cross_dim,
            "joint_dim": self.joint_dim,
            "actions": [a.to_dict() for a in self.actions],
            "A": A_serial,
            "b": b_serial,
            "A_shared": A_sh_serial,
            "b_shared": b_sh_serial,
            "cumulative_regret": float(self.cumulative_regret),
        }
        raw_json = json.dumps(payload, sort_keys=True)
        sha = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        payload["config_hash"] = sha

        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Saved bandit policy to {path} (hash={sha[:8]}).")
        return {"path": str(p), "config_hash": sha}

    def load_policy(self, path: str, freeze: bool = True) -> None:
        """
        Loads serialized policy checkpoint and optionally freezes updates (Requirement 23).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.algorithm = data.get("algorithm", self.algorithm)
        self.step_count = data.get("step_count", 0)
        if "state_dim" in data:
            self.state_dim = data["state_dim"]
            # Reconstruct salient state indices if state_dim changed
            if self.state_dim >= 31:
                self.salient_state_indices = [1, 2, 5, 7, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26]
            elif self.state_dim >= 15:
                self.salient_state_indices = [1, 2, 4, 5, 6, 7, 8, 9, 11, 14]
            else:
                self.salient_state_indices = list(range(self.state_dim))
            self.cross_dim = len(self.salient_state_indices) * len(self.salient_action_indices)
            self.joint_dim = self.state_dim + self.action_feat_dim + self.cross_dim

        if "actions" in data:
            self.actions = [SteeringAction(**a) for a in data["actions"]]
            self.num_actions = len(self.actions)
        if "expanded_action_space" in data:
            self.expanded_action_space = data["expanded_action_space"]
        if "exploration_c" in data:
            self.exploration_c = data["exploration_c"]
        if "min_exploration_c" in data:
            self.min_exploration_c = data["min_exploration_c"]
        if "decay_rate" in data:
            self.decay_rate = data["decay_rate"]
        if "alpha_decay" in data:
            self.alpha_decay = data["alpha_decay"]
        if "ridge_lambda" in data:
            self.ridge_lambda = data["ridge_lambda"]
        if "max_steering_magnitude" in data:
            self.max_steering_magnitude = data["max_steering_magnitude"]

        dev = self.A[0].device if self.A else torch.device("cpu")
        self.pull_counts = data.get("pull_counts", [0] * self.num_actions)
        self.A = [torch.tensor(mat, dtype=torch.float32, device=dev) for mat in data["A"]]
        self.b = [torch.tensor(vec, dtype=torch.float32, device=dev) for vec in data["b"]]
        if "A_shared" in data and "b_shared" in data:
            self.A_shared = torch.tensor(data["A_shared"], dtype=torch.float32, device=dev)
            self.b_shared = torch.tensor(data["b_shared"], dtype=torch.float32, device=dev)
            self.joint_dim = self.A_shared.shape[0]
        if "cumulative_regret" in data:
            self.cumulative_regret = float(data["cumulative_regret"])

        if freeze:
            self.rl_mode = "frozen_rl"
        logger.info(f"Loaded bandit policy from {path} (frozen={freeze}, actions={self.num_actions}, state_dim={self.state_dim}).")

    def get_temporal_learning_summary(self, epsilon: float = 0.05) -> Dict[str, Any]:
        """
        Computes temporal learning dynamics across the bandit decision trajectory:
        early vs. late reward, early vs. late regret, cumulative regret, % epsilon-optimal,
        and action distribution shifts.
        """
        steps = self.trajectory
        n = len(steps)
        if n == 0:
            return {
                "total_episodes": 0,
                "early_episodes": 0,
                "late_episodes": 0,
                "early_mean_reward": 0.0,
                "late_mean_reward": 0.0,
                "reward_gain": 0.0,
                "early_mean_regret": 0.0,
                "late_mean_regret": 0.0,
                "regret_reduction": 0.0,
                "cumulative_regret": 0.0,
                "pct_epsilon_optimal": 0.0,
                "early_action_distribution": {},
                "late_action_distribution": {},
                "action_frequency_shifts": {},
            }

        split_idx = max(1, n // 2)
        early_steps = steps[:split_idx]
        late_steps = steps[split_idx:] if n > 1 else steps

        early_r = [s["observed_reward"] for s in early_steps if s.get("observed_reward") is not None]
        late_r = [s["observed_reward"] for s in late_steps if s.get("observed_reward") is not None]
        early_mean_r = float(sum(early_r) / len(early_r)) if early_r else 0.0
        late_mean_r = float(sum(late_r) / len(late_r)) if late_r else 0.0

        early_reg = [s["instantaneous_regret"] for s in early_steps if s.get("instantaneous_regret") is not None]
        late_reg = [s["instantaneous_regret"] for s in late_steps if s.get("instantaneous_regret") is not None]
        all_reg = [s["instantaneous_regret"] for s in steps if s.get("instantaneous_regret") is not None]
        early_mean_reg = float(sum(early_reg) / len(early_reg)) if early_reg else 0.0
        late_mean_reg = float(sum(late_reg) / len(late_reg)) if late_reg else 0.0
        cum_reg = float(sum(all_reg)) if all_reg else float(self.cumulative_regret)
        eps_cnt = sum(1 for r in all_reg if r <= epsilon)
        pct_eps = float((eps_cnt / len(all_reg)) * 100.0) if all_reg else 0.0

        all_actions = sorted(set(s.get("action_name", "") for s in steps if "action_name" in s))
        early_dist = {a: round(sum(1 for s in early_steps if s.get("action_name") == a) / len(early_steps), 4) for a in all_actions}
        late_dist = {a: round(sum(1 for s in late_steps if s.get("action_name") == a) / len(late_steps), 4) for a in all_actions}
        act_shifts = {a: round(late_dist.get(a, 0.0) - early_dist.get(a, 0.0), 4) for a in all_actions}

        return {
            "total_episodes": n,
            "early_episodes": len(early_steps),
            "late_episodes": len(late_steps),
            "early_mean_reward": round(early_mean_r, 5),
            "late_mean_reward": round(late_mean_r, 5),
            "reward_gain": round(late_mean_r - early_mean_r, 5),
            "early_mean_regret": round(early_mean_reg, 5),
            "late_mean_regret": round(late_mean_reg, 5),
            "regret_reduction": round(early_mean_reg - late_mean_reg, 5),
            "cumulative_regret": round(cum_reg, 5),
            "pct_epsilon_optimal": round(pct_eps, 2),
            "early_action_distribution": early_dist,
            "late_action_distribution": late_dist,
            "action_frequency_shifts": act_shifts,
        }
