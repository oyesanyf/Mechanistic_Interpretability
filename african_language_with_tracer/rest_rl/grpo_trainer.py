#!/usr/bin/env python3
"""
ReST-GRPO Trainer: Group-Relative Policy Optimization.

Implements Stage 1 ReST-RL self-training:
- Computes group-relative normalized advantages without a critic/value model:
    A_i = (R_i - mean(R)) / (std(R) + eps)
- Evaluates token-level / sequence-level log-probabilities under policy and reference models.
- Clips surrogate policy objective:
    L_clip = min(r_i * A_i, clip(r_i, 1 - eps, 1 + eps) * A_i)
- Penalizes KL divergence against reference policy pi_ref:
    D_KL = pi_ref / pi_theta - log(pi_ref / pi_theta) - 1
- Handles gradient accumulation, clipping, and AdamW optimizer updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import PromptGroupSample, CompletionSample

logger = logging.getLogger("rest_rl.grpo_trainer")


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GRPOTrainerConfig:
    """Hyperparameters for ReST-GRPO training."""
    clip_eps: float = 0.2
    kl_coeff: float = 0.04
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    eps: float = 1e-4
    use_token_level_loss: bool = True
    device: Optional[str] = None


@dataclass
class GRPOLossMetrics:
    """Metrics recorded from a single GRPO policy loss computation."""
    total_loss: float
    surrogate_loss: float
    kl_divergence: float
    mean_reward: float
    std_reward: float
    clip_fraction: float
    mean_advantage: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_loss": self.total_loss,
            "surrogate_loss": self.surrogate_loss,
            "kl_divergence": self.kl_divergence,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "clip_fraction": self.clip_fraction,
            "mean_advantage": self.mean_advantage,
        }


# ---------------------------------------------------------------------------
# Core GRPO Math Functions (Independent of LLM Backbone)
# ---------------------------------------------------------------------------

def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes group-relative normalized advantages:
        A_i = (R_i - mean(R)) / (std(R) + eps)
    If all rewards in the group are identical or rewards has <= 1 element, returns zero tensor.
    Sanitizes NaNs or Infs to prevent gradient corruption.
    """
    if rewards.numel() == 0:
        return torch.empty(0, dtype=torch.float32)

    # Sanitize any NaNs or Infs
    valid_rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    if valid_rewards.numel() <= 1:
        return torch.zeros_like(valid_rewards)

    mean_r = valid_rewards.mean()
    std_r = valid_rewards.std(unbiased=False)

    if std_r < 1e-7:
        return torch.zeros_like(valid_rewards)

    return (valid_rewards - mean_r) / (std_r + eps)


def compute_surrogate_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes clipped surrogate objective:
        ratio = exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantage
        obj = min(surr1, surr2)
    Clamps log-ratios to [-20.0, 20.0] for numerical stability.
    Returns:
        (surrogate_loss_to_maximize, clip_fraction)
    """
    # ratio: [batch_size, seq_len] or [batch_size]
    log_ratio = torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Broadcast advantage if necessary
    if advantages.dim() == 1 and log_probs.dim() == 2:
        adv = advantages.unsqueeze(1)
    else:
        adv = advantages

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(surr1, surr2)

    # Compute clip fraction
    clipped = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
    if mask is not None:
        clip_fraction = (clipped.float() * mask).sum() / (mask.sum() + 1e-8)
        surrogate_mean = (surrogate * mask).sum() / (mask.sum() + 1e-8)
    else:
        clip_fraction = clipped.float().mean()
        surrogate_mean = surrogate.mean()

    return surrogate_mean, clip_fraction


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes numerically stable KL divergence penalty between policy and reference:
        D_KL approx exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    (Schulman non-negative formulation with [-20, 20] numerical clamping)
    """
    log_ratio = torch.clamp(ref_log_probs - log_probs, min=-20.0, max=20.0)
    # Non-negative estimator: exp(x) - x - 1 >= 0
    kl = torch.exp(log_ratio) - log_ratio - 1.0

    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


# ---------------------------------------------------------------------------
# ReST-GRPO Trainer Implementation
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    Coordinates ReST-GRPO self-training loop:
    - Calculates per-token log-probabilities under policy and reference models
    - Computes group-relative normalized advantages
    - Evaluates clipped surrogate objective + KL penalty
    - Performs backward pass and optimizer step
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[GRPOTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or GRPOTrainerConfig()

        self.device = self.config.device
        if self.device is None:
            if policy_model is not None:
                try:
                    self.device = str(next(policy_model.parameters()).device)
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"

        if optimizer is not None:
            self.optimizer = optimizer
        elif self.policy_model is not None:
            # Optimize parameters that require grad
            trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
            if trainable_params:
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            else:
                self.optimizer = None
        else:
            self.optimizer = None

        self._step_counter = 0

    def compute_log_probs_for_sequence(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of token sequence under the model.
        Returns: [batch_size, seq_len - 1] tensor of log-probs for each generated token.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather the log prob of the actual tokens
        selected_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [B, L - 1]

        return selected_log_probs

    def evaluate_group_loss(
        self,
        group_sample: PromptGroupSample,
    ) -> Tuple[torch.Tensor, GRPOLossMetrics]:
        """
        Computes the ReST-GRPO policy loss for a group of completions.
        Formula:
            L_GRPO = - (E[L_clip] - beta * D_KL)
        """
        # Ensure group statistics and normalized advantages are computed
        group_sample.compute_group_statistics(eps=self.config.eps)
        advantages = group_sample.advantages.to(self.device)

        surrogate_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        clip_fractions: List[float] = []

        for i, sample in enumerate(group_sample.samples):
            adv_i = advantages[i]

            if sample.input_ids is None or sample.completion_ids is None:
                # If raw token IDs were not pre-cached, tokenize full text
                if self.tokenizer is None:
                    continue
                encoded = self.tokenizer(sample.full_text, return_tensors="pt").to(self.device)
                full_ids = encoded["input_ids"]
                # Determine prompt prefix by stripping completion text if present
                if sample.completion_text and sample.completion_text in sample.full_text:
                    prefix_text = sample.full_text[:sample.full_text.rfind(sample.completion_text)]
                else:
                    prefix_text = group_sample.prompt
                prompt_ids = self.tokenizer(prefix_text, return_tensors="pt")["input_ids"]
                prompt_len = prompt_ids.shape[1]
            else:
                full_ids = torch.cat([sample.input_ids, sample.completion_ids]).unsqueeze(0).to(self.device)
                prompt_len = sample.input_ids.numel()

            seq_len = full_ids.shape[1]
            if seq_len <= prompt_len:
                continue

            # Token log probs under current policy
            policy_log_probs = self.compute_log_probs_for_sequence(self.policy_model, full_ids)

            # Completion token mask (1 for completion tokens, 0 for prompt)
            # Since shift cuts off the last token, completion tokens start at prompt_len - 1
            mask = torch.zeros_like(policy_log_probs)
            comp_start = max(0, prompt_len - 1)
            mask[:, comp_start:] = 1.0

            # Old / behavior policy log probs for importance sampling ratio pi_theta / pi_old
            if sample.token_log_probs is not None:
                old_log_probs = sample.token_log_probs.to(self.device)
            else:
                old_log_probs = policy_log_probs.detach().clone()

            # Reference policy log probs for KL divergence penalty D_KL(pi_theta || pi_ref)
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_log_probs = self.compute_log_probs_for_sequence(self.ref_model, full_ids)
                else:
                    ref_log_probs = old_log_probs.clone()

            # Surrogate loss for sample i (uses old_log_probs)
            surr_i, clip_frac_i = compute_surrogate_loss(
                log_probs=policy_log_probs,
                old_log_probs=old_log_probs,
                advantages=adv_i,
                clip_eps=self.config.clip_eps,
                mask=mask,
            )

            # KL penalty for sample i (uses ref_log_probs)
            kl_i = compute_kl_penalty(
                log_probs=policy_log_probs,
                ref_log_probs=ref_log_probs,
                mask=mask,
            )

            surrogate_terms.append(surr_i)
            kl_terms.append(kl_i)
            clip_fractions.append(float(clip_frac_i.item() if torch.is_tensor(clip_frac_i) else clip_frac_i))

        if not surrogate_terms:
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            metrics = GRPOLossMetrics(0.0, 0.0, 0.0, group_sample.mean_reward, group_sample.std_reward, 0.0, 0.0)
            return dummy_loss, metrics

        mean_surrogate = torch.stack(surrogate_terms).mean()
        mean_kl = torch.stack(kl_terms).mean()
        mean_clip_frac = sum(clip_fractions) / len(clip_fractions)

        # Minimize negative surrogate + KL divergence penalty
        total_loss = - (mean_surrogate - self.config.kl_coeff * mean_kl)

        metrics = GRPOLossMetrics(
            total_loss=float(total_loss.item()),
            surrogate_loss=float(mean_surrogate.item()),
            kl_divergence=float(mean_kl.item()),
            mean_reward=group_sample.mean_reward,
            std_reward=group_sample.std_reward,
            clip_fraction=mean_clip_frac,
            mean_advantage=float(advantages.mean().item()),
        )

        return total_loss, metrics

    def train_step(self, group_sample: PromptGroupSample) -> GRPOLossMetrics:
        """
        Executes a single ReST-GRPO training step on a prompt group sample.
        """
        if self.optimizer is None:
            raise RuntimeError("No optimizer initialized or no trainable parameters found.")

        self.policy_model.train()
        loss, metrics = self.evaluate_group_loss(group_sample)

        # Gradient accumulation scaling
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        self._step_counter += 1
        if self._step_counter % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return metrics
