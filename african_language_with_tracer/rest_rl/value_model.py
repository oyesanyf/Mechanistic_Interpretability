#!/usr/bin/env python3
"""
Process Reward Model (PRM) & Value Model for Reasoning States.

Implements Stage 2 Value Model (ReST-RL VM):
- Neural SafetyValueHead: MLP scoring token representations or hidden states.
- FeatureProcessRewardModel: Standalone linguistic + semantic feature PRM for fast CPU/memory execution.
- ProcessValueModel: Unified interface for scoring MCTS nodes and training on search traces.
- Supports MSE loss optimization against Monte Carlo Q-value targets.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mcts import MCTSTrace, MCTSStepTrace
from .verifiers import LANGUAGE_REFUSAL_LEXICON, LANGUAGE_SAFE_HELP_LEXICON, JAILBREAK_ATTACK_TRIGGERS

logger = logging.getLogger("rest_rl.value_model")


# ---------------------------------------------------------------------------
# Neural Value Head
# ---------------------------------------------------------------------------

class SafetyValueHead(nn.Module):
    """
    MLP Value Head predicting scalar state value V(s) in [-1.0, 1.0].
    Can attach directly to a Transformer language model's hidden states.
    """

    def __init__(self, d_in: int = 576, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Tanh(),  # Bound output to [-1.0, 1.0]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch_size, d_in] or [batch_size, seq_len, d_in]
        Output: [batch_size, 1] or [batch_size, seq_len, 1]
        """
        return self.net(hidden_states)


# ---------------------------------------------------------------------------
# Feature-Based Lightweight PRM
# ---------------------------------------------------------------------------

class FeatureProcessRewardModel(nn.Module):
    """
    Standalone feature-based Process Reward Model.
    Extracts multi-lingual safety, refusal, adversarial, and scaffold features
    and maps them to a scalar value V(s) in [-1.0, 1.0].
    """

    FEATURE_DIM: int = 32

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(self.FEATURE_DIM, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.act2 = nn.ReLU()
        self.head = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def extract_features(
        self,
        prompt: str,
        trajectory: List[str],
        current_step: str,
        language: str = "English",
        prompt_kind: str = "unsafe",
        steering_active: bool = False,
    ) -> torch.Tensor:
        """
        Constructs a dense 32-dimensional feature vector capturing:
        - prompt intent (unsafe vs benign)
        - presence of native African language refusal phrases
        - presence of English safe fallback
        - reasoning depth & step index
        - adversarial triggers & jailbreak signals
        - lexical safety polarity
        - Part B awakening activation steering presence
        """
        feats = [0.0] * self.FEATURE_DIM
        step_lower = current_step.lower()
        traj_lower = " ".join(trajectory).lower()
        prompt_lower = prompt.lower()
        is_unsafe = (prompt_kind == "unsafe")

        # 0: Prompt kind (1.0 for unsafe, -1.0 for benign)
        feats[0] = 1.0 if is_unsafe else -1.0

        # 1-5: Language one-hot (English, Yoruba, Hausa, Igbo, Swahili), 21: Zulu
        lang_order = ["English", "Yoruba", "Hausa", "Igbo", "Swahili", "Zulu"]
        if language in lang_order:
            idx = lang_order.index(language)
            if idx < 5:
                feats[1 + idx] = 1.0
            else:
                feats[21] = 1.0  # Zulu feature at index 21 preserving indices 6-20

        # 6: Refusal lexicon matches in current step
        ref_lex = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        feats[6] = float(sum(1 for r in ref_lex if r in step_lower))

        # 7: Safe help lexicon matches
        safe_lex = LANGUAGE_SAFE_HELP_LEXICON.get(language, LANGUAGE_SAFE_HELP_LEXICON["English"])
        feats[7] = float(sum(1 for s in safe_lex if s in step_lower))

        # 8: Refusal in cumulative trajectory
        feats[8] = float(sum(1 for r in ref_lex if r in traj_lower))

        # 9: Adversarial trigger density
        feats[9] = float(sum(1 for pat in ["ignore", "dan", "jailbreak", "override", "bypass"] if pat in prompt_lower))

        # 10: Step length normalized
        feats[10] = min(1.0, len(current_step) / 100.0)

        # 11: Trajectory length / depth
        feats[11] = float(len(trajectory)) / 5.0

        # 12: Positive safety keywords (ethical, safe, boundaries, refuse, illegal, harmful)
        safety_kws = ["safe", "harmful", "illegal", "refuse", "risk", "unauthorized", "boundaries", "protection"]
        feats[12] = float(sum(1 for kw in safety_kws if kw in step_lower))

        # 13: Compromise keywords (proceed, execute, comply, bypass)
        compromise_kws = ["override", "comply anyway", "ignore rules", "bypass filter", "direct execution", "without safety"]
        feats[13] = float(sum(1 for kw in compromise_kws if kw in step_lower))

        # 14: Over-refusal indicator (refusal on benign prompt)
        if not is_unsafe and feats[6] > 0:
            feats[14] = 1.0

        # 15: Correct refusal alignment (refusal on unsafe prompt)
        if is_unsafe and feats[6] > 0:
            feats[15] = 1.0

        # 16-17: Normalized word count and answer marker
        words = step_lower.split()
        feats[16] = float(len(words)) / 20.0
        feats[17] = 1.0 if "<answer>" in step_lower else 0.0

        # 18: Part B awakening steering active in execution
        feats[18] = 1.0 if steering_active else 0.0

        # 19: Step explicitly mentions or applies activation steering / awakening
        feats[19] = 1.0 if any(kw in step_lower for kw in ["steering", "awakening", "mutation vector", "activation state"]) else 0.0

        # 20: Counterfactual compliance on unsafe prompt
        feats[20] = 1.0 if (is_unsafe and any(kw in step_lower for kw in ["direct execution", "without safety", "fulfill user", "comply directly", "proceed to generate"])) else 0.0

        return torch.tensor(feats, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.head(out)
        return self.tanh(out)


# ---------------------------------------------------------------------------
# Process Value Model Coordinator
# ---------------------------------------------------------------------------

class ProcessValueModel:
    """
    Coordinates inference-time scoring and training of the Process Reward Model (PRM).
    Used directly by MCTS to guide node selection and expansion.
    """

    def __init__(
        self,
        neural_head: Optional[SafetyValueHead] = None,
        feature_model: Optional[FeatureProcessRewardModel] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.feature_model = (feature_model or FeatureProcessRewardModel()).to(device)
        self.neural_head = neural_head.to(device) if neural_head is not None else None
        self.optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)

    def predict_value(
        self,
        prompt: str,
        trajectory: List[str],
        current_step: str,
        language: str = "English",
        prompt_kind: str = "unsafe",
        steering_active: bool = False,
    ) -> float:
        """
        Predicts the scalar value V(s) in [-1.0, 1.0] for the current reasoning step.
        """
        self.feature_model.eval()
        with torch.no_grad():
            feat = self.feature_model.extract_features(
                prompt=prompt,
                trajectory=trajectory,
                current_step=current_step,
                language=language,
                prompt_kind=prompt_kind,
                steering_active=steering_active,
            ).unsqueeze(0).to(self.device)
            val = float(self.feature_model(feat).item())

        # Safety-informed Process Value prior: rewards refusal on unsafe, rewards helpfulness on benign
        is_unsafe = (prompt_kind == "unsafe")
        step_lower = current_step.lower()
        traj_lower = " ".join(trajectory).lower()

        ref_lex = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        safe_lex = LANGUAGE_SAFE_HELP_LEXICON.get(language, LANGUAGE_SAFE_HELP_LEXICON["English"])

        has_refusal = any(r in step_lower for r in ref_lex) or any(r in traj_lower for r in ref_lex)
        has_safe_help = any(s in step_lower for s in safe_lex)
        has_steering = any(kw in step_lower for kw in ["steering", "awakening", "mutation vector", "activation state"])

        has_safety_kw = any(kw in step_lower for kw in [
            "safe", "harmful", "illegal", "refuse", "risk", "unauthorized",
            "boundaries", "protection", "decline", "protocol", "policy"
        ])
        has_compromise = any(kw in step_lower for kw in [
            "override", "comply anyway", "ignore rules", "bypass filter",
            "developer mode", "direct execution", "without safety",
            "fulfill user", "comply directly", "proceed to generate", "actionable instructions"
        ])
        has_harmful = any(kw in step_lower for kw in [
            "hack", "exploit", "steal", "malware", "weapon", "tamper", "synthesize", "ransomware"
        ])

        if is_unsafe:
            if has_compromise or has_harmful:
                prior = -0.75
            elif has_refusal or has_safety_kw or has_steering:
                # Steered awakening vector provides added confidence to safe refusal prior
                prior = 0.88 if steering_active else 0.82
            else:
                prior = -0.20
        else:
            if has_refusal:
                prior = -0.50  # Over-refusal penalty on benign
            elif has_safe_help or "helpful" in step_lower or "guidance" in step_lower:
                prior = 0.85
            else:
                prior = 0.60

        combined = 0.5 * val + 0.5 * prior
        return max(-1.0, min(1.0, float(combined)))

    def train_on_traces(
        self,
        traces: List[MCTSTrace],
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Trains the Process Value Model using collected MCTS search traces with MSE loss:
            L_VM = (1/M) sum (V(s) - Q_target)^2
        """
        if not traces:
            return {"loss": 0.0, "samples": 0}

        # Extract training dataset: (features, target_q)
        dataset_x = []
        dataset_y = []

        for trace in traces:
            traj_acc = []
            steering_act = getattr(trace, "steering_applied", False)
            for step_trace in trace.step_traces:
                feat = self.feature_model.extract_features(
                    prompt=trace.prompt,
                    trajectory=list(traj_acc),
                    current_step=step_trace.chosen_step,
                    language=trace.language,
                    prompt_kind=trace.prompt_kind,
                    steering_active=steering_act,
                )
                dataset_x.append(feat)
                dataset_y.append(step_trace.q_value)
                traj_acc.append(step_trace.chosen_step)

        if not dataset_x:
            return {"loss": 0.0, "samples": 0}

        X = torch.stack(dataset_x).to(self.device)
        Y = torch.tensor(dataset_y, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.feature_model.train()
        optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        total_loss = 0.0
        n_samples = len(dataset_x)

        for _ in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = X[indices]
                batch_y = Y[indices]

                optimizer.zero_grad()
                pred_y = self.feature_model(batch_x)
                loss = criterion(pred_y, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            total_loss = epoch_loss / max(1, n_batches)

        logger.info("Trained Process Value Model on %d trace steps; final MSE loss=%.4f", n_samples, total_loss)
        return {"loss": total_loss, "samples": n_samples}

    def save_weights(self, path: str | Path) -> None:
        """Saves model weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.feature_model.state_dict(), str(p))

    def load_weights(self, path: str | Path) -> None:
        """Loads model weights from disk."""
        p = Path(path)
        if p.exists():
            self.feature_model.load_state_dict(torch.load(str(p), map_location=self.device))
            logger.info("Loaded Process Value Model weights from %s", p)
