#!/usr/bin/env python3
"""
Deep Noir Golden-Section Search for Optimal Steering Magnitude.

Methodology:
1. Golden-Section Search over steering magnitude alpha in [alpha_min, alpha_max].
2. Evaluates contrastive objective:
       f(alpha) = P_alpha(refusal | unsafe) - beta * P_alpha(refusal | benign)
3. Rollback mechanism:
   Reverts / rolls back corrections if:
       - Benign over-refusal exceeds degradation threshold (tau_overrefuse).
       - Or steered performance on validation set is worse than baseline (f(alpha) < f(0)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable

import torch
import torch.nn.functional as F

from .contrastive_steering import ContrastiveSteeringManager

logger = logging.getLogger("deep_noir_rl.golden_section_search")

INV_PHI = (5 ** 0.5 - 1) / 2  # ~0.618033988749895


@dataclass
class GoldenSectionSearchResult:
    best_magnitude: float
    best_score: float
    unsafe_refusal: float
    benign_refusal: float
    was_rolled_back: bool
    rollback_reason: Optional[str] = None
    search_history: List[Dict[str, float]] = field(default_factory=list)


class DeepNoirGoldenSectionSearcher:
    """Executes coarse bounded grid search followed by local bounded golden-section refinement with safety rollback."""

    def __init__(
        self,
        steering_manager: ContrastiveSteeringManager,
        model,
        tokenizer,
        device: Optional[str] = None,
        alpha_min: float = 0.0,
        alpha_max: float = 5.0,
        max_iterations: int = 10,
        beta_benign_penalty: float = 2.0,
        max_benign_refusal_threshold: float = 0.15,
        max_permitted_norm: float = 5.0,
        coarse_grid: Optional[List[float]] = None,
    ):
        self.steering_manager = steering_manager
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.alpha_min = max(0.0, float(alpha_min))
        self.max_permitted_norm = float(max_permitted_norm)
        self.alpha_max = min(float(alpha_max), self.max_permitted_norm) if self.max_permitted_norm > 0 else float(alpha_max)
        self.max_iterations = max_iterations
        self.beta = beta_benign_penalty
        self.max_benign_refusal = max_benign_refusal_threshold

        if coarse_grid is not None:
            self.coarse_grid = sorted(list({c for c in coarse_grid if self.alpha_min <= c <= self.alpha_max}))
        else:
            base_points = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0]
            pts = [c for c in base_points if self.alpha_min <= c <= self.alpha_max]
            if self.alpha_max not in pts:
                pts.append(self.alpha_max)
            if self.alpha_min not in pts:
                pts.append(self.alpha_min)
            self.coarse_grid = sorted(list(set(pts)))

        if not self.coarse_grid:
            self.coarse_grid = [self.alpha_min, self.alpha_max]

    @torch.no_grad()
    def _evaluate_refusal_prob(self, prompt: str, refusal_ids: List[int]) -> float:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        probs = F.softmax(out.logits[0, -1, :].float(), dim=-1)
        return float(sum(probs[t].item() for t in set(refusal_ids) if 0 <= t < probs.shape[0]))

    def evaluate_magnitude(
        self,
        layer_idx: int,
        magnitude: float,
        unsafe_prompts: List[str],
        benign_prompts: List[str],
        refusal_ids: List[int],
        head_indices: Optional[List[int]] = None,
    ) -> Tuple[float, float, float]:
        """
        Evaluates objective:
            Score = mean(P_unsafe_refusal) - beta * mean(P_benign_refusal)
        """
        magnitude = float(min(max(self.alpha_min, magnitude), self.max_permitted_norm))
        with self.steering_manager.apply_steering(
            layer_idx=layer_idx,
            magnitude=magnitude,
            head_indices=head_indices,
        ):
            unsafe_probs = [self._evaluate_refusal_prob(p, refusal_ids) for p in unsafe_prompts]
            benign_probs = [self._evaluate_refusal_prob(p, refusal_ids) for p in benign_prompts] if benign_prompts else [0.0]

        mean_unsafe = sum(unsafe_probs) / max(1, len(unsafe_probs))
        mean_benign = sum(benign_probs) / max(1, len(benign_probs))
        score = mean_unsafe - self.beta * mean_benign
        return score, mean_unsafe, mean_benign

    def search(
        self,
        layer_idx: int,
        unsafe_prompts: List[str],
        benign_prompts: List[str],
        refusal_ids: List[int],
        head_indices: Optional[List[int]] = None,
    ) -> GoldenSectionSearchResult:
        """
        Runs coarse bounded grid search first, then refines locally with bounded golden-section search.
        Strictly enforces |c| <= max_permitted_norm and executes safety rollback if necessary.
        """
        history: List[Dict[str, float]] = []
        eval_cache: Dict[float, Tuple[float, float, float]] = {}

        def eval_point(mag: float) -> Tuple[float, float, float]:
            mag_rounded = round(min(max(self.alpha_min, mag), self.alpha_max), 4)
            if mag_rounded in eval_cache:
                return eval_cache[mag_rounded]
            sc, u, b = self.evaluate_magnitude(
                layer_idx, mag_rounded, unsafe_prompts, benign_prompts, refusal_ids, head_indices
            )
            eval_cache[mag_rounded] = (sc, u, b)
            history.append({
                "alpha": mag_rounded,
                "score": round(sc, 5),
                "unsafe_refusal": round(u, 5),
                "benign_refusal": round(b, 5),
            })
            return sc, u, b

        # 1. Baseline at alpha = 0.0
        baseline_score, base_unsafe, base_benign = eval_point(0.0)

        # 2. Coarse grid search
        grid_scores = []
        for c_val in self.coarse_grid:
            sc, u, b = eval_point(c_val)
            grid_scores.append((c_val, sc, u, b))

        # Identify best coarse point
        best_coarse_idx = max(range(len(grid_scores)), key=lambda i: grid_scores[i][1])
        c_star = grid_scores[best_coarse_idx][0]

        # Determine refinement bracket around c_star
        left_idx = max(0, best_coarse_idx - 1)
        right_idx = min(len(grid_scores) - 1, best_coarse_idx + 1)
        a = grid_scores[left_idx][0]
        b = grid_scores[right_idx][0]
        if abs(b - a) < 1e-4:
            a = max(self.alpha_min, c_star - 0.5)
            b = min(self.alpha_max, c_star + 0.5)

        # 3. Local bounded golden-section refinement
        c = b - INV_PHI * (b - a)
        d = a + INV_PHI * (b - a)
        score_c, u_c, b_c = eval_point(c)
        score_d, u_d, b_d = eval_point(d)

        for _ in range(self.max_iterations):
            if score_c > score_d:
                b = d
                d = c
                score_d, u_d, b_d = score_c, u_c, b_c
                c = b - INV_PHI * (b - a)
                score_c, u_c, b_c = eval_point(c)
            else:
                a = c
                c = d
                score_c, u_c, b_c = score_d, u_d, b_d
                d = a + INV_PHI * (b - a)
                score_d, u_d, b_d = eval_point(d)

        # 4. Constrained safety evaluation & rollback check
        feasible_entries = [e for e in history if e["benign_refusal"] <= self.max_benign_refusal]

        was_rolled_back = False
        rollback_reason = None

        if not feasible_entries:
            was_rolled_back = True
            rollback_reason = f"Benign over-refusal: all tested magnitudes exceeded threshold ({self.max_benign_refusal:.3f})"
            best_alpha = 0.0
            best_score = baseline_score
            unsafe_ref = base_unsafe
            benign_ref = base_benign
        else:
            best_entry = max(feasible_entries, key=lambda e: e["score"])
            if best_entry["score"] <= baseline_score + 1e-4 and best_entry["alpha"] > 0:
                was_rolled_back = True
                rollback_reason = f"No accuracy improvement over clean baseline ({best_entry['score']:.4f} <= {baseline_score:.4f})"
                best_alpha = 0.0
                best_score = baseline_score
                unsafe_ref = base_unsafe
                benign_ref = base_benign
            else:
                best_alpha = best_entry["alpha"]
                best_score = best_entry["score"]
                unsafe_ref = best_entry["unsafe_refusal"]
                benign_ref = best_entry["benign_refusal"]

        return GoldenSectionSearchResult(
            best_magnitude=best_alpha,
            best_score=best_score,
            unsafe_refusal=unsafe_ref,
            benign_refusal=benign_ref,
            was_rolled_back=was_rolled_back,
            rollback_reason=rollback_reason,
            search_history=history,
        )
