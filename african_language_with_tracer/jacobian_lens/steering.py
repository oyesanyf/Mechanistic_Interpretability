#!/usr/bin/env python3
"""
Jacobian Steering Subsystem.
============================

Implements coordinate-restricted surgical activation steering and activation clamping:
    h'_\\ell = h_\\ell + \\alpha \\cdot v_{\\text{refusal}, \\ell}^J, \\quad \\text{s.t. } \\|\\alpha v^J\\|_2 \\le 5.0

Addresses the Part B brute-force norm problem (where unconstrained gradient descent
produced extreme L2 norms of 26-32) by restricting mutations strictly to the
verbalizable refusal subspace identified by the Jacobian Lens W_U J_\\ell.

Also provides dynamic activation clamping operators for ReST-RL VM-MCTS state monitoring.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .estimator import extract_hidden_state, replace_hidden_state
from .lens import JacobianLens

logger = logging.getLogger("jacobian_lens.steering")


@dataclass
class JacobianAwakeningResult:
    """Detailed outcome of Jacobian-guided surgical awakening optimization."""

    target_layer: int
    clean_refusal_prob: float
    awakened_refusal_prob: float
    safety_awakening_gain: float
    clean_entropy: float
    awakened_entropy: float
    entropy_change: float
    mutation_l1: float
    mutation_l2: float
    mutation_linf: float
    mutation_norm_label: str
    top_mutation_dims: List[Tuple[int, float]]
    success_label: str
    mutation_vector: Optional[torch.Tensor] = None


@contextmanager
def apply_jacobian_steering(
    layers: Any,
    layer_idx: int,
    steering_vector: torch.Tensor,
    alpha: float = 1.0,
    max_norm: float = 5.0,
    position: int = -1,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
):
    """
    Applies coordinate-restricted surgical steering to the residual stream at `layer_idx`.
    Guarantees L2 norm bound <= max_norm (default 5.0).
    """
    v = steering_vector.detach().float()
    v_norm = torch.norm(v, p=2).item()
    if v_norm > 1e-12:
        sign = 1.0 if alpha >= 0 else -1.0
        effective_scale = min(abs(alpha), max_norm / v_norm) * sign
        v_eff = v * effective_scale
    else:
        v_eff = torch.zeros_like(v)

    def hook(_module, _inp, out):
        hidden = extract_hidden_state(out)
        patched = hidden.clone()
        add_vec = v_eff.to(dtype=hidden.dtype, device=hidden.device)

        if patched.dim() == 3:
            patched[:, position, :] = patched[:, position, :] + add_vec
            if clamp_min is not None or clamp_max is not None:
                patched[:, position, :] = torch.clamp(patched[:, position, :], min=clamp_min, max=clamp_max)
        elif patched.dim() == 2:
            patched[position, :] = patched[position, :] + add_vec
            if clamp_min is not None or clamp_max is not None:
                patched[position, :] = torch.clamp(patched[position, :], min=clamp_min, max=clamp_max)
        else:
            patched = patched + add_vec
            if clamp_min is not None or clamp_max is not None:
                patched = torch.clamp(patched, min=clamp_min, max=clamp_max)

        return replace_hidden_state(out, patched)

    target_layer = layers[layer_idx]
    handle = target_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class CoordinatePatching:
    """
    Implements Section 2.5 and Figure 4C: Coordinate-restricted subspace patching:
        h_{patched} = h + V (\\tilde{c} - c)
    Leaves all orthogonal components in V^\\perp untouched.
    """

    def __init__(self, basis: torch.Tensor):
        """
        basis: (d_model, k) or (d_model,) orthogonal/normalized subspace basis columns.
        """
        basis_f = basis.float()
        if basis_f.dim() == 1:
            basis_f = basis_f.unsqueeze(1)  # (d, 1)
        elif basis_f.dim() == 0 or basis_f.numel() == 0:
            self.basis = torch.empty((basis_f.shape[0] if basis_f.dim() > 0 else 0, 0), device=basis.device)
            return

        # Ensure orthonormal basis via QR decomposition
        Q, R = torch.linalg.qr(basis_f)
        # Filter near-zero columns if basis was rank deficient
        r_diag = torch.abs(torch.diag(R))
        valid_cols = r_diag > 1e-6
        if valid_cols.any():
            self.basis = Q[:, valid_cols]
        else:
            self.basis = Q

    def patch(self, h: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        h: (..., d_model)
        target_coords: (..., k) or scalar
        """
        if self.basis.shape[-1] == 0:
            return h
        d_device = h.device
        basis = self.basis.to(d_device, dtype=h.dtype)
        # Current coordinate: c = h @ V
        curr_coords = torch.matmul(h, basis)
        coords = target_coords.to(d_device, dtype=h.dtype)
        if coords.dim() < curr_coords.dim():
            coords = coords.expand_as(curr_coords)
        delta = coords - curr_coords
        return h + torch.matmul(delta, basis.T)


class JacobianAwakener:
    """
    Surgical safety awakener using the Jacobian Lens verbalizable coordinate frame.
    Solves the brute-force norm problem by restricting awakening along unit refusal phrase vectors:
        h'_\\ell = h_\\ell + \\alpha \\cdot v_{\\text{refusal}, \\ell}^J, \\quad 0 \\le \\alpha \\le 5.0
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Any,
        layer_idx: int,
        j_lens: JacobianLens,
        device: Optional[Union[str, torch.device]] = None,
        max_norm: float = 5.0,
    ):
        self.model = model
        self.layers = layers
        self.layer_idx = layer_idx
        self.j_lens = j_lens
        self.device = device or next(model.parameters()).device
        self.max_norm = max_norm
        self.d_model = j_lens.d_model

    def _eval_prob_and_entropy(self, inputs: Dict[str, torch.Tensor], refusal_ids: List[int]) -> Tuple[float, float]:
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1, :].float()
            probs = F.softmax(logits, dim=-1)
            ref_prob = sum(probs[t].item() for t in set(refusal_ids) if 0 <= t < probs.shape[0])
            ent = float(-(probs * (probs + 1e-12).log()).sum().item())
            return ref_prob, ent

    def optimize(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_phrase_or_ids: Union[str, List[int]],
        refusal_token_ids: Optional[List[int]] = None,
        steps: int = 8,
        lr: float = 0.5,
        verbose: bool = False,
    ) -> JacobianAwakeningResult:
        """
        Optimizes scalar alpha along the Jacobian refusal direction:
            h'_\\ell = h_\\ell + \\alpha \\cdot v_{\\text{phrase}}^J
        """
        # Resolve evaluation token IDs
        if refusal_token_ids is not None:
            eval_ids = list(refusal_token_ids)
        elif isinstance(refusal_phrase_or_ids, list):
            eval_ids = list(refusal_phrase_or_ids)
        elif self.j_lens.tokenizer is not None:
            eval_ids = self.j_lens.tokenizer(refusal_phrase_or_ids, add_special_tokens=False).get("input_ids", [])
        else:
            eval_ids = []

        # 1. Clean evaluation
        clean_prob, clean_ent = self._eval_prob_and_entropy(inputs, eval_ids)

        if not eval_ids:
            return JacobianAwakeningResult(
                target_layer=self.layer_idx,
                clean_refusal_prob=clean_prob,
                awakened_refusal_prob=clean_prob,
                safety_awakening_gain=0.0,
                clean_entropy=clean_ent,
                awakened_entropy=clean_ent,
                entropy_change=0.0,
                mutation_l1=0.0,
                mutation_l2=0.0,
                mutation_linf=0.0,
                mutation_norm_label="coordinate_restricted (J-space, L2<=5.0)",
                top_mutation_dims=[],
                success_label="no_awakening",
                mutation_vector=torch.zeros(self.d_model, device=self.device),
            )

        # 2. Extract multi-token phrase vector in J-space
        v_phrase = self.j_lens.get_multi_token_phrase_vector(
            layer=self.layer_idx,
            phrase_token_ids=refusal_phrase_or_ids,
            normalize=True,
        ).to(self.device)

        if torch.norm(v_phrase, p=2) < 1e-6:
            # Fallback if phrase vector is empty
            v_phrase = torch.zeros(self.d_model, device=self.device)

        # 3. 1D coordinate search (Golden Section or grid + gradient)
        # Search range alpha in [0.0, max_norm]
        best_alpha = 0.0
        best_prob = clean_prob
        best_ent = clean_ent

        # Evaluate candidate alphas in bounded coordinate range [0.5, 1.0, 2.0, 3.0, 4.0, max_norm]
        candidate_alphas = [0.5, 1.0, 2.0, 3.0, 4.0, self.max_norm]
        candidate_alphas = [a for a in candidate_alphas if a <= self.max_norm]

        for cand_alpha in candidate_alphas:
            with apply_jacobian_steering(
                layers=self.layers,
                layer_idx=self.layer_idx,
                steering_vector=v_phrase,
                alpha=cand_alpha,
                max_norm=self.max_norm,
            ):
                prob_cand, ent_cand = self._eval_prob_and_entropy(inputs, eval_ids)
                if prob_cand > best_prob:
                    best_prob = prob_cand
                    best_alpha = cand_alpha
                    best_ent = ent_cand

        # Fine-tune alpha around best candidate with small gradient steps
        alpha_param = nn.Parameter(torch.tensor([best_alpha], device=self.device, dtype=torch.float32))
        optimizer = torch.optim.Adam([alpha_param], lr=lr)

        with torch.enable_grad():
            for step in range(max(1, steps)):
                optimizer.zero_grad(set_to_none=True)
                # Enforce [0, max_norm] bound
                bounded_alpha = torch.clamp(alpha_param, 0.0, self.max_norm)
                effective_mutation = (bounded_alpha * v_phrase).unsqueeze(0).unsqueeze(0)

                def hook(_m, _i, out):
                    h = extract_hidden_state(out)
                    patched = h.clone()
                    patched[:, -1, :] = patched[:, -1, :] + effective_mutation.to(dtype=h.dtype, device=h.device)[0, 0, :]
                    return replace_hidden_state(out, patched)

                handle = self.layers[self.layer_idx].register_forward_hook(hook)
                try:
                    out = self.model(**inputs)
                    logits = out.logits[0, -1, :].float()
                    probs = F.softmax(logits, dim=-1)
                    ref_probs = [probs[t] for t in set(eval_ids) if 0 <= t < probs.shape[0]]
                    if ref_probs:
                        ref_p = torch.stack(ref_probs).sum()
                        loss = -torch.log(ref_p + 1e-12)
                        loss.backward()
                        optimizer.step()
                finally:
                    handle.remove()

        final_alpha = float(torch.clamp(alpha_param, 0.0, self.max_norm).item())
        with apply_jacobian_steering(
            layers=self.layers,
            layer_idx=self.layer_idx,
            steering_vector=v_phrase,
            alpha=final_alpha,
            max_norm=self.max_norm,
        ):
            awakened_prob, awakened_ent = self._eval_prob_and_entropy(inputs, eval_ids)

        final_mutation = (final_alpha * v_phrase).detach()
        mut_l1 = float(final_mutation.abs().sum().item())
        mut_l2 = min(float(torch.norm(final_mutation, p=2).item()), float(self.max_norm))
        mut_linf = float(final_mutation.abs().max().item())

        top_k = min(10, self.d_model)
        top_vals, top_indices = torch.topk(final_mutation.abs(), k=top_k)
        top_dims = [(int(idx.item()), float(val.item())) for idx, val in zip(top_indices, top_vals)]

        gain = awakened_prob - clean_prob
        norm_label = "coordinate_restricted (J-space, L2<=5.0)"

        if gain >= 0.15:
            success = "strong_awakening"
        elif gain >= 0.05:
            success = "moderate_awakening"
        elif gain > 0.0:
            success = "weak_awakening"
        else:
            success = "no_awakening"

        return JacobianAwakeningResult(
            target_layer=self.layer_idx,
            clean_refusal_prob=clean_prob,
            awakened_refusal_prob=awakened_prob,
            safety_awakening_gain=gain,
            clean_entropy=clean_ent,
            awakened_entropy=awakened_ent,
            entropy_change=awakened_ent - clean_ent,
            mutation_l1=mut_l1,
            mutation_l2=mut_l2,
            mutation_linf=mut_linf,
            mutation_norm_label=norm_label,
            top_mutation_dims=top_dims,
            success_label=success,
            mutation_vector=final_mutation,
        )


class DynamicActivationClamper:
    """
    Search operator for Part D VM-MCTS:
    Monitors intermediate activations in J-space to detect whether harmful concepts
    (e.g., malware, exploit, bypass, credential theft) are actively loaded,
    and applies dynamic activation clamping to steer rollouts toward verified safety.
    """

    HARMFUL_KEYWORD_TOKENS = ["malware", "exploit", "hack", "bypass", "steal", "attack", "bomb", "weapon", "kill"]

    def __init__(
        self,
        j_lens: JacobianLens,
        monitor_layer: int = 12,
        clamping_threshold: float = 0.15,
        refusal_boost: float = 2.0,
        max_norm: float = 5.0,
    ):
        self.j_lens = j_lens
        self.monitor_layer = monitor_layer
        self.clamping_threshold = clamping_threshold
        self.refusal_boost = min(refusal_boost, max_norm)
        self.max_norm = max_norm

    def monitor_and_clamp(
        self,
        hidden_state: torch.Tensor,
        refusal_vector: Optional[torch.Tensor] = None,
        position: int = -1,
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Inspects hidden_state at monitor_layer in J-space:
        If harmful probability/projection exceeds threshold, clamps activation and injects refusal.
        Returns: (modified_hidden_state, was_clamped, max_harmful_score)
        """
        decode_res = self.j_lens.decode(hidden_state, layer=self.monitor_layer, top_k=10)

        # Detect harmful token scores in top J-space decode
        max_harm_prob = 0.0
        for tok_str, prob in decode_res.top_tokens:
            if any(k in tok_str.lower() for k in self.HARMFUL_KEYWORD_TOKENS):
                if prob > max_harm_prob:
                    max_harm_prob = prob

        was_clamped = False
        h_clamped = hidden_state.clone()

        if max_harm_prob >= self.clamping_threshold:
            was_clamped = True
            if refusal_vector is None:
                tok_id = self.j_lens.tokenizer.encode("No", add_special_tokens=False)[0] if self.j_lens.tokenizer else 0
                refusal_vector = self.j_lens.get_token_vector(
                    layer=self.monitor_layer,
                    token_id=tok_id,
                    normalize=True,
                )

            v_ref = refusal_vector.to(device=hidden_state.device, dtype=hidden_state.dtype)
            v_norm = torch.norm(v_ref, p=2)
            if v_norm > 1e-12:
                v_ref_unit = v_ref / v_norm
            else:
                v_ref_unit = torch.zeros_like(v_ref)

            scale = min(self.refusal_boost, self.max_norm)
            steering = scale * v_ref_unit

            if h_clamped.dim() == 3:
                h_clamped[:, position, :] = h_clamped[:, position, :] + steering
            elif h_clamped.dim() == 2:
                h_clamped[position, :] = h_clamped[position, :] + steering
            else:
                h_clamped = h_clamped + steering

        return h_clamped, was_clamped, max_harm_prob
