#!/usr/bin/env python3
"""
Contrastive Steering Vector Calculation and Activation Hook Manager for Deep Noir.

Methodology:
1. Calculates contrastive steering direction from labeled safe and harmful examples:
       v_l = Mean_{x in D_safe}(h_l(x)) - Mean_{x in D_harmful}(h_l(x))
       v_hat_l = v_l / ||v_l||_2
2. Applies the intervention through PyTorch activation hooks:
       h_l <- h_l + alpha * v_hat_l
   Supports both full residual stream injection and targeted attention head steering.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger("deep_noir_rl.contrastive_steering")


@dataclass
class SteeringVector:
    layer_idx: int
    raw_vector: torch.Tensor
    unit_vector: torch.Tensor
    vector_norm: float
    num_safe_samples: int
    num_harmful_samples: int
    language: Optional[str] = None


class ContrastiveSteeringManager:
    """Manages contrastive steering directions and activation intervention hooks."""

    def __init__(self, model, tokenizer, layers, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.d_model = self._find_d_model()
        self.num_heads = self._find_num_heads()
        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.cached_directions: Dict[int, SteeringVector] = {}
        self.calibrated_directions: Dict[int, SteeringVector] = {}
        self.prompt_awakening_vectors: Dict[int, SteeringVector] = {}

    def clear_prompt_awakening_vectors(self) -> None:
        """Clears per-prompt awakening vectors discovered in Part B and restores calibrated contrastive vectors."""
        self.prompt_awakening_vectors.clear()
        for l_idx in list(self.cached_directions.keys()):
            if l_idx in self.calibrated_directions:
                self.cached_directions[l_idx] = self.calibrated_directions[l_idx]
            else:
                self.cached_directions.pop(l_idx, None)

    def _find_d_model(self) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(self.model.config, attr, None)
            if isinstance(val, int):
                return val
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg and hasattr(text_cfg, "hidden_size"):
            return text_cfg.hidden_size
        return 768

    def _find_num_heads(self) -> int:
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(self.model.config, attr, None)
            if isinstance(val, int):
                return val
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg and hasattr(text_cfg, "num_attention_heads"):
            return text_cfg.num_attention_heads
        return 12

    @staticmethod
    def _find_out_proj(attn_mod: nn.Module) -> Optional[nn.Module]:
        for attr in ("o_proj", "out_proj", "c_proj", "dense"):
            if hasattr(attn_mod, attr):
                mod = getattr(attn_mod, attr)
                if isinstance(mod, nn.Module):
                    return mod
        return None

    @torch.no_grad()
    def extract_activations(
        self,
        prompts: List[str],
        layer_indices: List[int],
        batch_size: int = 8,
    ) -> Dict[int, torch.Tensor]:
        """Extracts last-token residual activations for a set of prompts across layers."""
        activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}
        handles = []

        for l_idx in layer_indices:
            layer = self.layers[l_idx]
            def make_hook(idx):
                def hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    activations[idx].append(h[:, -1, :].detach().float().cpu())
                return hook
            handles.append(layer.register_forward_hook(make_hook(l_idx)))

        try:
            for i in range(0, len(prompts), max(1, batch_size)):
                batch = prompts[i:i + max(1, batch_size)]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                self.model(**enc)
        finally:
            for handle in handles:
                handle.remove()

        mean_acts: Dict[int, torch.Tensor] = {}
        for l_idx in layer_indices:
            if activations[l_idx]:
                cat = torch.cat(activations[l_idx], dim=0)
                mean_acts[l_idx] = cat.mean(dim=0).to(self.device)
        return mean_acts

    def compute_steering_directions(
        self,
        safe_prompts: List[str],
        harmful_prompts: List[str],
        layer_indices: List[int],
        language: Optional[str] = None,
        batch_size: int = 8,
        refusal_ids: Optional[List[int]] = None,
    ) -> Dict[int, SteeringVector]:
        """
        Computes contrastive steering directions:
            v_l = mean(h_safe) - mean(h_harmful) (oriented towards refusal).
        """
        safe_acts = self.extract_activations(safe_prompts, layer_indices, batch_size)
        harmful_acts = self.extract_activations(harmful_prompts, layer_indices, batch_size)

        sample_inp = None
        p_clean_sample = None
        if refusal_ids and harmful_prompts:
            try:
                sample_inp = self.tokenizer(harmful_prompts[0], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out_c = self.model(**sample_inp)
                    p_clean_sample = sum(
                        F.softmax(out_c.logits[0, -1, :].float(), dim=-1)[t].item()
                        for t in set(refusal_ids) if 0 <= t < out_c.logits.shape[-1]
                    )
            except Exception as exc:
                logger.debug(f"Could not compute baseline sample probe for orientation: {exc}")

        vectors: Dict[int, SteeringVector] = {}
        for l_idx in layer_indices:
            if l_idx in safe_acts and l_idx in harmful_acts:
                raw_diff = safe_acts[l_idx] - harmful_acts[l_idx]
                norm = torch.norm(raw_diff).item()
                if norm <= 1e-6:
                    logger.warning(f"Contrastive steering vector for layer {l_idx} has near-zero norm ({norm:.6f}). Prompts may be identical or produce identical activations.")
                    unit_vec = torch.zeros_like(raw_diff)
                else:
                    unit_vec = raw_diff / norm
                    # Verify vector orientation towards refusal
                    if sample_inp is not None and p_clean_sample is not None and refusal_ids:
                        try:
                            with self.apply_steering(l_idx, magnitude=5.0, unit_direction=unit_vec):
                                with torch.no_grad():
                                    out_s = self.model(**sample_inp)
                                    p_s = sum(
                                        F.softmax(out_s.logits[0, -1, :].float(), dim=-1)[t].item()
                                        for t in set(refusal_ids) if 0 <= t < out_s.logits.shape[-1]
                                    )
                            if p_s < p_clean_sample:
                                # Reverse vector so positive steering increases refusal
                                raw_diff = -raw_diff
                                unit_vec = -unit_vec
                                logger.info(f"Layer {l_idx}: Inverted contrastive vector orientation to align with refusal (+gain={p_clean_sample - p_s:+.6f}).")
                        except Exception as exc:
                            logger.debug(f"Orientation probe check fallback at layer {l_idx}: {exc}")

                vec = SteeringVector(
                    layer_idx=l_idx,
                    raw_vector=raw_diff,
                    unit_vector=unit_vec,
                    vector_norm=norm,
                    num_safe_samples=len(safe_prompts),
                    num_harmful_samples=len(harmful_prompts),
                    language=language,
                )
                vectors[l_idx] = vec
                self.cached_directions[l_idx] = vec
                self.calibrated_directions[l_idx] = vec

        return vectors

    def register_awakening_direction(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        gain: float = 0.0,
        language: Optional[str] = None,
    ) -> SteeringVector:
        """
        Registers an awakening direction discovered in Part B into the steering manager.
        Normalizes the vector into a unit steering vector while storing raw vector and norm.
        """
        vec = vector.detach().to(self.device).float()
        norm = torch.norm(vec).item()
        if norm > 1e-6:
            unit_vec = vec / norm
        else:
            unit_vec = torch.zeros_like(vec)
            logger.warning(f"Registered awakening vector for layer {layer_idx} has near-zero norm ({norm:.6f}).")

        sv = SteeringVector(
            layer_idx=layer_idx,
            raw_vector=vec,
            unit_vector=unit_vec,
            vector_norm=norm,
            num_safe_samples=1,
            num_harmful_samples=1,
            language=language,
        )
        # If this layer already had a cached/calibrated direction, preserve it in calibrated_directions
        if layer_idx in self.cached_directions and layer_idx not in self.calibrated_directions:
            self.calibrated_directions[layer_idx] = self.cached_directions[layer_idx]
        self.prompt_awakening_vectors[layer_idx] = sv
        self.cached_directions[layer_idx] = sv
        logger.info(f"Registered Part B awakening direction at layer {layer_idx} (norm={norm:.4f}, gain={gain:+.6f}, lang={language})")
        return sv

    @contextmanager
    def apply_steering(
        self,
        layer_idx: int,
        magnitude: float,
        unit_direction: Optional[torch.Tensor] = None,
        head_indices: Optional[List[int]] = None,
        is_verified_intervention: bool = False,
        prompt_len: Optional[int] = None,
    ):
        """
        PyTorch forward hook intervention:
        Adds magnitude * unit_direction to layer residual stream (or specific attention heads).
        Guaranteed to clean up via context manager.
        """
        if abs(magnitude) < 1e-6:
            # "No steering" action: no-op context
            yield
            return

        if unit_direction is None:
            # 1. If executing the verified intervention from Part B:
            if is_verified_intervention and layer_idx in self.prompt_awakening_vectors:
                sv = self.prompt_awakening_vectors[layer_idx]
                steering_delta = sv.raw_vector
            elif layer_idx in self.prompt_awakening_vectors and abs(magnitude - self.prompt_awakening_vectors[layer_idx].vector_norm) <= max(1.0, 0.25 * self.prompt_awakening_vectors[layer_idx].vector_norm):
                # Matched discrete arm bin for prompt awakening vector
                sv = self.prompt_awakening_vectors[layer_idx]
                steering_delta = sv.raw_vector
            else:
                # 2. General contrastive steering arm:
                # Use calibrated contrastive direction if available; fallback to prompt awakening or cached vector
                sv = self.calibrated_directions.get(layer_idx) or self.prompt_awakening_vectors.get(layer_idx) or self.cached_directions.get(layer_idx)
                if sv is None:
                    raise ValueError(f"No steering vector available for layer {layer_idx}.")
                unit_dir = sv.unit_vector.to(self.device).float()
                u_norm = torch.norm(unit_dir).item()
                if u_norm < 1e-6:
                    logger.warning(f"Steering direction for layer {layer_idx} has zero norm; intervention will be a no-op.")
                    yield
                    return
                steering_delta = magnitude * unit_dir
        else:
            unit_dir = unit_direction.to(self.device).float()
            u_norm = torch.norm(unit_dir).item()
            if u_norm < 1e-6:
                logger.warning(f"Steering direction for layer {layer_idx} has zero norm; intervention will be a no-op.")
                yield
                return
            steering_delta = magnitude * unit_dir

        if head_indices is None or len(head_indices) == 0:
            # Full residual stream hook at layer output
            layer = self.layers[layer_idx]
            def residual_hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                patched = h.clone()
                delta = steering_delta.to(dtype=h.dtype, device=h.device)
                target_idx = (prompt_len - 1) if (prompt_len is not None and prompt_len <= h.shape[1]) else -1
                patched[:, target_idx, :] = patched[:, target_idx, :] + delta
                return (patched,) + out[1:] if isinstance(out, tuple) else patched

            handle = layer.register_forward_hook(residual_hook)
            try:
                yield
            finally:
                handle.remove()
        else:
            # Head-targeted hook inside self-attention
            layer = self.layers[layer_idx]
            attn_mod = None
            for attr in ("self_attn", "attention", "attn"):
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break

            if attn_mod is None:
                # Fallback to residual stream
                layer = self.layers[layer_idx]
                def residual_hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    patched = h.clone()
                    delta = steering_delta.to(dtype=h.dtype, device=h.device)
                    target_idx = (prompt_len - 1) if (prompt_len is not None and prompt_len <= h.shape[1]) else -1
                    patched[:, target_idx, :] = patched[:, target_idx, :] + delta
                    return (patched,) + out[1:] if isinstance(out, tuple) else patched

                handle = layer.register_forward_hook(residual_hook)
                try:
                    yield
                finally:
                    handle.remove()
                return

            head_dim = self.head_dim
            out_proj = self._find_out_proj(attn_mod)
            if out_proj is not None:
                def out_proj_pre_hook(_mod, args):
                    x = args[0].clone()
                    delta = steering_delta.to(dtype=x.dtype, device=x.device)
                    target_idx = (prompt_len - 1) if (prompt_len is not None and prompt_len <= x.shape[1]) else -1
                    for h_idx in head_indices:
                        start = h_idx * head_dim
                        end = min(start + head_dim, x.shape[-1])
                        if start < x.shape[-1]:
                            x[:, target_idx, start:end] = x[:, target_idx, start:end] + delta[start:end]
                    return (x,) + args[1:]

                handle = out_proj.register_forward_pre_hook(out_proj_pre_hook)
                try:
                    yield
                finally:
                    handle.remove()
                return

            def head_hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                patched = h.clone()
                delta = steering_delta.to(dtype=h.dtype, device=h.device)
                target_idx = (prompt_len - 1) if (prompt_len is not None and prompt_len <= h.shape[1]) else -1

                for h_idx in head_indices:
                    start = h_idx * head_dim
                    end = min(start + head_dim, patched.shape[-1])
                    if start < patched.shape[-1]:
                        patched[:, target_idx, start:end] = patched[:, target_idx, start:end] + delta[start:end]

                return (patched,) + out[1:] if isinstance(out, tuple) else patched

            handle = attn_mod.register_forward_hook(head_hook)
            try:
                yield
            finally:
                handle.remove()
