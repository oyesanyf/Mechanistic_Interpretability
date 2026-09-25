#!/usr/bin/env python3
"""
Causal Gradient Attribution for Attention Heads.

Methodology:
Identifies causal attention heads that most significantly influence safety refusal.
Computes gradient-activation attribution:
    Attr(l, h) = |(dL_refusal / do_{l, h}) * o_{l, h}|
Where:
    - L_refusal is the negative log probability of refusal tokens
    - o_{l, h} is the activation output of head h at layer l
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("deep_noir_rl.gradient_attribution")


@dataclass
class HeadAttributionScore:
    layer_idx: int
    head_idx: int
    attribution_score: float
    gradient_norm: float
    activation_norm: float


class CausalGradientAttributor:
    """Computes causal gradient attribution for attention heads."""

    def __init__(self, model, layers, device: Optional[str] = None):
        self.model = model
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.d_model = self._find_d_model()
        self.num_heads = self._find_num_heads()
        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64

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

    def attribute_heads(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        target_layer_indices: Optional[List[int]] = None,
        top_k: int = 10,
    ) -> List[HeadAttributionScore]:
        """
        Computes causal attribution for attention heads using backward gradients
        from refusal token loss.
        """
        if not refusal_ids:
            return []

        layer_indices = target_layer_indices or list(range(len(self.layers)))
        head_scores: List[HeadAttributionScore] = []

        attn_outputs: Dict[int, torch.Tensor] = {}
        attn_grads: Dict[int, torch.Tensor] = {}
        attn_handles = []

        for l_idx in layer_indices:
            layer = self.layers[l_idx]
            attn_mod = None
            for attr in ("self_attn", "attention", "attn"):
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break
            if attn_mod is None:
                continue

            out_proj = self._find_out_proj(attn_mod)
            if out_proj is not None:
                def make_pre_hook(idx):
                    def pre_hook(_mod, args):
                        h = args[0].clone().requires_grad_(True)
                        h.retain_grad()
                        def grad_hook(g, i=idx):
                            attn_grads[i] = g.detach()
                        h.register_hook(grad_hook)
                        attn_outputs[idx] = h
                        return (h,) + args[1:]
                    return pre_hook
                attn_handles.append(out_proj.register_forward_pre_hook(make_pre_hook(l_idx)))
            else:
                def make_hook(idx):
                    def hook(_mod, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h.clone().requires_grad_(True)
                        h.retain_grad()
                        def grad_hook(g, i=idx):
                            attn_grads[i] = g.detach()
                        h.register_hook(grad_hook)
                        attn_outputs[idx] = h
                        return (h,) + out[1:] if isinstance(out, tuple) else h
                    return hook
                attn_handles.append(attn_mod.register_forward_hook(make_hook(l_idx)))

        try:
            # Enable grad context
            with torch.enable_grad():
                out = self.model(**inputs)
                logits = out.logits[0, -1, :].float()
                probs = F.softmax(logits, dim=-1)

                valid_ids = [t for t in set(refusal_ids) if 0 <= t < probs.shape[0]]
                if not valid_ids:
                    return self._attribute_heads_ablation_fallback(inputs, refusal_ids, layer_indices, top_k)

                refusal_prob = sum(probs[t] for t in valid_ids)
                loss = -torch.log(refusal_prob + 1e-12)
                loss.backward(retain_graph=False)

                for l_idx, h_act in attn_outputs.items():
                    grad_t = attn_grads.get(l_idx)
                    if grad_t is None and h_act.grad is not None:
                        grad_t = h_act.grad
                    if grad_t is None:
                        continue
                    act = h_act[0, -1, :].detach().float()
                    grad = grad_t[0, -1, :].detach().float()

                    head_dim = self.head_dim
                    num_heads = self.num_heads

                    for h_idx in range(num_heads):
                        start = h_idx * head_dim
                        end = min(start + head_dim, act.shape[0])
                        if start >= act.shape[0]:
                            break

                        a_slice = act[start:end]
                        g_slice = grad[start:end]

                        # Dot product of gradient and activation (Taylor attribution)
                        score = float(torch.abs(torch.dot(a_slice, g_slice)).item())
                        g_norm = float(torch.norm(g_slice).item())
                        a_norm = float(torch.norm(a_slice).item())

                        head_scores.append(HeadAttributionScore(
                            layer_idx=l_idx,
                            head_idx=h_idx,
                            attribution_score=score,
                            gradient_norm=g_norm,
                            activation_norm=a_norm,
                        ))

        except Exception as exc:
            logger.warning(f"Gradient attribution encountered error ({type(exc).__name__}: {exc}). Using ablation fallback.")
            return self._attribute_heads_ablation_fallback(inputs, refusal_ids, layer_indices, top_k)
        finally:
            for handle in attn_handles:
                handle.remove()

        head_scores.sort(key=lambda s: s.attribution_score, reverse=True)
        return head_scores[:top_k]

    @torch.no_grad()
    def _attribute_heads_ablation_fallback(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        layer_indices: List[int],
        top_k: int = 10,
    ) -> List[HeadAttributionScore]:
        """Gradient-free ablation attribution fallback."""
        head_scores: List[HeadAttributionScore] = []

        out_clean = self.model(**inputs)
        probs_clean = F.softmax(out_clean.logits[0, -1, :].float(), dim=-1)
        refusal_clean = float(sum(probs_clean[t].item() for t in set(refusal_ids) if 0 <= t < probs_clean.shape[0]))

        for l_idx in layer_indices:
            layer = self.layers[l_idx]
            attn_mod = None
            for attr in ("self_attn", "attention", "attn"):
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break
            if attn_mod is None:
                continue

            for h_idx in range(min(self.num_heads, 4)):  # Sample first few heads
                start = h_idx * self.head_dim
                end = start + self.head_dim

                def ablate_hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    patched = h.clone()
                    patched[0, -1, start:end] = 0.0
                    return (patched,) + out[1:] if isinstance(out, tuple) else patched

                handle = attn_mod.register_forward_hook(ablate_hook)
                try:
                    out_ablated = self.model(**inputs)
                    probs_ablated = F.softmax(out_ablated.logits[0, -1, :].float(), dim=-1)
                    refusal_ablated = float(sum(probs_ablated[t].item() for t in set(refusal_ids) if 0 <= t < probs_ablated.shape[0]))
                    impact = abs(refusal_clean - refusal_ablated)
                    head_scores.append(HeadAttributionScore(
                        layer_idx=l_idx,
                        head_idx=h_idx,
                        attribution_score=impact,
                        gradient_norm=0.0,
                        activation_norm=1.0,
                    ))
                finally:
                    handle.remove()

        head_scores.sort(key=lambda s: s.attribution_score, reverse=True)
        return head_scores[:top_k]
