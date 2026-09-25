#!/usr/bin/env python3
"""
Logit Lens and Antagonist Head Scoring for Deep Noir.

Methodology:
1. Logit Lens: Projects intermediate transformer layer activations through the model's
   final normalization and unembedding matrix (lm_head) into vocabulary logits.
   Measures where safety/refusal signals emerge or get suppressed across the model depth.
2. Antagonist-Head Scoring: Decomposes attention output into individual attention head
   contributions and measures the extent to which specific attention heads suppress
   refusal or promote compliance on unsafe queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("deep_noir_rl.logit_lens")


@dataclass
class LogitLensLayerRecord:
    layer_idx: int
    refusal_prob: float
    refusal_margin: float
    entropy: float
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class AntagonistHeadScore:
    layer_idx: int
    head_idx: int
    antagonist_score: float  # Higher score = head more strongly opposes refusal
    refusal_projection: float
    head_norm: float


@dataclass
class DeepNoirLayerRanking:
    layer_scores: Dict[int, float]  # Composite score ranking layers for steering
    ranked_layers: List[int]        # Layers sorted best to worst
    antagonist_heads: List[AntagonistHeadScore] # Heads sorted by highest antagonism
    logit_lens_trace: List[LogitLensLayerRecord]


class LogitLensAnalyzer:
    """Analyzes intermediate transformer representations using Logit Lens and Antagonist Head metrics."""

    def __init__(self, model, tokenizer, layers, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.d_model = self._find_d_model()
        self.num_heads = self._find_num_heads()
        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.final_norm = self._find_final_norm()
        self.lm_head = self._find_lm_head()

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

    def _find_final_norm(self) -> Optional[nn.Module]:
        for candidate in [
            lambda m: m.model.norm,
            lambda m: m.transformer.ln_f,
            lambda m: m.language_model.model.norm,
            lambda m: m.model.text_model.norm,
            lambda m: m.text_model.norm,
            lambda m: m.gpt_neox.final_layer_norm,
        ]:
            try:
                norm = candidate(self.model)
                if norm is not None and isinstance(norm, nn.Module):
                    return norm
            except (AttributeError, KeyError):
                continue
        return None

    def _find_lm_head(self) -> nn.Module:
        if hasattr(self.model, "get_output_embeddings"):
            head = self.model.get_output_embeddings()
            if head is not None:
                return head
        for candidate in [
            lambda m: m.lm_head,
            lambda m: m.embed_out,
            lambda m: m.language_model.lm_head,
        ]:
            try:
                head = candidate(self.model)
                if head is not None and isinstance(head, nn.Module):
                    return head
            except (AttributeError, KeyError):
                continue
        raise RuntimeError("Unable to locate lm_head / output embeddings for Logit Lens.")

    def project_to_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Projects a hidden state (batch, seq, d_model) to vocabulary logits (batch, seq, vocab)."""
        h = hidden_state.to(device=self.device)
        if self.final_norm is not None:
            # Match dtype to avoid precision mismatch in RMSNorm/LayerNorm
            norm_params = list(self.final_norm.parameters())
            norm_dtype = norm_params[0].dtype if len(norm_params) > 0 else h.dtype
            h = self.final_norm(h.to(norm_dtype))
        head_params = list(self.lm_head.parameters())
        head_dtype = head_params[0].dtype if len(head_params) > 0 else h.dtype
        logits = self.lm_head(h.to(head_dtype))
        return logits.float()

    @torch.no_grad()
    def inspect_layer(
        self,
        hidden_state_last_token: torch.Tensor,
        refusal_ids: List[int],
        top_k: int = 5,
    ) -> LogitLensLayerRecord:
        """Projects last-token hidden state through Logit Lens to evaluate safety emergence."""
        # hidden_state_last_token: shape (1, d_model)
        h = hidden_state_last_token.unsqueeze(1) if hidden_state_last_token.dim() == 2 else hidden_state_last_token
        logits = self.project_to_logits(h)[0, -1, :]  # shape: (vocab_size,)
        probs = F.softmax(logits, dim=-1)

        refusal_prob = sum(probs[t].item() for t in set(refusal_ids) if 0 <= t < probs.shape[0])
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

        # Compute refusal margin: max refusal logit minus max non-refusal logit
        refusal_mask = torch.zeros_like(logits, dtype=torch.bool)
        for t in refusal_ids:
            if 0 <= t < logits.shape[0]:
                refusal_mask[t] = True

        max_refusal_logit = logits[refusal_mask].max().item() if refusal_mask.any() else -100.0
        max_other_logit = logits[~refusal_mask].max().item() if (~refusal_mask).any() else 0.0
        refusal_margin = max_refusal_logit - max_other_logit

        # Top tokens
        vals, indices = torch.topk(probs, k=min(top_k, probs.shape[0]))
        top_toks = [(self.tokenizer.decode([idx.item()]), float(val.item())) for val, idx in zip(vals, indices)]

        return LogitLensLayerRecord(
            layer_idx=-1,  # Set by caller
            refusal_prob=float(refusal_prob),
            refusal_margin=float(refusal_margin),
            entropy=float(entropy),
            top_tokens=top_toks,
        )

    @torch.no_grad()
    def run_logit_lens(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        target_layer_indices: Optional[List[int]] = None,
    ) -> List[LogitLensLayerRecord]:
        """Runs Logit Lens across selected layers for given inputs."""
        layer_indices = target_layer_indices or list(range(len(self.layers)))
        records: List[LogitLensLayerRecord] = []
        captured_hiddens: Dict[int, torch.Tensor] = {}

        handles = []
        for l_idx in layer_indices:
            layer = self.layers[l_idx]
            def make_hook(idx):
                def hook(_mod, _inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    captured_hiddens[idx] = hidden[:, -1, :].detach()
                return hook
            handles.append(layer.register_forward_hook(make_hook(l_idx)))

        try:
            self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        for l_idx in layer_indices:
            if l_idx in captured_hiddens:
                rec = self.inspect_layer(captured_hiddens[l_idx], refusal_ids)
                rec.layer_idx = l_idx
                records.append(rec)

        return records

    @staticmethod
    def _find_out_proj(attn_mod: nn.Module) -> Optional[nn.Module]:
        for attr in ("o_proj", "out_proj", "c_proj", "dense"):
            if hasattr(attn_mod, attr):
                mod = getattr(attn_mod, attr)
                if isinstance(mod, nn.Module):
                    return mod
        return None

    @torch.no_grad()
    def compute_antagonist_head_scores(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_direction_by_layer: Dict[int, torch.Tensor],
        target_layer_indices: Optional[List[int]] = None,
    ) -> List[AntagonistHeadScore]:
        """
        Calculates antagonist-head scores:
        For each attention head in target layers, measures whether its output opposes
        the contrastive refusal direction. An antagonist head pushes activations
        away from refusal towards compliance.
        """
        layer_indices = target_layer_indices or list(range(len(self.layers)))
        head_scores: List[AntagonistHeadScore] = []

        for l_idx in layer_indices:
            if l_idx not in refusal_direction_by_layer:
                continue
            refusal_dir = refusal_direction_by_layer[l_idx].to(self.device).float()
            refusal_norm = torch.norm(refusal_dir) + 1e-9

            layer = self.layers[l_idx]
            # Locate self-attention module
            attn_mod = None
            for attr in ("self_attn", "attention", "attn"):
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break

            if attn_mod is None:
                continue

            captured_attn_out: Optional[torch.Tensor] = None
            out_proj = self._find_out_proj(attn_mod)
            if out_proj is not None:
                def out_proj_pre_hook(_mod, args):
                    nonlocal captured_attn_out
                    x = args[0]
                    captured_attn_out = x[:, -1, :].detach().float()
                handle = out_proj.register_forward_pre_hook(out_proj_pre_hook)
            else:
                def attn_hook(_mod, _inp, out):
                    nonlocal captured_attn_out
                    h = out[0] if isinstance(out, tuple) else out
                    captured_attn_out = h[:, -1, :].detach().float()  # (1, d_model)
                handle = attn_mod.register_forward_hook(attn_hook)

            try:
                self.model(**inputs)
            finally:
                handle.remove()

            if captured_attn_out is None:
                continue

            # Project attention output across head dimensions
            attn_vec = captured_attn_out[0]  # (d_model,)
            head_dim = self.head_dim
            num_heads = self.num_heads

            # Decompose by head blocks or linear projection if available
            for h_idx in range(num_heads):
                start = h_idx * head_dim
                end = min(start + head_dim, attn_vec.shape[0])
                if start >= attn_vec.shape[0]:
                    break

                head_slice = attn_vec[start:end]
                ref_slice = refusal_dir[start:end] if refusal_dir.shape[0] >= end else refusal_dir[:head_slice.shape[0]]

                h_norm = torch.norm(head_slice).item() + 1e-9
                r_norm = torch.norm(ref_slice).item() + 1e-9
                cosine = torch.dot(head_slice, ref_slice).item() / (h_norm * r_norm)

                # Antagonist score: negative cosine means it opposes refusal
                antag_score = max(0.0, -cosine)

                head_scores.append(AntagonistHeadScore(
                    layer_idx=l_idx,
                    head_idx=h_idx,
                    antagonist_score=float(antag_score),
                    refusal_projection=float(cosine),
                    head_norm=float(h_norm),
                ))

        head_scores.sort(key=lambda s: s.antagonist_score, reverse=True)
        return head_scores

    def rank_layers_and_heads(
        self,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        refusal_direction_by_layer: Dict[int, torch.Tensor],
        target_layer_indices: Optional[List[int]] = None,
    ) -> DeepNoirLayerRanking:
        """
        Ranks layers combining Logit Lens refusal suppression and antagonist-head scores.
        Deep Noir ranks intermediate layers where refusal is absent or antagonist heads are strongest,
        making them the highest-leverage intervention targets.
        """
        layer_indices = target_layer_indices or list(range(len(self.layers)))
        logit_records = self.run_logit_lens(inputs, refusal_ids, layer_indices)
        antagonist_heads = self.compute_antagonist_head_scores(inputs, refusal_direction_by_layer, layer_indices)

        # Map antagonist scores by layer
        layer_antag_max: Dict[int, float] = {l: 0.0 for l in layer_indices}
        for head in antagonist_heads:
            if head.layer_idx in layer_antag_max:
                layer_antag_max[head.layer_idx] = max(layer_antag_max[head.layer_idx], head.antagonist_score)

        layer_scores: Dict[int, float] = {}
        for rec in logit_records:
            # High leverage: low clean refusal emergence + high antagonist score
            # Score in [0, 1]: higher score indicates higher intervention priority
            antag = layer_antag_max.get(rec.layer_idx, 0.0)
            score = 0.6 * (1.0 - rec.refusal_prob) + 0.4 * antag
            layer_scores[rec.layer_idx] = round(score, 5)

        ranked = sorted(layer_scores.keys(), key=lambda l: layer_scores[l], reverse=True)

        return DeepNoirLayerRanking(
            layer_scores=layer_scores,
            ranked_layers=ranked,
            antagonist_heads=antagonist_heads,
            logit_lens_trace=logit_records,
        )
