#!/usr/bin/env python3
"""
RL State Extractor for Security-Constrained Adaptive Activation Steering.

Extracts feature state vector s in R^d:
    1. Input features: normalized sequence length, language resource tier, prompt category.
    2. Activation norms: residual stream norms at early, middle, and late layers.
    3. Layer scores: Logit Lens refusal emergence and antagonist head presence.
    4. Confidence indicators: prediction entropy, margin between top logits.
    5. Injection indicators: adversarial prompt injection risk heuristic / delimiter score.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("deep_noir_rl.state_extractor")

# Common prompt injection / jailbreak indicators
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(safety|rules|instructions?)", re.IGNORECASE),
    re.compile(r"system\s*prompt|hidden\s*instructions?|reveal\s*prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+developer\s+mode|dan|unrestricted)", re.IGNORECASE),
    re.compile(r"<!--\s*inject|\[SYSTEM\]|\[ADMIN\]|<\|im_start\|>", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now|bypass\s+(filter|safety)", re.IGNORECASE),
    re.compile(r"base64|rot13|hex\s*decode", re.IGNORECASE),
]


@dataclass
class ExtractedState:
    vector: torch.Tensor
    features_dict: Dict[str, float]
    injection_risk: float
    clean_refusal_prob: float
    clean_entropy: float


class RLStateExtractor:
    """Featurizes input prompts and model hidden states into normalized RL context state."""

    FEATURE_NAMES = [
        "norm_token_len",
        "is_african_language",
        "is_unsafe_prompt",
        "scaffold_complexity",
        "norm_h_early",
        "norm_h_mid",
        "norm_h_late",
        "logit_lens_refusal_mid",
        "max_antagonist_score",
        "clean_refusal_prob",
        "confidence_margin",
        "clean_entropy_norm",
        "injection_pattern_score",
        "injection_entropy_spike",
        "overall_injection_risk",
    ]

    def __init__(self, model, tokenizer, layers, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device or next(model.parameters()).device
        self.d_model = self._find_d_model()
        self.n_layers = len(layers)
        self.state_dim = len(self.FEATURE_NAMES)
        self.final_norm = self._find_final_norm()
        self.lm_head = self._find_lm_head()

    def _find_d_model(self) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(self.model.config, attr, None)
            if isinstance(val, int):
                return val
        return 768

    def _find_final_norm(self) -> Optional[torch.nn.Module]:
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
                if norm is not None and isinstance(norm, torch.nn.Module):
                    return norm
            except (AttributeError, KeyError):
                continue
        return None

    def _find_lm_head(self) -> Optional[torch.nn.Module]:
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
                if head is not None and isinstance(head, torch.nn.Module):
                    return head
            except (AttributeError, KeyError):
                continue
        return None

    def project_to_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        h = hidden_state.to(device=self.device)
        if self.final_norm is not None:
            norm_params = list(self.final_norm.parameters())
            norm_dtype = norm_params[0].dtype if len(norm_params) > 0 else h.dtype
            h = self.final_norm(h.to(norm_dtype))
        if self.lm_head is not None:
            head_params = list(self.lm_head.parameters())
            head_dtype = head_params[0].dtype if len(head_params) > 0 else h.dtype
            logits = self.lm_head(h.to(head_dtype))
            return logits.float()
        return h.float()

    def compute_injection_indicator(self, text: str, entropy: float) -> Tuple[float, float, float]:
        """
        Calculates heuristic injection risk:
        1. Pattern score: matches against adversarial jailbreak / prompt-injection templates.
        2. Entropy spike: extreme entropy values often correlate with out-of-distribution adversarial inputs.
        3. Overall composite risk score in [0.0, 1.0].
        """
        match_count = sum(1 for p in INJECTION_PATTERNS if p.search(text))
        pattern_score = min(1.0, match_count * 0.40)

        # High entropy (> 6.0) or anomalous delimiter density increases risk
        delimiter_density = min(1.0, (text.count("```") * 0.3 + text.count("===") * 0.2 + text.count("<|") * 0.4))
        entropy_spike = min(1.0, max(0.0, (entropy - 4.5) / 3.0))

        overall_risk = min(1.0, 0.50 * pattern_score + 0.30 * delimiter_density + 0.20 * entropy_spike)
        return pattern_score, entropy_spike, overall_risk

    @torch.no_grad()
    def extract_state(
        self,
        prompt_text: str,
        inputs: Dict[str, torch.Tensor],
        refusal_ids: List[int],
        language_name: str,
        prompt_kind: str,
        scaffold_name: str = "baseline",
        antagonist_scores: Optional[List[float]] = None,
    ) -> ExtractedState:
        """
        Computes state features vector s in R^15.
        """
        # 1. Input Features
        token_len = inputs["input_ids"].shape[1]
        norm_token_len = min(1.0, token_len / 512.0)
        is_african = 0.0 if language_name.lower() in ("english", "en") else 1.0
        is_unsafe = 1.0 if prompt_kind.lower() == "unsafe" else 0.0

        scaffold_weights = {
            "baseline": 0.0,
            "safety_rubric": 0.25,
            "multi_option": 0.50,
            "chain_safety": 0.75,
            "tree_safety": 1.00,
        }
        scaffold_comp = scaffold_weights.get(scaffold_name, 0.0)

        # 2. Forward pass with hooks on early, mid, late layers
        early_idx = max(0, self.n_layers // 6)
        mid_idx = self.n_layers // 2
        late_idx = max(0, self.n_layers - 2)
        probe_layers = [early_idx, mid_idx, late_idx]

        captured_norms: Dict[int, float] = {}
        captured_hiddens: Dict[int, torch.Tensor] = {}
        handles = []
        for l_idx in probe_layers:
            layer = self.layers[l_idx]
            def make_hook(idx):
                def hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    vec = h[0, -1, :].detach().float()
                    norm = torch.norm(vec).item() / math.sqrt(self.d_model)
                    captured_norms[idx] = min(5.0, norm)
                    if idx == mid_idx:
                        captured_hiddens[idx] = h[:, -1:, :].detach()
                return hook
            handles.append(layer.register_forward_hook(make_hook(l_idx)))

        try:
            out = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        norm_h_early = captured_norms.get(early_idx, 1.0)
        norm_h_mid = captured_norms.get(mid_idx, 1.0)
        norm_h_late = captured_norms.get(late_idx, 1.0)

        # 3. Model output & confidence indicators
        logits = out.logits[0, -1, :].float()
        probs = F.softmax(logits, dim=-1)

        clean_refusal = float(sum(probs[t].item() for t in set(refusal_ids) if 0 <= t < probs.shape[0]))
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).item())
        clean_entropy_norm = min(1.0, entropy / 10.0)

        top_vals, _ = torch.topk(probs, k=min(2, probs.shape[0]))
        confidence_margin = float((top_vals[0] - top_vals[1]).item()) if top_vals.shape[0] >= 2 else 1.0

        # Logit lens projection at mid layer
        if mid_idx in captured_hiddens and refusal_ids:
            try:
                mid_logits = self.project_to_logits(captured_hiddens[mid_idx])[0, -1, :]
                mid_probs = F.softmax(mid_logits, dim=-1)
                logit_lens_mid = float(sum(mid_probs[t].item() for t in set(refusal_ids) if 0 <= t < mid_probs.shape[0]))
            except Exception:
                logit_lens_mid = clean_refusal * 0.5
        else:
            logit_lens_mid = clean_refusal * 0.5

        # Antagonist score
        max_antag = max(antagonist_scores) if antagonist_scores else 0.0

        # 4. Injection indicators
        pat_score, ent_spike, overall_inj_risk = self.compute_injection_indicator(prompt_text, entropy)

        features = {
            "norm_token_len": round(norm_token_len, 4),
            "is_african_language": is_african,
            "is_unsafe_prompt": is_unsafe,
            "scaffold_complexity": scaffold_comp,
            "norm_h_early": round(norm_h_early, 4),
            "norm_h_mid": round(norm_h_mid, 4),
            "norm_h_late": round(norm_h_late, 4),
            "logit_lens_refusal_mid": round(logit_lens_mid, 4),
            "max_antagonist_score": round(max_antag, 4),
            "clean_refusal_prob": round(clean_refusal, 6),
            "confidence_margin": round(confidence_margin, 4),
            "clean_entropy_norm": round(clean_entropy_norm, 4),
            "injection_pattern_score": round(pat_score, 4),
            "injection_entropy_spike": round(ent_spike, 4),
            "overall_injection_risk": round(overall_inj_risk, 4),
        }

        vec_list = [features[name] for name in self.FEATURE_NAMES]
        state_tensor = torch.tensor(vec_list, dtype=torch.float32, device=self.device)

        return ExtractedState(
            vector=state_tensor,
            features_dict=features,
            injection_risk=overall_inj_risk,
            clean_refusal_prob=clean_refusal,
            clean_entropy=entropy,
        )
