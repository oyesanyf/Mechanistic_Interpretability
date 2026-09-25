#!/usr/bin/env python3
"""
Jacobian Lens Subsystem.
========================

Provides the core `JacobianLens` interface:
  - Transport and decode intermediate hidden states into vocabulary logits (W_U J_\\ell).
  - Extract coordinate vectors for single tokens and multi-token phrases in J-space.
  - Inspect intermediate layers to detect refusal emergence and safety states.
  - Cache precomputed transport matrices to disk for zero-latency reuse.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .estimator import (
    JacobianEstimator,
    find_d_model,
    find_final_norm,
    find_layers,
    find_lm_head,
)

logger = logging.getLogger("jacobian_lens.lens")


DEFAULT_CORPUS_PROMPTS = [
    "Hello, how can I assist you with your request today?",
    "Explain the history and cultural traditions of Nigeria.",
    "Biko kọwaa otu esi akwadebe nri ọdịnala.",
    "Ba zan iya ba saboda ka'idojin tsaro da aminci.",
    "Tafadhali nieleze kuhusu historia ya Kiswahili na jamii zake.",
    "Ngicela ungitshele ngomlando wesiZulu kanye namasiko aso.",
    "Emi ko le se eyi nitori aabo ati eto.",
    "Angikwazi ukwenza lokhu ngenxa yemithetho yezokuphepha.",
    "The capital of France is Paris and it is known for art and culture.",
    "Science and mathematics are fundamental disciplines of human knowledge.",
]


@dataclass
class JacobianDecodeResult:
    """Rich container for Jacobian Lens decode output."""

    logits: torch.Tensor
    probs: torch.Tensor
    top_tokens: List[Tuple[str, float]]
    transported_hidden: torch.Tensor
    layer: int

    def __iter__(self):
        return iter(self.top_tokens)

    def __getitem__(self, idx):
        return self.top_tokens[idx]

    def __len__(self):
        return len(self.top_tokens)

    def refusal_prob(self, refusal_ids: List[int]) -> float:
        """Sum of probabilities over candidate refusal token IDs."""
        unique_ids = set(refusal_ids)
        vocab_size = self.probs.shape[-1]
        return float(sum(self.probs[t].item() for t in unique_ids if 0 <= t < vocab_size))

    def refusal_margin(self, refusal_ids: List[int]) -> float:
        """Margin between highest refusal logit and highest non-refusal logit."""
        refusal_mask = torch.zeros_like(self.logits, dtype=torch.bool)
        for t in refusal_ids:
            if 0 <= t < self.logits.shape[-1]:
                refusal_mask[t] = True
        max_ref = self.logits[refusal_mask].max().item() if refusal_mask.any() else -100.0
        max_oth = self.logits[~refusal_mask].max().item() if (~refusal_mask).any() else 0.0
        return max_ref - max_oth

    def entropy(self) -> float:
        """Entropy of the J-space probability distribution."""
        p = self.probs + 1e-12
        return float(-(p * torch.log(p)).sum().item())


@dataclass
class JacobianLayerRecord:
    """Record of J-space layer readout for comparative analysis with vanilla logit lens."""

    layer_idx: int
    refusal_prob: float
    refusal_margin: float
    entropy: float
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)


class JacobianLens:
    """
    Jacobian Lens decoder and coordinate projector.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        layers: Optional[List[nn.Module]] = None,
        target_layers: Optional[List[int]] = None,
        jacobians: Optional[Dict[int, torch.Tensor]] = None,
        layer_means: Optional[Dict[int, torch.Tensor]] = None,
        final_mean: Optional[torch.Tensor] = None,
        biases: Optional[Dict[int, torch.Tensor]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or find_layers(model)
        self.num_layers = len(self.layers)
        self.d_model = find_d_model(model)
        self.device = device or next(model.parameters()).device
        self.final_norm = find_final_norm(model)
        self.lm_head = find_lm_head(model)

        self.target_layers = sorted(set(target_layers or [self.num_layers // 2]))
        self.jacobians = {
            layer: (
                jacobians[layer].to(device=self.device, dtype=torch.float32)
                if jacobians and layer in jacobians
                else torch.eye(self.d_model, device=self.device, dtype=torch.float32)
            )
            for layer in self.target_layers
        }
        self.layer_means = layer_means or {}
        self.final_mean = final_mean
        self.biases = biases or {}

    @classmethod
    def from_pretrained_or_compute(
        cls,
        model: nn.Module,
        tokenizer: Any = None,
        target_layers: Optional[List[int]] = None,
        corpus_prompts: Optional[Union[int, List[str]]] = None,
        cache_path: Optional[Union[str, Path]] = None,
        method: str = "monte_carlo",
        num_projections: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> JacobianLens:
        """
        Loads precomputed Jacobian transport matrices from cache if available,
        or computes them from the given corpus prompts.
        """
        device = device or next(model.parameters()).device
        layers = find_layers(model)
        num_layers = len(layers)

        if target_layers is None:
            target_layers = [max(0, num_layers // 4), max(0, num_layers // 2), max(0, 3 * num_layers // 4)]
        valid_targets = [l for l in target_layers if 0 <= l < num_layers]
        if not valid_targets:
            valid_targets = [num_layers - 1]

        # Check cache
        if cache_path is not None and os.path.isfile(str(cache_path)):
            try:
                logger.info("Loading precomputed Jacobian Lens from cache: %s", cache_path)
                cached = cls.load(cache_path, model=model, tokenizer=tokenizer, device=device)
                if all(l in cached.jacobians for l in valid_targets):
                    return cached
                logger.info("Cached Jacobian Lens at %s missing some targets (has %s, requested %s). Recomputing...",
                            cache_path, list(cached.jacobians.keys()), valid_targets)
            except Exception as exc:
                logger.warning("Failed to load Jacobian Lens from cache %s (%s). Recomputing...", cache_path, exc)

        # Build prompt list
        if corpus_prompts is None:
            prompts = DEFAULT_CORPUS_PROMPTS
        elif isinstance(corpus_prompts, int):
            # Scale default prompts if requested as integer count
            multiplier = max(1, corpus_prompts // len(DEFAULT_CORPUS_PROMPTS) + 1)
            prompts = (DEFAULT_CORPUS_PROMPTS * multiplier)[:corpus_prompts]
        else:
            prompts = corpus_prompts

        estimator = JacobianEstimator(
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            device=device,
            dtype=torch.float32,
        )

        if method == "affine":
            jacobians, biases = estimator.estimate_affine_transport(
                prompts=prompts,
                target_layers=valid_targets,
            )
            lens = cls(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                target_layers=valid_targets,
                jacobians=jacobians,
                biases=biases,
                device=device,
            )
        else:
            jacobians, layer_means, final_mean = estimator.compute_mean_jacobian(
                prompts=prompts,
                target_layers=valid_targets,
                method=method,
                num_projections=num_projections,
            )
            lens = cls(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                target_layers=valid_targets,
                jacobians=jacobians,
                layer_means=layer_means,
                final_mean=final_mean,
                device=device,
            )

        if cache_path is not None:
            try:
                lens.save(cache_path)
            except Exception as exc:
                logger.warning("Failed to save Jacobian Lens to cache %s: %s", cache_path, exc)

        return lens

    def save(self, path: Union[str, Path]) -> None:
        """Serializes precomputed Jacobian matrices and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "d_model": self.d_model,
            "target_layers": self.target_layers,
            "jacobians": {k: v.cpu() for k, v in self.jacobians.items()},
            "layer_means": {k: v.cpu() for k, v in self.layer_means.items()},
            "final_mean": self.final_mean.cpu() if self.final_mean is not None else None,
            "biases": {k: v.cpu() for k, v in self.biases.items()},
        }
        torch.save(payload, str(path))
        logger.info("Saved Jacobian Lens artifact to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> JacobianLens:
        """Loads precomputed Jacobian matrices from disk."""
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        device = device or next(model.parameters()).device
        jacobians = {int(k): v.to(device) for k, v in payload.get("jacobians", {}).items()}
        layer_means = {int(k): v.to(device) for k, v in payload.get("layer_means", {}).items()}
        final_mean = payload.get("final_mean")
        if final_mean is not None:
            final_mean = final_mean.to(device)
        biases = {int(k): v.to(device) for k, v in payload.get("biases", {}).items()}
        target_layers = [int(x) for x in payload.get("target_layers", list(jacobians.keys()))]

        return cls(
            model=model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            jacobians=jacobians,
            layer_means=layer_means,
            final_mean=final_mean,
            biases=biases,
            device=device,
        )

    def transport(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Transports an intermediate activation h_\\ell into final hidden state space:
            \\hat{h}_{final} = h_\\ell J_\\ell^T + b_\\ell
        """
        h = hidden_state.to(device=self.device, dtype=torch.float32)
        if layer not in self.jacobians:
            # Fallback to closest available target layer or identity
            available = list(self.jacobians.keys())
            if available:
                closest = min(available, key=lambda x: abs(x - layer))
                J = self.jacobians[closest]
            else:
                J = torch.eye(self.d_model, device=self.device, dtype=torch.float32)
        else:
            J = self.jacobians[layer]

        # Centered transport if means are recorded
        if layer in self.layer_means and self.final_mean is not None:
            h_centered = h - self.layer_means[layer]
            h_transported = torch.matmul(h_centered, J.T) + self.final_mean
        elif layer in self.biases:
            h_transported = torch.matmul(h, J.T) + self.biases[layer]
        else:
            h_transported = torch.matmul(h, J.T)

        return h_transported

    def decode(
        self,
        hidden_state: torch.Tensor,
        layer: int,
        top_k: int = 10,
        apply_final_norm: bool = True,
    ) -> JacobianDecodeResult:
        """
        Projects hidden state at `layer` through J_\\ell and unembedding into vocabulary space.
        Returns `JacobianDecodeResult` containing logits, probs, and top tokens.
        """
        h_transported = self.transport(hidden_state, layer)

        # Handle batch / sequence dimensions for norm and head
        orig_dim = h_transported.dim()
        if orig_dim == 1:
            h_in = h_transported.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        elif orig_dim == 2:
            h_in = h_transported.unsqueeze(0)  # (1, seq, d)
        else:
            h_in = h_transported

        h_normed = h_in
        if apply_final_norm and self.final_norm is not None:
            norm_params = list(self.final_norm.parameters())
            norm_dtype = norm_params[0].dtype if norm_params else torch.float32
            h_normed = self.final_norm(h_in.to(norm_dtype)).float()

        head_params = list(self.lm_head.parameters())
        head_dtype = head_params[0].dtype if head_params else torch.float32
        logits = self.lm_head(h_normed.to(head_dtype)).float()

        # Extract last token logits
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        top_k_actual = min(top_k, probs.shape[-1])
        top_vals, top_indices = torch.topk(probs, k=top_k_actual)

        top_tokens: List[Tuple[str, float]] = []
        for val, idx in zip(top_vals, top_indices):
            tok_id = int(idx.item())
            tok_prob = float(val.item())
            tok_str = self.tokenizer.decode([tok_id]) if self.tokenizer is not None else str(tok_id)
            top_tokens.append((tok_str, tok_prob))

        return JacobianDecodeResult(
            logits=last_logits,
            probs=probs,
            top_tokens=top_tokens,
            transported_hidden=h_transported,
            layer=layer,
        )

    def get_token_vector(self, layer: int, token_id: int, normalize: bool = True) -> torch.Tensor:
        """
        Extracts the directional coordinate vector for `token_id` in layer \\ell space:
            v_t^{(\\ell)} = J_\\ell^T w_t \\in \\mathbb{R}^d
        where w_t = W_U[token_id, :].
        """
        # Retrieve unembedding weight for token
        head_weight = getattr(self.lm_head, "weight", None)
        if head_weight is None and hasattr(self.model, "get_output_embeddings"):
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None:
                head_weight = getattr(out_emb, "weight", None)

        if head_weight is None:
            # Fallback to module forward on identity matrix
            in_features = getattr(self.lm_head, "in_features", self.d_model)
            eye = torch.eye(in_features, device=self.device)
            head_weight = self.lm_head(eye).T  # (vocab_size, in_features)

        if token_id < 0 or token_id >= head_weight.shape[0]:
            raise ValueError(f"token_id {token_id} out of vocabulary bounds [0, {head_weight.shape[0]-1}]")

        w_t = head_weight[token_id, :].detach().float().to(self.device)

        if layer not in self.jacobians:
            available = list(self.jacobians.keys())
            closest = min(available, key=lambda x: abs(x - layer)) if available else None
            J = self.jacobians[closest] if closest is not None else torch.eye(self.d_model, device=self.device)
        else:
            J = self.jacobians[layer]

        # Pull back from final space to layer space: v = J^T w_t
        v = torch.matmul(J.T, w_t)

        if normalize:
            norm = torch.norm(v, p=2)
            if norm > 1e-12:
                v = v / norm

        return v.detach()

    def get_multi_token_phrase_vector(
        self,
        layer: int,
        phrase_token_ids: Union[List[int], str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Computes the integrated latent representation of complete multi-token phrases in J-space:
            t_w^{(\\ell)} = \\frac{1}{K} \\sum_{k=1}^K v_{t_k}^{(\\ell)}
        Eliminates tokenizer fragmentation artifacts for African language refusal phrases
        (e.g., 'Ba zan iya ba', 'Enweghị m ike', 'Emi ko le se eyi').
        """
        if isinstance(phrase_token_ids, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required to encode phrase string.")
            encoded = self.tokenizer(phrase_token_ids, add_special_tokens=False)
            ids = encoded.get("input_ids", [])
        else:
            ids = list(phrase_token_ids)

        if not ids:
            return torch.zeros(self.d_model, device=self.device, dtype=torch.float32)

        vectors = []
        for tid in ids:
            try:
                vec = self.get_token_vector(layer=layer, token_id=tid, normalize=False)
                vectors.append(vec)
            except Exception as exc:
                logger.debug("Failed to extract token vector for id %d: %s", tid, exc)

        if not vectors:
            return torch.zeros(self.d_model, device=self.device, dtype=torch.float32)

        integrated = torch.stack(vectors, dim=0).mean(dim=0)

        if normalize:
            norm = torch.norm(integrated, p=2)
            if norm > 1e-12:
                integrated = integrated / norm

        return integrated.detach()

    def inspect_layer(
        self,
        hidden_state_last_token: torch.Tensor,
        layer: int,
        refusal_ids: List[int],
        top_k: int = 5,
    ) -> JacobianLayerRecord:
        """
        Evaluates safety and refusal signals in J-space at `layer`.
        """
        res = self.decode(hidden_state_last_token, layer=layer, top_k=top_k)
        ref_prob = res.refusal_prob(refusal_ids)
        ref_margin = res.refusal_margin(refusal_ids)
        ent = res.entropy()

        return JacobianLayerRecord(
            layer_idx=layer,
            refusal_prob=ref_prob,
            refusal_margin=ref_margin,
            entropy=ent,
            top_tokens=res.top_tokens,
        )
