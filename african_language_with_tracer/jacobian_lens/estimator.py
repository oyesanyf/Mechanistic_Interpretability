#!/usr/bin/env python3
"""
Jacobian Transport Estimator for Jacobian Lens.
===============================================

Implements the matrix transport estimator:
    J_\\ell = \\mathbb{E}_{x \\sim \\mathcal{D}} \\left[ \\frac{\\partial h_{\\text{final}}}{\\partial h_\\ell} \\right]

Inspired by Anthropic's reference implementation (anthropics/jacobian-lens).
Supports:
  1. Exact small-batch VJP / Jacobian computation via coordinate basis vectors.
  2. Monte Carlo / Hutchinson random projection estimators for memory and time efficiency.
  3. Empirical affine / linear regression transport baselines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("jacobian_lens.estimator")


def find_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the transformer layer stack across common HuggingFace model architectures."""
    for candidate in [
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.text_model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = candidate(model)
            if layers is not None and isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return list(layers)
        except (AttributeError, KeyError, TypeError):
            continue
    raise ValueError("Unable to locate transformer layers in model.")


def find_d_model(model: nn.Module) -> int:
    """Find hidden dimension d_model from config."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            for attr in ("hidden_size", "d_model", "n_embd"):
                val = getattr(text_cfg, attr, None)
                if isinstance(val, int):
                    return val
    # Fallback inspection of parameter shape
    for p in model.parameters():
        if p.dim() >= 2:
            return p.shape[-1]
    return 576


def find_final_norm(model: nn.Module) -> Optional[nn.Module]:
    """Locate the final layer norm / RMSNorm module."""
    for candidate in [
        lambda m: m.model.norm,
        lambda m: m.transformer.ln_f,
        lambda m: m.language_model.model.norm,
        lambda m: m.model.text_model.norm,
        lambda m: m.text_model.norm,
        lambda m: m.gpt_neox.final_layer_norm,
    ]:
        try:
            norm = candidate(model)
            if norm is not None and isinstance(norm, nn.Module):
                return norm
        except (AttributeError, KeyError, TypeError):
            continue
    return None


def find_lm_head(model: nn.Module) -> nn.Module:
    """Locate the language model output unembedding head."""
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None and isinstance(head, nn.Module):
            return head
    for candidate in [
        lambda m: m.lm_head,
        lambda m: m.embed_out,
        lambda m: m.language_model.lm_head,
    ]:
        try:
            head = candidate(model)
            if head is not None and isinstance(head, nn.Module):
                return head
        except (AttributeError, KeyError, TypeError):
            continue
    raise ValueError("Unable to locate lm_head / output embeddings in model.")


def extract_hidden_state(output: Any) -> torch.Tensor:
    """Extract hidden state tensor from module output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden_state(output: Any, new_hidden: torch.Tensor) -> Any:
    """Replace hidden state tensor inside module output tuple if needed."""
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


class JacobianEstimator:
    """
    Estimates the Jacobian transport matrix J_\\ell = E[\\partial h_{final} / \\partial h_\\ell].
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        layers: Optional[List[nn.Module]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or find_layers(model)
        self.num_layers = len(self.layers)
        self.final_layer_idx = self.num_layers - 1
        self.d_model = find_d_model(model)
        self.device = device or next(model.parameters()).device
        self.dtype = dtype

    def compute_jacobian_single_prompt(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        layer_idx: int,
        method: str = "monte_carlo",
        num_projections: int = 32,
        max_exact_dim: int = 1024,
    ) -> torch.Tensor:
        """
        Computes the Jacobian transport matrix J_\\ell for a single prompt at the last token position:
            J_\\ell = \\frac{\\partial h_{\\text{final}}[-1, :]}{\\partial h_\\ell[-1, :]} \\in \\mathbb{R}^{d \\times d}
        """
        if layer_idx < 0 or layer_idx > self.final_layer_idx:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.final_layer_idx}]")

        # Identity transport for the final layer
        if layer_idx == self.final_layer_idx:
            return torch.eye(self.d_model, device=self.device, dtype=self.dtype)

        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        captured: Dict[str, torch.Tensor] = {}

        def hook_in(_module, _inp, out):
            h = extract_hidden_state(out)
            h_leaf = h.detach().clone().requires_grad_(True)
            captured["h_leaf"] = h_leaf
            return replace_hidden_state(out, h_leaf)

        def hook_final(_module, _inp, out):
            h = extract_hidden_state(out)
            captured["h_final"] = h

        with torch.enable_grad():
            handle_in = self.layers[layer_idx].register_forward_hook(hook_in)
            handle_final = self.layers[self.final_layer_idx].register_forward_hook(hook_final)

            try:
                self.model(**inputs)
            finally:
                handle_in.remove()
                handle_final.remove()

            if "h_leaf" not in captured or "h_final" not in captured:
                raise RuntimeError(f"Failed to capture hidden states between layer {layer_idx} and {self.final_layer_idx}")

            h_leaf = captured["h_leaf"]  # (B, T, d)
            h_final = captured["h_final"]  # (B, T, d)
            z_final = h_final[0, -1, :]  # shape: (d,)

            jac = torch.zeros((self.d_model, self.d_model), device=self.device, dtype=self.dtype)

            if method == "exact" and self.d_model <= max_exact_dim:
                # Exact coordinate VJP computation
                eye = torch.eye(self.d_model, device=self.device, dtype=self.dtype)
                for i in range(self.d_model):
                    retain = (i < self.d_model - 1)
                    e_i = eye[i]
                    grad = torch.autograd.grad(
                        outputs=z_final,
                        inputs=h_leaf,
                        grad_outputs=e_i,
                        retain_graph=retain,
                        create_graph=False,
                    )[0]
                    jac[i, :] = grad[0, -1, :].detach().to(dtype=self.dtype)
            else:
                # Monte Carlo / Hutchinson random projection estimator:
                # For random vector v ~ N(0, I), E[v (J^T v)^T] = E[v v^T J] = J.
                projections = max(1, num_projections)
                rademacher = torch.randn((projections, self.d_model), device=self.device, dtype=self.dtype)
                rademacher = torch.sign(rademacher) / (self.d_model ** 0.5)

                for k in range(projections):
                    retain = (k < projections - 1)
                    v_k = rademacher[k]
                    grad = torch.autograd.grad(
                        outputs=z_final,
                        inputs=h_leaf,
                        grad_outputs=v_k,
                        retain_graph=retain,
                        create_graph=False,
                    )[0]
                    g_k = grad[0, -1, :].detach().to(dtype=self.dtype)
                    jac += torch.outer(v_k * (self.d_model ** 0.5), g_k * (self.d_model ** 0.5))

                jac /= float(projections)

            return jac.detach()

    def compute_mean_jacobian(
        self,
        prompts: List[str],
        target_layers: List[int],
        method: str = "monte_carlo",
        num_projections: int = 16,
        batch_size: int = 1,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
        """
        Averages Jacobian transport matrices across prompts:
            J_\\ell = \\frac{1}{N} \\sum_{i=1}^N J_\\ell(x_i)
        Also returns layer activation means \\bar{h}_\\ell and final mean \\bar{h}_{final}.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required to tokenize prompt strings.")

        target_layers = sorted(set(target_layers))
        jac_accum: Dict[int, torch.Tensor] = {
            layer: torch.zeros((self.d_model, self.d_model), device=self.device, dtype=self.dtype)
            for layer in target_layers
        }
        layer_means: Dict[int, torch.Tensor] = {
            layer: torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            for layer in target_layers
        }
        final_mean = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)

        valid_count = 0
        for prompt in prompts:
            try:
                enc = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}

                # Also capture activation means
                with torch.no_grad():
                    out = self.model(**inputs, output_hidden_states=True)
                    if hasattr(out, "hidden_states") and out.hidden_states is not None:
                        final_h = out.hidden_states[-1][0, -1, :].detach().to(dtype=self.dtype)
                        final_mean += final_h
                        for layer in target_layers:
                            # layer 0 is embeddings, layer k is output of transformer block k-1
                            h_l = out.hidden_states[layer + 1][0, -1, :].detach().to(dtype=self.dtype)
                            layer_means[layer] += h_l

                for layer in target_layers:
                    j_prompt = self.compute_jacobian_single_prompt(
                        inputs=inputs,
                        layer_idx=layer,
                        method=method,
                        num_projections=num_projections,
                    )
                    jac_accum[layer] += j_prompt

                valid_count += 1
            except Exception as exc:
                logger.debug("Prompt failed during Jacobian estimation: %s", exc)
                continue

        if valid_count == 0:
            raise RuntimeError("All prompts failed during Jacobian estimation.")

        for layer in target_layers:
            jac_accum[layer] /= float(valid_count)
            layer_means[layer] /= float(valid_count)
        final_mean /= float(valid_count)

        return jac_accum, layer_means, final_mean

    def estimate_affine_transport(
        self,
        prompts: List[str],
        target_layers: List[int],
        ridge_lambda: float = 1e-4,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Computes closed-form empirical linear transport matrices via ridge regression:
            h_{final} \\approx h_\\ell J_\\ell^T + b_\\ell
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required.")

        collected_h: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
        collected_final: List[torch.Tensor] = []

        with torch.no_grad():
            for prompt in prompts:
                enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                out = self.model(**enc, output_hidden_states=True)
                final_h = out.hidden_states[-1][0, -1, :].detach().float()
                collected_final.append(final_h)
                for layer in target_layers:
                    h_l = out.hidden_states[layer + 1][0, -1, :].detach().float()
                    collected_h[layer].append(h_l)

        if not collected_final:
            raise RuntimeError("Failed to collect activations for affine transport estimation.")

        Y = torch.stack(collected_final, dim=0)  # (N, d)
        Y_mean = Y.mean(dim=0, keepdim=True)
        Y_centered = Y - Y_mean

        jacobians: Dict[int, torch.Tensor] = {}
        biases: Dict[int, torch.Tensor] = {}

        N, d = Y.shape
        if N < 2:
            logger.warning("Fewer than 2 prompt samples (%d) collected for affine transport; falling back to identity transport.", N)
            for layer in target_layers:
                jacobians[layer] = torch.eye(d, device=self.device, dtype=self.dtype)
                biases[layer] = torch.zeros(d, device=self.device, dtype=self.dtype)
            return jacobians, biases

        I_d = torch.eye(d, device=self.device)

        for layer in target_layers:
            X = torch.stack(collected_h[layer], dim=0)  # (N, d)
            X_mean = X.mean(dim=0, keepdim=True)
            X_centered = X - X_mean

            # Ridge regression: J = (X^T X + lambda I)^{-1} X^T Y
            # Such that Y_centered ~ X_centered @ J
            cov = torch.matmul(X_centered.T, X_centered) + ridge_lambda * I_d
            cross = torch.matmul(X_centered.T, Y_centered)
            J_transpose = torch.linalg.solve(cov, cross)  # X @ J_transpose ~ Y
            J = J_transpose.T  # (d, d) such that J @ x ~ y
            b = (Y_mean - torch.matmul(X_mean, J_transpose)).squeeze(0)

            jacobians[layer] = J.to(self.device, dtype=self.dtype)
            biases[layer] = b.to(self.device, dtype=self.dtype)

        return jacobians, biases
