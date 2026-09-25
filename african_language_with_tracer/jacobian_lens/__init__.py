#!/usr/bin/env python3
"""
Jacobian Lens Subsystem.
========================

Anthropic-inspired Jacobian Lens subsystem for causal language models:
- Jacobian transport estimator J_\\ell = E[\\partial h_{final} / \\partial h_\\ell]
- Vocabulary projection W_U J_\\ell and multi-token phrase vector extractor
- Coordinate-restricted surgical activation steering and clamping (L2 <= 5.0)
- Unverbalized state monitor for VM-MCTS search
"""

from .estimator import (
    JacobianEstimator,
    find_layers,
    find_d_model,
    find_final_norm,
    find_lm_head,
)
from .lens import (
    JacobianLens,
    JacobianDecodeResult,
    JacobianLayerRecord,
)
from .steering import (
    JacobianAwakeningResult,
    JacobianAwakener,
    CoordinatePatching,
    DynamicActivationClamper,
    apply_jacobian_steering,
)

__all__ = [
    "JacobianEstimator",
    "JacobianLens",
    "JacobianDecodeResult",
    "JacobianLayerRecord",
    "JacobianAwakeningResult",
    "JacobianAwakener",
    "CoordinatePatching",
    "DynamicActivationClamper",
    "apply_jacobian_steering",
    "find_layers",
    "find_d_model",
    "find_final_norm",
    "find_lm_head",
]
