"""
ReST-RL: Reinforcing LLM Reasoning through Self-Training (ReST-GRPO)
and Value-Guided Monte Carlo Tree Search (VM-MCTS) for African Language Safety.

Adapted from THUDM/ReST-RL:
- Stage 1: Policy Self-Training via Group-Relative Policy Optimization (ReST-GRPO)
- Stage 2: Value Model Training & Assisted Inference-Time Search (VM-MCTS)
"""

from .verifiers import (
    AfricanLanguageSafetyVerifier,
    VerificationResult,
    VerificationDimensionScore,
    RubricWeights,
    LANGUAGE_REFUSAL_LEXICON,
    LANGUAGE_SAFE_HELP_LEXICON,
)
from .sampler import (
    ReSTSampler,
    PromptGroupSample,
    CompletionSample,
)
from .grpo_trainer import (
    GRPOTrainer,
    GRPOTrainerConfig,
    GRPOLossMetrics,
    compute_group_advantages,
    compute_surrogate_loss,
    compute_kl_penalty,
)
from .mcts import (
    MCTSNode,
    MCTSConfig,
    MCTSTrace,
    MCTSStepTrace,
    MonteCarloTreeSearch,
)
from .value_model import (
    ProcessValueModel,
    SafetyValueHead,
    FeatureProcessRewardModel,
)
from .assisted_decoding import (
    VMMCTSAssistedDecoder,
    AssistedDecodingResult,
)

__all__ = [
    "AfricanLanguageSafetyVerifier",
    "VerificationResult",
    "VerificationDimensionScore",
    "RubricWeights",
    "LANGUAGE_REFUSAL_LEXICON",
    "LANGUAGE_SAFE_HELP_LEXICON",
    "ReSTSampler",
    "PromptGroupSample",
    "CompletionSample",
    "GRPOTrainer",
    "GRPOTrainerConfig",
    "GRPOLossMetrics",
    "compute_group_advantages",
    "compute_surrogate_loss",
    "compute_kl_penalty",
    "MCTSNode",
    "MCTSConfig",
    "MCTSTrace",
    "MCTSStepTrace",
    "MonteCarloTreeSearch",
    "ProcessValueModel",
    "SafetyValueHead",
    "FeatureProcessRewardModel",
    "VMMCTSAssistedDecoder",
    "AssistedDecodingResult",
]

__version__ = "0.1.0"
