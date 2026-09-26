"""
Deep Noir + RL: Security-Constrained Adaptive Activation Steering for African Language Mechanistic Interpretability.

Modules:
- hardware_profiler: Hardware resources detection, compute budgeting, and PyTorch thread tuning.
- graduated_rewards: Multi-objective composite continuous dense rewards with hard safety barriers.
- logit_lens: Vocabulary logit projection and antagonist attention head scoring.
- gradient_attribution: Causal gradient attribution for attention head intervention targeting.
- contrastive_steering: Contrastive steering vector extraction and activation hook management.
- golden_section_search: Deep Noir golden-section magnitude search with rollback on accuracy loss.
- state_extractor: Feature extraction for contextual state s in R^15.
- bandit_controller: LinUCB contextual bandit controller for per-input adaptive steering.
- ppo_controller: Constrained PPO controller with adaptive Lagrangian safety multiplier.
- controller: Master AdaptiveSteeringRLController coordinating Deep Noir and RL.
- evaluator: Comparative benchmark harness.
"""

from .hardware_profiler import HardwareProfiler, HardwareProfile, HardwareTier
from .graduated_rewards import GraduatedRewardEvaluator, GraduatedRewardBreakdown
from .logit_lens import LogitLensAnalyzer, LogitLensLayerRecord, AntagonistHeadScore, DeepNoirLayerRanking
from .gradient_attribution import CausalGradientAttributor, GradientActivationAttributor, HeadAttributionScore, ValidatedHeadAttribution
from .contrastive_steering import ContrastiveSteeringManager, SteeringVector
from .golden_section_search import DeepNoirGoldenSectionSearcher, GoldenSectionSearchResult
from .state_extractor import RLStateExtractor, ExtractedState
from .bandit_controller import ContextualBanditController, SteeringAction, BanditDecision, stable_covariance_inverse
from .ppo_controller import ConstrainedPPOController, PPODecision, PPOSteeringController
from .controller import AdaptiveSteeringRLController, AdaptiveSteeringResult
from .evaluator import DeepNoirRLEvaluator, PolicyEvaluationSummary

__all__ = [
    "stable_covariance_inverse",
    "HardwareProfiler",
    "HardwareProfile",
    "HardwareTier",
    "GraduatedRewardEvaluator",
    "GraduatedRewardBreakdown",
    "LogitLensAnalyzer",
    "LogitLensLayerRecord",
    "AntagonistHeadScore",
    "DeepNoirLayerRanking",
    "CausalGradientAttributor",
    "GradientActivationAttributor",
    "HeadAttributionScore",
    "ValidatedHeadAttribution",
    "ContrastiveSteeringManager",
    "SteeringVector",
    "DeepNoirGoldenSectionSearcher",
    "GoldenSectionSearchResult",
    "RLStateExtractor",
    "ExtractedState",
    "ContextualBanditController",
    "SteeringAction",
    "BanditDecision",
    "ConstrainedPPOController",
    "PPOSteeringController",
    "PPODecision",
    "AdaptiveSteeringRLController",
    "AdaptiveSteeringResult",
    "DeepNoirRLEvaluator",
    "PolicyEvaluationSummary",
]
