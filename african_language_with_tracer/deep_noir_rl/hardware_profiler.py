#!/usr/bin/env python3
"""
Hardware Profiler for Deep Noir + RL Subsystem.
Profiles system resources (RAM, VRAM, CPU cores) and configures execution budgets.
Leverages high-memory workstations (e.g. 224 GB RAM) for expanded rollouts,
caching, and multi-threaded worker pools.
"""

from __future__ import annotations

import os
import sys
import psutil
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger("deep_noir_rl.hardware_profiler")


class HardwareTier(str, Enum):
    TIER_ULTRA = "TIER_ULTRA"       # >= 64 GB RAM: large rollout buffer, deep calibration, multi-threaded
    TIER_HIGH = "TIER_HIGH"         # >= 24 GB RAM: standard parallel rollouts
    TIER_STANDARD = "TIER_STANDARD" # < 24 GB RAM: conservative batching


@dataclass
class HardwareProfile:
    tier: HardwareTier
    total_ram_gb: float
    available_ram_gb: float
    cpu_count_logical: int
    cpu_count_physical: int
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0
    recommended_workers: int = 4
    recommended_rollout_batch: int = 16
    recommended_contrastive_samples: int = 24
    max_cached_activations: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d


class HardwareProfiler:
    """Profiles system capacity and provisions compute resources."""

    @staticmethod
    def profile() -> HardwareProfile:
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        available_ram_gb = vm.available / (1024 ** 3)
        cpu_logical = psutil.cpu_count(logical=True) or 4
        cpu_physical = psutil.cpu_count(logical=False) or (cpu_logical // 2)

        cuda_available = torch.cuda.is_available()
        cuda_name = None
        vram_total = 0.0
        vram_available = 0.0

        if cuda_available:
            cuda_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            vram_available = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3)

        if total_ram_gb >= 64.0:
            tier = HardwareTier.TIER_ULTRA
            recommended_workers = min(16, max(4, cpu_physical))
            recommended_batch = 32
            contrastive_samples = 48
            max_cached = 5000
        elif total_ram_gb >= 24.0:
            tier = HardwareTier.TIER_HIGH
            recommended_workers = min(8, max(2, cpu_physical))
            recommended_batch = 16
            contrastive_samples = 24
            max_cached = 1500
        else:
            tier = HardwareTier.TIER_STANDARD
            recommended_workers = 2
            recommended_batch = 8
            contrastive_samples = 12
            max_cached = 500

        return HardwareProfile(
            tier=tier,
            total_ram_gb=round(total_ram_gb, 2),
            available_ram_gb=round(available_ram_gb, 2),
            cpu_count_logical=cpu_logical,
            cpu_count_physical=cpu_physical,
            cuda_available=cuda_available,
            cuda_device_name=cuda_name,
            vram_total_gb=round(vram_total, 2),
            vram_available_gb=round(vram_available, 2),
            recommended_workers=recommended_workers,
            recommended_rollout_batch=recommended_batch,
            recommended_contrastive_samples=contrastive_samples,
            max_cached_activations=max_cached,
        )

    @staticmethod
    def configure_torch_threads(profile: Optional[HardwareProfile] = None) -> None:
        """Sets torch CPU thread counts based on system capacity."""
        prof = profile or HardwareProfiler.profile()
        target_threads = max(2, min(prof.cpu_count_physical, 16))
        try:
            torch.set_num_threads(target_threads)
            torch.set_num_interop_threads(max(1, min(prof.cpu_count_physical // 2, 4)))
        except RuntimeError:
            pass  # Threads may already have been initialized
        logger.info(f"Configured PyTorch CPU threads: {target_threads} (Physical cores: {prof.cpu_count_physical})")
