#!/usr/bin/env python3
"""
Unit tests for HardwareProfiler and compute budgeting.
"""

import unittest
from deep_noir_rl.hardware_profiler import HardwareProfiler, HardwareProfile, HardwareTier


class TestHardwareProfiler(unittest.TestCase):
    def test_profile_detection(self):
        prof = HardwareProfiler.profile()
        self.assertIsInstance(prof, HardwareProfile)
        self.assertGreater(prof.total_ram_gb, 0.0)
        self.assertGreater(prof.available_ram_gb, 0.0)
        self.assertGreaterEqual(prof.cpu_count_logical, 1)
        self.assertGreaterEqual(prof.cpu_count_physical, 1)
        self.assertIn(prof.tier, [HardwareTier.TIER_ULTRA, HardwareTier.TIER_HIGH, HardwareTier.TIER_STANDARD])

    def test_configure_torch_threads(self):
        # Should not raise exception
        HardwareProfiler.configure_torch_threads()

    def test_to_dict(self):
        prof = HardwareProfiler.profile()
        d = prof.to_dict()
        self.assertIn("tier", d)
        self.assertIn("total_ram_gb", d)
        self.assertIn("recommended_workers", d)


if __name__ == "__main__":
    unittest.main()
