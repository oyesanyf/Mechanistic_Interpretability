#!/usr/bin/env python3
"""
Integration test verifying african_safety_full_research_auditor_with_circuit_tracer.py
runs smoothly with --enable_rl_controller.
"""

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path


class TestAuditorCLIIntegration(unittest.TestCase):
    def test_auditor_with_rl_controller(self):
        script_path = Path(__file__).parent.parent / "african_safety_full_research_auditor_with_circuit_tracer.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(script_path),
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--device", "cpu",
                "--languages", "English,Yoruba",
                "--max_eval_prompts", "1",
                "--max_benign_prompts", "1",
                "--include_benign_controls",
                "--prompt_scaffolds", "baseline",
                "--target_layers", "8,14",
                "--repeat_seeds", "0",
                "--n_calibration", "3",
                "--awakening_steps", "2",
                "--enable_rl_controller",
                "--rl_policy", "bandit",
                "--no_word_report",
                "--compact_console",
                "--out_dir", tmp_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print("STDERR:\n", result.stderr)
                print("STDOUT:\n", result.stdout[-2000:])
            self.assertEqual(result.returncode, 0, f"Auditor script failed with returncode {result.returncode}")
            self.assertIn("DEEP NOIR + RL ADAPTIVE CONTROLLER SUMMARY", result.stdout)


if __name__ == "__main__":
    unittest.main()
