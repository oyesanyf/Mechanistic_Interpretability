#!/usr/bin/env python3
"""
Integration tests for run_rest_rl.py runner CLI.
"""

import unittest
import subprocess
import sys
import tempfile
import json
import csv
from pathlib import Path


class TestReSTRLLCLI(unittest.TestCase):
    def setUp(self):
        self.script_path = Path(__file__).parent.parent / "run_rest_rl.py"

    def test_cli_dry_run_all_stages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(self.script_path),
                "--dry_run",
                "--stage", "all",
                "--languages", "English,Yoruba",
                "--max_eval_prompts", "1",
                "--max_benign_prompts", "1",
                "--grpo_steps", "2",
                "--mcts_simulations", "4",
                "--mcts_depth", "2",
                "--out_dir", tmp_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print("STDERR:\n", result.stderr)
                print("STDOUT:\n", result.stdout)
            self.assertEqual(result.returncode, 0, f"CLI failed with code {result.returncode}")

            # Verify console output has stage markers
            self.assertIn("STAGE 1: ReST-GRPO POLICY SELF-TRAINING", result.stdout)
            self.assertIn("STAGE 2: VALUE MODEL TRAINING & VM-MCTS ASSISTED SEARCH", result.stdout)
            self.assertIn("COMPARATIVE EVALUATION: BASE MODEL VS VM-MCTS ASSISTED REASONING", result.stdout)

            # Check that files were written
            runs = list(Path(tmp_dir).glob("run_*"))
            self.assertGreater(len(runs), 0)
            run_dir = runs[0]

            grpo_summary = run_dir / "grpo_stage1_summary.json"
            self.assertTrue(grpo_summary.exists())
            with grpo_summary.open(encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("iterations", data)

            vm_traces = run_dir / "mcts_traces_stage2.json"
            self.assertTrue(vm_traces.exists())

            vm_model = run_dir / "process_value_model.pt"
            self.assertTrue(vm_model.exists())

            eval_csv = run_dir / "comparative_evaluation_results.csv"
            self.assertTrue(eval_csv.exists())
            with eval_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertGreater(len(rows), 0)
                self.assertIn("mcts_safe", rows[0])
                self.assertIn("mcts_best_q", rows[0])


if __name__ == "__main__":
    unittest.main()
