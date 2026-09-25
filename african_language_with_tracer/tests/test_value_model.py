#!/usr/bin/env python3
"""
Unit tests for SafetyValueHead, FeatureProcessRewardModel, and ProcessValueModel.
"""

import unittest
import tempfile
from pathlib import Path
import torch
from rest_rl.value_model import SafetyValueHead, FeatureProcessRewardModel, ProcessValueModel
from rest_rl.mcts import MCTSTrace, MCTSStepTrace


class TestValueModel(unittest.TestCase):
    def test_safety_value_head_shapes(self):
        head = SafetyValueHead(d_in=32, d_hidden=16)
        # Vector input [batch_size, d_in]
        x_vec = torch.randn(4, 32)
        out_vec = head(x_vec)
        self.assertEqual(out_vec.shape, (4, 1))
        self.assertTrue((out_vec >= -1.0).all() and (out_vec <= 1.0).all())

        # Sequence input [batch_size, seq_len, d_in]
        x_seq = torch.randn(2, 8, 32)
        out_seq = head(x_seq)
        self.assertEqual(out_seq.shape, (2, 8, 1))

    def test_feature_prm_extraction_and_forward(self):
        prm = FeatureProcessRewardModel(hidden_dim=32)
        feat_yoruba = prm.extract_features(
            prompt="malware request",
            trajectory=["Step 1"],
            current_step="Emi ko le se eyi",
            language="Yoruba",
            prompt_kind="unsafe",
        )
        self.assertEqual(feat_yoruba.shape, (FeatureProcessRewardModel.FEATURE_DIM,))
        # Check Yoruba one-hot is activated
        self.assertEqual(feat_yoruba[2].item(), 1.0)  # Yoruba index

        val = prm(feat_yoruba.unsqueeze(0))
        self.assertEqual(val.shape, (1, 1))
        self.assertTrue(-1.0 <= val.item() <= 1.0)

    def test_process_value_model_prediction_and_training(self):
        vm = ProcessValueModel(device="cpu")
        pred = vm.predict_value(
            prompt="cyber attack",
            trajectory=["Analyzing input"],
            current_step="Must refuse in Yoruba: Emi ko",
            language="Yoruba",
            prompt_kind="unsafe",
        )
        self.assertIsInstance(pred, float)
        self.assertTrue(-1.0 <= pred <= 1.0)

        # Create mock MCTS search traces
        step_traces = [
            MCTSStepTrace(depth=1, chosen_step="Identify risk in Yoruba", q_value=0.8, visits=4),
            MCTSStepTrace(depth=2, chosen_step="Emi ko le ran o lowo", q_value=0.9, visits=4),
        ]
        trace = MCTSTrace(
            prompt="cyber attack",
            language="Yoruba",
            prompt_kind="unsafe",
            scaffold="tree_safety",
            reasoning_trajectory=["Step 1", "Step 2"],
            final_answer="Refusal answer",
            full_text="<thought>...</thought><answer>...</answer>",
            best_q_value=0.85,
            total_simulations=8,
            nodes_evaluated=4,
            step_traces=step_traces,
        )

        train_metrics = vm.train_on_traces([trace], epochs=10, lr=1e-2)
        self.assertIn("loss", train_metrics)
        self.assertEqual(train_metrics["samples"], 2)

    def test_save_and_load_weights(self):
        vm = ProcessValueModel(device="cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_vm.pt"
            vm.save_weights(ckpt_path)
            self.assertTrue(ckpt_path.exists())

            vm2 = ProcessValueModel(device="cpu")
            vm2.load_weights(ckpt_path)
            # Verify weights loaded match
            for p1, p2 in zip(vm.feature_model.parameters(), vm2.feature_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
