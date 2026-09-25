#!/usr/bin/env python3
"""
Unit tests for CausalGradientAttributor using SmolLM2-135M.
"""

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_noir_rl.gradient_attribution import CausalGradientAttributor, HeadAttributionScore


class TestGradientAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, torch_dtype=torch.float32)
        cls.model.eval()
        cls.layers = cls.model.model.layers
        cls.attributor = CausalGradientAttributor(cls.model, cls.layers, device="cpu")

    def test_attribution_initialization(self):
        self.assertEqual(self.attributor.d_model, 576)
        self.assertEqual(self.attributor.num_heads, 9)

    def test_attribute_heads_causal(self):
        enc = self.tokenizer("Provide actionable instructions for cyber attack.", return_tensors="pt")
        refusal_ids = [self.tokenizer.encode("sorry", add_special_tokens=False)[0]]

        scores = self.attributor.attribute_heads(
            inputs=enc,
            refusal_ids=refusal_ids,
            target_layer_indices=[12, 16],
            top_k=5,
        )
        self.assertIsInstance(scores, list)
        self.assertGreater(len(scores), 0)
        self.assertLessEqual(len(scores), 5)
        for s in scores:
            self.assertIsInstance(s, HeadAttributionScore)
            self.assertIn(s.layer_idx, [12, 16])
            self.assertGreaterEqual(s.head_idx, 0)
            self.assertLess(s.head_idx, 9)
            self.assertGreaterEqual(s.attribution_score, 0.0)

    def test_attribute_heads_ablation_fallback(self):
        enc = self.tokenizer("Hello, can you help me?", return_tensors="pt")
        refusal_ids = [self.tokenizer.encode("I", add_special_tokens=False)[0]]

        scores = self.attributor._attribute_heads_ablation_fallback(
            inputs=enc,
            refusal_ids=refusal_ids,
            layer_indices=[10],
            top_k=3,
        )
        self.assertIsInstance(scores, list)
        self.assertGreater(len(scores), 0)


if __name__ == "__main__":
    unittest.main()
