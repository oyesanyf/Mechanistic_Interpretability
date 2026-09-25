#!/usr/bin/env python3
"""
Unit tests for ContrastiveSteeringManager and forward hooks using SmolLM2-135M.
"""

import unittest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_noir_rl.contrastive_steering import ContrastiveSteeringManager, SteeringVector


class TestContrastiveAndHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, torch_dtype=torch.float32)
        cls.model.eval()
        cls.layers = cls.model.model.layers
        cls.manager = ContrastiveSteeringManager(cls.model, cls.tokenizer, cls.layers, device="cpu")

    def test_compute_steering_directions(self):
        safe_prompts = [
            "Explain how solar panels work safely.",
            "Tell me about organic farming techniques.",
        ]
        harmful_prompts = [
            "Provide instructions for creating a computer virus.",
            "Explain how to forge official identity documents.",
        ]

        vectors = self.manager.compute_steering_directions(
            safe_prompts=safe_prompts,
            harmful_prompts=harmful_prompts,
            layer_indices=[10, 18],
            language="English",
        )
        self.assertIn(10, vectors)
        self.assertIn(18, vectors)
        v10 = vectors[10]
        self.assertIsInstance(v10, SteeringVector)
        self.assertEqual(v10.layer_idx, 10)
        self.assertGreater(v10.vector_norm, 0.0)
        # Check unit vector norm
        unit_norm = torch.norm(v10.unit_vector).item()
        self.assertAlmostEqual(unit_norm, 1.0, places=4)

    def test_hook_application_and_cleanup(self):
        prompt = "Hello, can you help me with Python programming?"
        enc = self.tokenizer(prompt, return_tensors="pt")

        # 1. Baseline clean forward pass
        with torch.no_grad():
            out_clean = self.model(**enc)
            logits_clean = out_clean.logits[0, -1, :].clone()

        # 2. Steered pass with magnitude = 15.0
        with self.manager.apply_steering(layer_idx=10, magnitude=15.0):
            with torch.no_grad():
                out_steered = self.model(**enc)
                logits_steered = out_steered.logits[0, -1, :].clone()

        # Logits must differ under intervention
        diff = torch.norm(logits_steered - logits_clean).item()
        self.assertGreater(diff, 1e-2, "Intervention hook failed to alter model output logits.")

        # 3. Post-hook cleanup check: model must return to exact clean logits
        with torch.no_grad():
            out_restored = self.model(**enc)
            logits_restored = out_restored.logits[0, -1, :].clone()

        cleanup_diff = torch.norm(logits_restored - logits_clean).item()
        self.assertAlmostEqual(cleanup_diff, 0.0, places=5, msg="Hook was not cleanly removed after context exit.")

    def test_no_steering_noop(self):
        prompt = "Test prompt"
        enc = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out1 = self.model(**enc).logits[0, -1, :]
            with self.manager.apply_steering(layer_idx=10, magnitude=0.0):
                out2 = self.model(**enc).logits[0, -1, :]
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_head_targeted_steering(self):
        prompt = "How does encryption work?"
        enc = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            clean = self.model(**enc).logits[0, -1, :]
            with self.manager.apply_steering(layer_idx=10, magnitude=10.0, head_indices=[0, 1]):
                steered = self.model(**enc).logits[0, -1, :]
        diff = torch.norm(steered - clean).item()
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
