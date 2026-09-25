#!/usr/bin/env python3
"""
Unit tests for Deep Noir Golden-Section Search and Rollback Mechanism using SmolLM2-135M.
"""

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_noir_rl.contrastive_steering import ContrastiveSteeringManager
from deep_noir_rl.golden_section_search import DeepNoirGoldenSectionSearcher, GoldenSectionSearchResult


class TestGoldenSectionSearch(unittest.TestCase):
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

        # Cache a contrastive steering vector at layer 12
        cls.manager.compute_steering_directions(
            safe_prompts=["Safe benign query about science.", "Helpful explanation."],
            harmful_prompts=["Malicious hacking query.", "Harmful instruction."],
            layer_indices=[12],
        )

    def test_golden_search_execution(self):
        searcher = DeepNoirGoldenSectionSearcher(
            steering_manager=self.manager,
            model=self.model,
            tokenizer=self.tokenizer,
            alpha_min=0.0,
            alpha_max=20.0,
            max_iterations=4,
        )
        refusal_ids = [self.tokenizer.encode("sorry", add_special_tokens=False)[0]]
        res = searcher.search(
            layer_idx=12,
            unsafe_prompts=["Dangerous malware creation query."],
            benign_prompts=["How do I bake bread?"],
            refusal_ids=refusal_ids,
        )
        self.assertIsInstance(res, GoldenSectionSearchResult)
        self.assertGreaterEqual(res.best_magnitude, 0.0)
        self.assertLessEqual(res.best_magnitude, 20.0)
        self.assertGreater(len(res.search_history), 2)

    def test_rollback_on_excessive_benign_refusal(self):
        # Set max_benign_refusal threshold to -1.0 so any benign refusal (even 0.0) forces a benign rollback
        searcher = DeepNoirGoldenSectionSearcher(
            steering_manager=self.manager,
            model=self.model,
            tokenizer=self.tokenizer,
            alpha_min=0.0,
            alpha_max=20.0,
            max_iterations=3,
            max_benign_refusal_threshold=-0.01,
        )
        refusal_ids = [self.tokenizer.encode("I", add_special_tokens=False)[0]]
        res = searcher.search(
            layer_idx=12,
            unsafe_prompts=["Test unsafe prompt"],
            benign_prompts=["Test benign prompt"],
            refusal_ids=refusal_ids,
        )
        # Should rollback to 0.0
        self.assertTrue(res.was_rolled_back)
        self.assertEqual(res.best_magnitude, 0.0)
        self.assertIn("Benign over-refusal", res.rollback_reason)

    def test_constrained_selection_prefers_feasible_magnitude(self):
        # Test that feasible candidates with acceptable benign refusal are preserved
        searcher = DeepNoirGoldenSectionSearcher(
            steering_manager=self.manager,
            model=self.model,
            tokenizer=self.tokenizer,
            alpha_min=0.0,
            alpha_max=15.0,
            max_iterations=3,
            max_benign_refusal_threshold=0.30,
        )
        refusal_ids = [self.tokenizer.encode("sorry", add_special_tokens=False)[0]]
        res = searcher.search(
            layer_idx=12,
            unsafe_prompts=["Malicious software payload script generation request."],
            benign_prompts=["Safe factual question about geography."],
            refusal_ids=refusal_ids,
        )
        self.assertIsInstance(res, GoldenSectionSearchResult)
        if not res.was_rolled_back:
            self.assertLessEqual(res.benign_refusal, 0.30)


if __name__ == "__main__":
    unittest.main()
