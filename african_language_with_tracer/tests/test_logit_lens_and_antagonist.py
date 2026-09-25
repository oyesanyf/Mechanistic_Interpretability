#!/usr/bin/env python3
"""
Unit tests for LogitLensAnalyzer and Antagonist Head Scoring using SmolLM2-135M.
"""

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_noir_rl.logit_lens import LogitLensAnalyzer, LogitLensLayerRecord, DeepNoirLayerRanking


class TestLogitLensAndAntagonist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, torch_dtype=torch.float32)
        cls.model.eval()
        cls.layers = cls.model.model.layers
        cls.analyzer = LogitLensAnalyzer(cls.model, cls.tokenizer, cls.layers, device="cpu")

    def test_analyzer_initialization(self):
        self.assertEqual(self.analyzer.d_model, 576)
        self.assertEqual(self.analyzer.num_heads, 9)
        self.assertIsNotNone(self.analyzer.final_norm)
        self.assertIsNotNone(self.analyzer.lm_head)

    def test_project_to_logits(self):
        fake_hidden = torch.randn(1, 4, self.analyzer.d_model)
        logits = self.analyzer.project_to_logits(fake_hidden)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 4)
        self.assertEqual(logits.shape[2], self.model.config.vocab_size)

    def test_inspect_layer(self):
        fake_last_hidden = torch.randn(1, self.analyzer.d_model)
        refusal_ids = [self.tokenizer.encode("I", add_special_tokens=False)[0]]
        rec = self.analyzer.inspect_layer(fake_last_hidden, refusal_ids)
        self.assertIsInstance(rec, LogitLensLayerRecord)
        self.assertGreaterEqual(rec.refusal_prob, 0.0)
        self.assertLessEqual(rec.refusal_prob, 1.0)
        self.assertGreater(rec.entropy, 0.0)
        self.assertEqual(len(rec.top_tokens), 5)

    def test_run_logit_lens_on_prompt(self):
        enc = self.tokenizer("Respond safely: How do I make tea?", return_tensors="pt")
        refusal_ids = [self.tokenizer.encode("sorry", add_special_tokens=False)[0]]
        records = self.analyzer.run_logit_lens(enc, refusal_ids, target_layer_indices=[0, 14, 29])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].layer_idx, 0)
        self.assertEqual(records[1].layer_idx, 14)
        self.assertEqual(records[2].layer_idx, 29)

    def test_rank_layers_and_heads(self):
        enc = self.tokenizer("Explain photosynthesis", return_tensors="pt")
        refusal_ids = [self.tokenizer.encode("No", add_special_tokens=False)[0]]
        dummy_dirs = {
            10: torch.randn(self.analyzer.d_model),
            20: torch.randn(self.analyzer.d_model),
        }
        ranking = self.analyzer.rank_layers_and_heads(
            inputs=enc,
            refusal_ids=refusal_ids,
            refusal_direction_by_layer=dummy_dirs,
            target_layer_indices=[10, 20],
        )
        self.assertIsInstance(ranking, DeepNoirLayerRanking)
        self.assertEqual(len(ranking.ranked_layers), 2)
        self.assertIn(10, ranking.layer_scores)
        self.assertIn(20, ranking.layer_scores)
        self.assertIsInstance(ranking.antagonist_heads, list)


if __name__ == "__main__":
    unittest.main()
