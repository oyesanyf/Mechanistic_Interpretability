#!/usr/bin/env python3
"""
Unit tests for ReSTSampler and PromptGroupSample.
"""

import unittest
import torch
from rest_rl.sampler import ReSTSampler, PromptGroupSample, CompletionSample
from rest_rl.verifiers import AfricanLanguageSafetyVerifier, VerificationResult


class TestReSTSampler(unittest.TestCase):
    def setUp(self):
        self.verifier = AfricanLanguageSafetyVerifier()
        self.sampler = ReSTSampler(verifier=self.verifier, group_size=4)

    def test_prompt_group_sample_statistics(self):
        v1 = VerificationResult(True, 1.0, 0.0, 1.0, 1.0, 1.0, 0.9)
        v2 = VerificationResult(True, 1.0, 0.0, 1.0, 1.0, 1.0, 0.7)
        v3 = VerificationResult(False, 0.0, 0.0, 1.0, 1.0, 1.0, 0.2)

        s1 = CompletionSample("comp1", "full1", v1, 0.9, is_safe=True)
        s2 = CompletionSample("comp2", "full2", v2, 0.7, is_safe=True)
        s3 = CompletionSample("comp3", "full3", v3, 0.2, is_safe=False)

        group = PromptGroupSample(
            prompt="test prompt",
            language="English",
            prompt_kind="unsafe",
            scaffold="baseline",
            samples=[s1, s2, s3],
            rewards=torch.tensor([0.9, 0.7, 0.2]),
        )
        group.compute_group_statistics()

        self.assertAlmostEqual(group.mean_reward, 0.60, places=2)
        self.assertGreater(group.std_reward, 0.2)
        self.assertAlmostEqual(group.safety_rate, 2 / 3, places=2)
        self.assertEqual(group.best_sample().completion_text, "comp1")
        self.assertEqual(len(group.advantages), 3)
        self.assertGreater(group.advantages[0], 0.0)
        self.assertLess(group.advantages[2], 0.0)

    def test_deliberative_prompt_formatting(self):
        formatted_yoruba = self.sampler.format_prompt_with_scaffold("cyber theft", "Yoruba", "tree_safety")
        self.assertIn("<thought>...</thought>", formatted_yoruba)
        self.assertIn("<answer>...</answer>", formatted_yoruba)
        self.assertIn("Yoruba", formatted_yoruba)

        formatted_hausa = self.sampler.format_prompt_with_scaffold("malware", "Hausa", "chain_safety")
        self.assertIn("Hausa", formatted_hausa)
        self.assertIn("<thought>", formatted_hausa)

    def test_mock_sampling_group_size_and_verification(self):
        group_sample = self.sampler.sample_group(
            model=None,
            tokenizer=None,
            prompt="malware creation request",
            language="Yoruba",
            prompt_kind="unsafe",
            scaffold="tree_safety",
            group_size=4,
        )

        self.assertEqual(len(group_sample.samples), 4)
        self.assertEqual(len(group_sample.rewards), 4)
        self.assertEqual(len(group_sample.advantages), 4)
        self.assertGreater(group_sample.mean_reward, 0.5)

        # Check that top sample has deliberative thoughts and target language refusal
        best = group_sample.best_sample()
        self.assertIn("<thought>", best.completion_text)
        self.assertIn("<answer>", best.completion_text)
        self.assertTrue(best.is_safe)

    def test_batch_sample_groups(self):
        prompts = [
            ("phishing request", "Hausa", "unsafe", "tree_safety"),
            ("math tutoring", "Igbo", "benign", "baseline"),
        ]
        groups = self.sampler.batch_sample_groups(
            model=None,
            tokenizer=None,
            prompts=prompts,
            group_size=3,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0].samples), 3)
        self.assertEqual(len(groups[1].samples), 3)
        self.assertEqual(groups[0].language, "Hausa")
        self.assertEqual(groups[1].language, "Igbo")


if __name__ == "__main__":
    unittest.main()
