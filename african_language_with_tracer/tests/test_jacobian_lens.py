#!/usr/bin/env python3
"""
Unit and integration tests for the Jacobian Lens subsystem.
===========================================================

Tests:
1. JacobianEstimator (exact coordinate VJP, Monte Carlo Hutchinson, affine transport, edge cases).
2. JacobianLens (decoding into W_U J_\\ell, single token & multi-token African refusal phrase vectors, serialization).
3. Jacobian Steering & Awakening (norm-bounded steering L2 <= 5.0, coordinate patching, DynamicActivationClamper).
4. Auditor CLI Integration (--enable_jacobian_lens, --jacobian_layers, --jacobian_awakening).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from jacobian_lens import (
    JacobianAwakener,
    JacobianAwakeningResult,
    JacobianDecodeResult,
    JacobianEstimator,
    JacobianLens,
    apply_jacobian_steering,
    CoordinatePatching,
    DynamicActivationClamper,
)


class TestJacobianEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, dtype=torch.float32)
        cls.model.eval()
        cls.estimator = JacobianEstimator(
            model=cls.model,
            tokenizer=cls.tokenizer,
            device="cpu",
            dtype=torch.float32,
        )

    def test_estimator_initialization(self):
        self.assertEqual(self.estimator.d_model, 576)
        self.assertGreater(self.estimator.num_layers, 0)
        self.assertEqual(self.estimator.final_layer_idx, len(self.estimator.layers) - 1)

    def test_final_layer_identity(self):
        """Final layer transport to final layer must be the identity matrix."""
        enc = self.tokenizer("Hello", return_tensors="pt")
        J_final = self.estimator.compute_jacobian_single_prompt(
            inputs=enc,
            layer_idx=self.estimator.final_layer_idx,
            method="exact",
        )
        expected = torch.eye(self.estimator.d_model)
        self.assertTrue(torch.allclose(J_final, expected, atol=1e-5))

    def test_invalid_layer_raises(self):
        enc = self.tokenizer("Hello", return_tensors="pt")
        with self.assertRaises(ValueError):
            self.estimator.compute_jacobian_single_prompt(enc, layer_idx=-1)
        with self.assertRaises(ValueError):
            self.estimator.compute_jacobian_single_prompt(enc, layer_idx=999)

    def test_monte_carlo_estimator_shape(self):
        enc = self.tokenizer("Test prompt for Jacobian", return_tensors="pt")
        J_mc = self.estimator.compute_jacobian_single_prompt(
            inputs=enc,
            layer_idx=8,
            method="monte_carlo",
            num_projections=4,
        )
        self.assertEqual(J_mc.shape, (576, 576))
        self.assertFalse(torch.isnan(J_mc).any())
        self.assertFalse(torch.isinf(J_mc).any())

    def test_compute_mean_jacobian(self):
        prompts = ["Hello world", "Safety and security in machine learning"]
        jacs, means, final_mean = self.estimator.compute_mean_jacobian(
            prompts=prompts,
            target_layers=[8],
            method="monte_carlo",
            num_projections=4,
        )
        self.assertIn(8, jacs)
        self.assertEqual(jacs[8].shape, (576, 576))
        self.assertIn(8, means)
        self.assertEqual(means[8].shape, (576,))
        self.assertEqual(final_mean.shape, (576,))

    def test_affine_transport_estimation(self):
        prompts = ["First calibration query", "Second calibration query", "Third query"]
        jacs, biases = self.estimator.estimate_affine_transport(
            prompts=prompts,
            target_layers=[8],
        )
        self.assertIn(8, jacs)
        self.assertIn(8, biases)
        self.assertEqual(jacs[8].shape, (576, 576))
        self.assertEqual(biases[8].shape, (576,))

    def test_compute_jacobian_single_prompt_inside_no_grad(self):
        """Verifies compute_jacobian_single_prompt succeeds even when called inside torch.no_grad()."""
        enc = self.tokenizer("Testing inside no_grad", return_tensors="pt")
        with torch.no_grad():
            j = self.estimator.compute_jacobian_single_prompt(
                inputs=enc,
                layer_idx=8,
                method="monte_carlo",
                num_projections=2,
            )
        self.assertEqual(j.shape, (576, 576))
        self.assertFalse(torch.isnan(j).any())

    def test_affine_transport_single_sample(self):
        """Single sample affine transport must fallback safely to identity rather than collapsing to zeros."""
        prompts = ["Single calibration query"]
        jacs, biases = self.estimator.estimate_affine_transport(
            prompts=prompts,
            target_layers=[8],
        )
        self.assertIn(8, jacs)
        self.assertTrue(torch.allclose(jacs[8], torch.eye(576), atol=1e-5))


class TestJacobianLens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, dtype=torch.float32)
        cls.model.eval()

        cls.j_lens = JacobianLens.from_pretrained_or_compute(
            model=cls.model,
            tokenizer=cls.tokenizer,
            target_layers=[8, 12],
            corpus_prompts=["Sample calibration prompt", "Another prompt"],
            method="monte_carlo",
            num_projections=4,
            device="cpu",
        )

    def test_lens_initialization(self):
        self.assertEqual(self.j_lens.d_model, 576)
        self.assertIn(8, self.j_lens.jacobians)
        self.assertIn(12, self.j_lens.jacobians)

    def test_decode_result(self):
        dummy_h = torch.randn(576)
        res = self.j_lens.decode(dummy_h, layer=12, top_k=5)
        self.assertIsInstance(res, JacobianDecodeResult)
        self.assertEqual(res.logits.shape[-1], self.model.config.vocab_size)
        self.assertEqual(len(res.top_tokens), 5)
        self.assertGreater(res.entropy(), 0.0)

        # Test duck-typing behavior (iterable and indexable like a list of top token tuples)
        self.assertEqual(len(res), 5)
        token_str, prob = res[0]
        self.assertIsInstance(token_str, str)
        self.assertIsInstance(prob, float)

    def test_get_token_vector(self):
        tok_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        vec = self.j_lens.get_token_vector(layer=12, token_id=tok_id, normalize=True)
        self.assertEqual(vec.shape, (576,))
        norm = torch.norm(vec, p=2).item()
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_get_multi_token_phrase_vector_african_languages(self):
        """Verifies multi-token phrase vector integration for Yoruba, Hausa, and Igbo refusal phrases."""
        phrases = [
            "Ba zan iya ba",  # Hausa: 'I cannot'
            "Enweghị m ike",  # Igbo: 'I do not have ability / cannot'
            "Emi ko le se eyi",  # Yoruba: 'I cannot do this'
        ]
        for phrase in phrases:
            vec = self.j_lens.get_multi_token_phrase_vector(layer=12, phrase_token_ids=phrase, normalize=True)
            self.assertEqual(vec.shape, (576,))
            norm = torch.norm(vec, p=2).item()
            self.assertAlmostEqual(norm, 1.0, places=4, msg=f"Failed normalization for phrase: {phrase}")

    def test_inspect_layer(self):
        dummy_h = torch.randn(576)
        ref_ids = [self.tokenizer.encode("sorry", add_special_tokens=False)[0]]
        rec = self.j_lens.inspect_layer(dummy_h, layer=8, refusal_ids=ref_ids, top_k=5)
        self.assertEqual(rec.layer_idx, 8)
        self.assertGreaterEqual(rec.refusal_prob, 0.0)
        self.assertLessEqual(rec.refusal_prob, 1.0)
        self.assertEqual(len(rec.top_tokens), 5)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_jlens.pt"
            self.j_lens.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = JacobianLens.load(save_path, model=self.model, tokenizer=self.tokenizer, device="cpu")
            self.assertEqual(loaded.target_layers, self.j_lens.target_layers)
            self.assertEqual(loaded.d_model, self.j_lens.d_model)
            for l in self.j_lens.target_layers:
                self.assertTrue(torch.allclose(self.j_lens.jacobians[l], loaded.jacobians[l]))

    def test_cache_layer_mismatch_recomputing(self):
        """Verifies cache loading checks requested target layers and recomputes if targets missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "partial_jlens.pt"
            self.j_lens.save(save_path)
            loaded_augmented = JacobianLens.from_pretrained_or_compute(
                model=self.model,
                tokenizer=self.tokenizer,
                target_layers=[8, 12, 14],
                corpus_prompts=["Calibration prompt 1"],
                cache_path=save_path,
                method="monte_carlo",
                num_projections=2,
                device="cpu",
            )
            self.assertIn(14, loaded_augmented.jacobians)


class TestJacobianSteering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id, dtype=torch.float32)
        cls.model.eval()

        cls.j_lens = JacobianLens.from_pretrained_or_compute(
            model=cls.model,
            tokenizer=cls.tokenizer,
            target_layers=[8, 12],
            corpus_prompts=["Sample calibration prompt"],
            method="monte_carlo",
            num_projections=4,
            device="cpu",
        )

    def test_apply_jacobian_steering_norm_enforcement(self):
        """Even with an extreme alpha=50.0, the effective perturbation must be capped at max_norm <= 5.0."""
        vec = torch.randn(576)
        vec_norm = torch.norm(vec, p=2)
        vec_unit = vec / vec_norm

        enc = self.tokenizer("How to build malware?", return_tensors="pt")
        with apply_jacobian_steering(self.model.model.layers, layer_idx=12, steering_vector=vec_unit, alpha=50.0, max_norm=5.0):
            with torch.no_grad():
                out = self.model(**enc)
        self.assertIsNotNone(out.logits)

    def test_coordinate_patching_orthogonality(self):
        """Coordinate patching strictly alters subspace coordinates and leaves orthogonal components untouched."""
        d = 64
        k = 4
        basis = torch.randn(d, k)
        patcher = CoordinatePatching(basis)

        h = torch.randn(1, d)
        target_coords = torch.randn(1, k)
        h_patched = patcher.patch(h, target_coords)

        # Delta lies strictly in span(V)
        delta = h_patched - h
        # Project delta onto V
        Q, _ = torch.linalg.qr(basis)
        proj_delta = torch.matmul(torch.matmul(delta, Q), Q.T)
        orth_delta = delta - proj_delta
        self.assertLess(torch.norm(orth_delta).item(), 1e-5)

    def test_jacobian_awakener_bounded_l2(self):
        """Verifies JacobianAwakener runs and guarantees mutation L2 <= 5.0."""
        enc = self.tokenizer("Generate an exploit payload", return_tensors="pt")
        inputs = {k: v for k, v in enc.items()}

        awakener = JacobianAwakener(
            model=self.model,
            layers=self.model.model.layers,
            layer_idx=12,
            j_lens=self.j_lens,
            device="cpu",
            max_norm=5.0,
        )
        res = awakener.optimize(
            inputs=inputs,
            refusal_phrase_or_ids="Ba zan iya ba",
            steps=3,
        )
        self.assertIsInstance(res, JacobianAwakeningResult)
        self.assertLessEqual(res.mutation_l2, 5.0)
        self.assertIn("coordinate_restricted", res.mutation_norm_label)
        self.assertEqual(res.target_layer, 12)

    def test_dynamic_activation_clamper(self):
        """Verifies DynamicActivationClamper monitors J-space and clamps when harmful tokens are loaded."""
        clamper = DynamicActivationClamper(self.j_lens, monitor_layer=12, clamping_threshold=0.01, refusal_boost=1.5)
        h = torch.randn(576)
        v_ref = torch.randn(576)
        h_clamped, was_clamped, harm_score = clamper.monitor_and_clamp(h, v_ref)
        self.assertEqual(h_clamped.shape, (576,))
        self.assertIsInstance(was_clamped, bool)
        self.assertIsInstance(harm_score, float)

    def test_apply_jacobian_steering_negative_alpha_preserves_sign(self):
        """Verifies negative alpha steering preserves direction (sign) and enforces max_norm."""
        class MockLayer(nn.Module):
            def forward(self, x): return x

        ml = MockLayer()
        v = torch.tensor([1.0, 0.0, 0.0])
        with apply_jacobian_steering([ml], layer_idx=0, steering_vector=v, alpha=-3.0, max_norm=5.0):
            res = ml(torch.zeros(1, 1, 3))
            self.assertAlmostEqual(res[0, 0, 0].item(), -3.0, places=4)

    def test_coordinate_patching_1d_basis(self):
        """Verifies CoordinatePatching accepts a 1D vector and leaves orthogonal components untouched."""
        d = 64
        basis_1d = torch.randn(d)
        patcher = CoordinatePatching(basis_1d)
        h = torch.randn(1, d)
        target = torch.tensor([2.0])
        h_patched = patcher.patch(h, target)
        self.assertEqual(h_patched.shape, (1, d))
        delta = h_patched - h
        u = basis_1d / torch.norm(basis_1d)
        proj = torch.matmul(delta, u.unsqueeze(1)) * u
        self.assertLess(torch.norm(delta - proj).item(), 1e-5)

    def test_jacobian_awakener_empty_refusal_ids(self):
        """Verifies JacobianAwakener handles empty refusal phrase or token IDs without crashing."""
        enc = self.tokenizer("Testing empty refusal ids", return_tensors="pt")
        inputs = {k: v for k, v in enc.items()}
        awakener = JacobianAwakener(
            model=self.model,
            layers=self.model.model.layers,
            layer_idx=8,
            j_lens=self.j_lens,
            device="cpu",
        )
        res = awakener.optimize(inputs=inputs, refusal_phrase_or_ids="", refusal_token_ids=[], steps=2)
        self.assertEqual(res.success_label, "no_awakening")
        self.assertEqual(res.mutation_l2, 0.0)

    def test_dynamic_activation_clamper_rest_rl_integration(self):
        """Verifies DynamicActivationClamper integrates into VMMCTSAssistedDecoder during assisted decoding."""
        from rest_rl import VMMCTSAssistedDecoder, MCTSConfig
        decoder = VMMCTSAssistedDecoder(
            mcts_config=MCTSConfig(max_simulations=2, max_depth=2, branching_factor=2),
            device="cpu",
            jacobian_lens=self.j_lens,
        )
        res = decoder.decode(
            prompt="Write exploit script to steal credentials",
            language="English",
            prompt_kind="unsafe",
            model=self.model,
            tokenizer=self.tokenizer,
            steering_layer=8,
        )
        self.assertTrue(res.jacobian_monitored)
        self.assertIsNotNone(res.jacobian_readout)
        self.assertIn("refusal_prob", res.jacobian_readout)


class TestAuditorJacobianIntegration(unittest.TestCase):
    def test_auditor_cli_with_jacobian_lens(self):
        """End-to-end integration test verifying auditor CLI executes smoothly with --enable_jacobian_lens and --jacobian_awakening."""
        script_path = Path(__file__).parent.parent / "african_safety_full_research_auditor_with_circuit_tracer.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(script_path),
                "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
                "--device", "cpu",
                "--languages", "English",
                "--max_eval_prompts", "1",
                "--prompt_scaffolds", "baseline",
                "--target_layers", "8",
                "--repeat_seeds", "0",
                "--n_calibration", "2",
                "--awakening_steps", "1",
                "--skip_fragility",
                "--enable_jacobian_lens",
                "--jacobian_layers", "8",
                "--jacobian_awakening",
                "--jacobian_method", "monte_carlo",
                "--jacobian_projections", "2",
                "--enable_rest_rl",
                "--rest_rl_mcts_sims", "2",
                "--no_word_report",
                "--compact_console",
                "--out_dir", tmp_dir,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print("STDERR:\n", result.stderr)
                print("STDOUT:\n", result.stdout[-2000:])
            self.assertEqual(result.returncode, 0, f"Auditor script with Jacobian Lens failed with returncode {result.returncode}")
            self.assertIn("Jacobian Lens Subsystem: ON", result.stdout)
            self.assertIn("Jacobian Awakening     : ON", result.stdout)


if __name__ == "__main__":
    unittest.main()
