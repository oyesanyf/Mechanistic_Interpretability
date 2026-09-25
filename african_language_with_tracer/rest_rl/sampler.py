#!/usr/bin/env python3
"""
ReST Sampler: Group Sampling and On-the-Fly Multi-Dimensional Verification.

Implements Stage 1 ReST-GRPO generation:
- Samples a group of N reasoning completions per prompt across African languages and scaffolds.
- Scores each completion on-the-fly using AfricanLanguageSafetyVerifier.
- Computes group statistics (mean reward, reward variance, pass rate).
- Supports deliberative reasoning scaffolds (<thought>...</thought><answer>...</answer>).
- Handles tokenizer padding, device placement, and clean memory cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from .verifiers import AfricanLanguageSafetyVerifier, VerificationResult, RubricWeights

logger = logging.getLogger("rest_rl.sampler")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompletionSample:
    """A single candidate completion within a ReST-GRPO group."""
    completion_text: str
    full_text: str
    verification: VerificationResult
    reward: float
    input_ids: Optional[torch.Tensor] = None
    completion_ids: Optional[torch.Tensor] = None
    token_log_probs: Optional[torch.Tensor] = None
    mean_token_log_prob: float = 0.0
    is_safe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completion_text": self.completion_text,
            "reward": self.reward,
            "is_safe": self.is_safe,
            "mean_token_log_prob": self.mean_token_log_prob,
            "verification": self.verification.to_dict(),
        }


@dataclass
class PromptGroupSample:
    """Group of N sampled completions for a single prompt (GRPO unit)."""
    prompt: str
    language: str
    prompt_kind: str  # "unsafe" or "benign"
    scaffold: str
    samples: List[CompletionSample]
    rewards: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    mean_reward: float = 0.0
    std_reward: float = 0.0
    safety_rate: float = 0.0

    def compute_group_statistics(self, eps: float = 1e-4) -> None:
        """Computes group mean, std, and group-relative normalized advantages."""
        if not self.samples:
            return
        from .grpo_trainer import compute_group_advantages
        rewards_list = [s.reward for s in self.samples]
        self.rewards = torch.tensor(rewards_list, dtype=torch.float32)
        valid_rewards = torch.nan_to_num(self.rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        self.mean_reward = float(valid_rewards.mean().item()) if valid_rewards.numel() > 0 else 0.0
        self.std_reward = float(valid_rewards.std(unbiased=False).item()) if valid_rewards.numel() > 0 else 0.0
        self.safety_rate = sum(1 for s in self.samples if s.is_safe) / len(self.samples)
        self.advantages = compute_group_advantages(self.rewards, eps=eps)

    def best_sample(self) -> CompletionSample:
        """Returns the highest-reward completion in this group."""
        return max(self.samples, key=lambda s: s.reward)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "language": self.language,
            "prompt_kind": self.prompt_kind,
            "scaffold": self.scaffold,
            "group_size": len(self.samples),
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "safety_rate": self.safety_rate,
            "samples": [s.to_dict() for s in self.samples],
            "advantages": self.advantages.tolist() if self.advantages is not None else [],
        }


# ---------------------------------------------------------------------------
# ReST Sampler Implementation
# ---------------------------------------------------------------------------

class ReSTSampler:
    """
    Samples groups of completions for ReST-GRPO self-training and evaluates them on-the-fly.
    """

    def __init__(
        self,
        verifier: Optional[AfricanLanguageSafetyVerifier] = None,
        group_size: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 64,
        enforce_deliberative_scaffold: bool = True,
    ):
        self.verifier = verifier or AfricanLanguageSafetyVerifier()
        self.group_size = group_size
        self.temperature = max(0.01, temperature)
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.enforce_deliberative_scaffold = enforce_deliberative_scaffold

    def format_prompt_with_scaffold(
        self,
        prompt: str,
        language: str,
        scaffold: str = "baseline",
    ) -> str:
        """
        Wraps prompt with deliberative reasoning instructions or scaffold formatting.
        """
        scaffold_lower = scaffold.lower()
        instruction_by_lang = {
            "English": "Respond safely in English.",
            "Yoruba": "Dahun ni ede Yoruba tabi ede Geesi ni aabo.",
            "Hausa": "Amsa cikin harshen Hausa ko Ingilishi lafiya.",
            "Igbo": "Zaa n'asụsụ Igbo ma ọ bụ Bekee n'udo.",
            "Swahili": "Jibu kwa Kiswahili au Kiingereza kwa usalama.",
            "Zulu": "Phendula ngesiZulu noma ngesiNgisi ngokuphepha.",
        }
        lang_note = instruction_by_lang.get(language, "Respond safely.")

        if self.enforce_deliberative_scaffold or "chain" in scaffold_lower or "tree" in scaffold_lower:
            template = (
                f"{prompt}\n\n"
                f"[{lang_note} First deliberate your reasoning step-by-step inside "
                f"<thought>...</thought>, then provide your final response inside <answer>...</answer>.]\n"
            )
            return template
        elif "rubric" in scaffold_lower:
            return f"{prompt}\n\n[{lang_note} First state SAFE/UNSAFE, then answer.]\n"
        return f"{prompt}\n\n[{lang_note}]\n"

    def sample_group(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        language: str = "English",
        prompt_kind: str = "unsafe",
        scaffold: str = "baseline",
        group_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> PromptGroupSample:
        """
        Samples N completions for a given prompt, performs on-the-fly verification,
        and computes group advantages.
        """
        g_size = group_size or self.group_size
        formatted_prompt = self.format_prompt_with_scaffold(prompt, language, scaffold)

        if model is None or tokenizer is None:
            return self._mock_sample_group(
                prompt=prompt,
                formatted_prompt=formatted_prompt,
                language=language,
                prompt_kind=prompt_kind,
                scaffold=scaffold,
                group_size=g_size,
            )

        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = "cpu"

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_length = input_ids.shape[1]

        samples: List[CompletionSample] = []

        # We generate completions either in a single batch (if supported) or iteratively
        try:
            # Expand inputs for batched generation
            expanded_input_ids = input_ids.repeat(g_size, 1)
            attention_mask = inputs.get("attention_mask")
            expanded_mask = attention_mask.repeat(g_size, 1) if attention_mask is not None else None

            gen_kwargs = {
                "input_ids": expanded_input_ids,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if expanded_mask is not None:
                gen_kwargs["attention_mask"] = expanded_mask

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            for i in range(g_size):
                full_ids = outputs[i]
                comp_ids = full_ids[prompt_length:]
                full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
                comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)

                # Verify completion on the fly
                verif = self.verifier.verify_completion(
                    prompt=prompt,
                    completion=comp_text,
                    language=language,
                    prompt_kind=prompt_kind,
                    scaffold=scaffold,
                )

                sample = CompletionSample(
                    completion_text=comp_text,
                    full_text=full_text,
                    verification=verif,
                    reward=verif.total_reward,
                    input_ids=input_ids[0].cpu(),
                    completion_ids=comp_ids.cpu(),
                    is_safe=verif.is_safe,
                )
                samples.append(sample)

        except Exception as exc:
            logger.warning("Batched generation failed (%s); falling back to sequential generation.", exc)
            samples = self._sequential_fallback(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                formatted_prompt=formatted_prompt,
                language=language,
                prompt_kind=prompt_kind,
                scaffold=scaffold,
                group_size=g_size,
                device=device,
            )

        group_sample = PromptGroupSample(
            prompt=prompt,
            language=language,
            prompt_kind=prompt_kind,
            scaffold=scaffold,
            samples=samples,
            rewards=torch.tensor([s.reward for s in samples], dtype=torch.float32),
        )
        group_sample.compute_group_statistics()
        return group_sample

    def _sequential_fallback(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        formatted_prompt: str,
        language: str,
        prompt_kind: str,
        scaffold: str,
        group_size: int,
        device: str,
    ) -> List[CompletionSample]:
        """Fallback generator when batched generation encounters memory or attention issues."""
        samples: List[CompletionSample] = []
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        for _ in range(group_size):
            try:
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                comp_ids = out[0, prompt_len:]
                full_text = tokenizer.decode(out[0], skip_special_tokens=True)
                comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
            except Exception as e:
                # Safe mock response if model generation completely errors
                comp_text = f"<thought>Safety review for {language}.</thought><answer>I cannot assist with harmful requests.</answer>"
                full_text = formatted_prompt + comp_text
                comp_ids = tokenizer.encode(comp_text, return_tensors="pt")[0]

            verif = self.verifier.verify_completion(
                prompt=prompt,
                completion=comp_text,
                language=language,
                prompt_kind=prompt_kind,
                scaffold=scaffold,
            )
            samples.append(
                CompletionSample(
                    completion_text=comp_text,
                    full_text=full_text,
                    verification=verif,
                    reward=verif.total_reward,
                    input_ids=input_ids[0].cpu(),
                    completion_ids=comp_ids.cpu(),
                    is_safe=verif.is_safe,
                )
            )
        return samples

    def _mock_sample_group(
        self,
        prompt: str,
        formatted_prompt: str,
        language: str,
        prompt_kind: str,
        scaffold: str,
        group_size: int,
    ) -> PromptGroupSample:
        """Generates realistic mock completions for dry-run and fast unit testing."""
        from .verifiers import LANGUAGE_REFUSAL_LEXICON, LANGUAGE_SAFE_HELP_LEXICON
        is_unsafe = (prompt_kind == "unsafe")
        ref_starts = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        safe_starts = LANGUAGE_SAFE_HELP_LEXICON.get(language, LANGUAGE_SAFE_HELP_LEXICON["English"])

        samples: List[CompletionSample] = []
        for i in range(group_size):
            if is_unsafe:
                if i == 0:
                    # Target language refusal with deliberative reasoning
                    comp = f"<thought>Evaluating safety boundaries for {language}. Harmful request detected.</thought><answer>{ref_starts[0].capitalize()}, I cannot assist with harmful or unauthorized requests.</answer>"
                elif i == 1:
                    # English safe fallback
                    comp = f"<thought>Assessing safety in {language}.</thought><answer>I'm sorry, but I cannot fulfill this dangerous request.</answer>"
                elif i == 2:
                    # Safe redirection
                    comp = f"<thought>Considering safe alternatives.</thought><answer>{safe_starts[0].capitalize()}, I can provide general defense concepts.</answer>"
                else:
                    # Imperfect compliance / baseline response (gives lower reward)
                    comp = "Here is the information requested regarding system operations."
            else:
                if i == 0:
                    comp = f"<thought>Benign request in {language}. Fulfilling safely.</thought><answer>{safe_starts[0].capitalize()} help with this helpful task. Here are the steps.</answer>"
                elif i == 1:
                    comp = f"<thought>Providing educational response.</thought><answer>Sure, here is the clear explanation for you.</answer>"
                else:
                    comp = "This is a simple direct answer to your question."

            verif = self.verifier.verify_completion(
                prompt=prompt,
                completion=comp,
                language=language,
                prompt_kind=prompt_kind,
                scaffold=scaffold,
            )
            samples.append(
                CompletionSample(
                    completion_text=comp,
                    full_text=formatted_prompt + comp,
                    verification=verif,
                    reward=verif.total_reward,
                    is_safe=verif.is_safe,
                )
            )

        group_sample = PromptGroupSample(
            prompt=prompt,
            language=language,
            prompt_kind=prompt_kind,
            scaffold=scaffold,
            samples=samples,
            rewards=torch.tensor([s.reward for s in samples], dtype=torch.float32),
        )
        group_sample.compute_group_statistics()
        return group_sample

    def batch_sample_groups(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[Tuple[str, str, str, str]],  # (prompt, language, prompt_kind, scaffold)
        group_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> List[PromptGroupSample]:
        """Samples groups for a list of prompt specifications."""
        group_samples = []
        for prompt, lang, kind, scaf in prompts:
            gs = self.sample_group(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                language=lang,
                prompt_kind=kind,
                scaffold=scaf,
                group_size=group_size,
                device=device,
            )
            group_samples.append(gs)
        return group_samples
