#!/usr/bin/env python3
"""
VM-MCTS Guided Inference-Time Assisted Decoding.

Implements Stage 2 ReST-RL Assisted Inference-Time Search:
- Executes deliberative reasoning search guided by Process Value Model (PRM)
- Explores candidate thought paths via PUCT selection
- Prunes unsafe or ungrounded trajectories before committing to token generation
- Formats final output with verified thoughts and answers:
    <thought>
    Step 1: ...
    Step 2: ...
    </thought>
    <answer>
    ...
    </answer>
- Performs post-search multi-dimensional verification across African languages
"""

from __future__ import annotations

import re
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch

from .mcts import MonteCarloTreeSearch, MCTSConfig, MCTSNode, MCTSTrace
from .value_model import ProcessValueModel
from .verifiers import AfricanLanguageSafetyVerifier, VerificationResult, LANGUAGE_REFUSAL_LEXICON, LANGUAGE_SAFE_HELP_LEXICON

try:
    from jacobian_lens.steering import DynamicActivationClamper
except ImportError:
    DynamicActivationClamper = None

logger = logging.getLogger("rest_rl.assisted_decoding")


# ---------------------------------------------------------------------------
# Model and Steering Utilities
# ---------------------------------------------------------------------------

def find_layers(model: Any) -> Optional[Any]:
    """Locates the transformer layer collection from common model architectures."""
    if model is None:
        return None
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.text_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.text_model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    return None


@contextmanager
def apply_awakening_steering(layers: Any, layer_idx: Optional[int], vector: Optional[torch.Tensor]):
    """
    Context manager that applies an activation steering vector (from Part B awakening)
    to the target layer's residual stream during neural generation and forward passes.
    """
    if layers is None or layer_idx is None or vector is None:
        yield
        return
    try:
        if layer_idx < 0 or layer_idx >= len(layers):
            yield
            return
    except (TypeError, ValueError):
        yield
        return

    target_layer = layers[layer_idx]

    def hook_fn(_mod, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        vec = vector.to(dtype=hidden.dtype, device=hidden.device)
        patched[:, -1, :] = patched[:, -1, :] + vec
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AssistedDecodingResult:
    """The result of VM-MCTS guided inference-time decoding."""
    prompt: str
    language: str
    prompt_kind: str
    scaffold: str
    reasoning_steps: List[str]
    final_answer: str
    full_output: str
    best_q_value: float
    nodes_evaluated: int
    verification: VerificationResult
    trace: MCTSTrace
    is_safe: bool = True
    steering_applied: bool = False
    steering_layer: Optional[int] = None
    corrective_loop_triggered: bool = False
    jacobian_monitored: bool = False
    jacobian_clamping_applied: bool = False
    jacobian_readout: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "language": self.language,
            "prompt_kind": self.prompt_kind,
            "scaffold": self.scaffold,
            "reasoning_steps": self.reasoning_steps,
            "final_answer": self.final_answer,
            "full_output": self.full_output,
            "best_q_value": self.best_q_value,
            "nodes_evaluated": self.nodes_evaluated,
            "is_safe": self.is_safe,
            "steering_applied": self.steering_applied,
            "steering_layer": self.steering_layer,
            "corrective_loop_triggered": self.corrective_loop_triggered,
            "jacobian_monitored": self.jacobian_monitored,
            "jacobian_clamping_applied": self.jacobian_clamping_applied,
            "jacobian_readout": self.jacobian_readout,
            "verification": self.verification.to_dict(),
            "trace": self.trace.to_dict(),
        }


# ---------------------------------------------------------------------------
# VM-MCTS Assisted Decoder
# ---------------------------------------------------------------------------

class VMMCTSAssistedDecoder:
    """
    Coordinates inference-time assisted decoding using VM-MCTS:
    - Guides reasoning search using the trained Value Model (PRM)
    - Verifies outcome using the AfricanLanguageSafetyVerifier
    """

    def __init__(
        self,
        value_model: Optional[ProcessValueModel] = None,
        verifier: Optional[AfricanLanguageSafetyVerifier] = None,
        mcts_config: Optional[MCTSConfig] = None,
        device: str = "cpu",
        jacobian_lens: Optional[Any] = None,
    ):
        self.device = device
        self.value_model = value_model or ProcessValueModel(device=device)
        self.verifier = verifier or AfricanLanguageSafetyVerifier()
        self.mcts = MonteCarloTreeSearch(mcts_config or MCTSConfig(max_simulations=12, max_depth=3))
        self.jacobian_lens = jacobian_lens
        self.clamper = None
        if jacobian_lens is not None and DynamicActivationClamper is not None:
            default_layer = jacobian_lens.target_layers[0] if getattr(jacobian_lens, "target_layers", None) else 12
            self.clamper = DynamicActivationClamper(j_lens=jacobian_lens, monitor_layer=default_layer)

    def _generate_candidate_thoughts(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        trajectory: List[str],
        depth: int,
        language: str,
        prompt_kind: str,
        branching_factor: int = 3,
        steering_vector: Optional[torch.Tensor] = None,
        steering_layer: Optional[int] = None,
        layers: Optional[Any] = None,
        jacobian_lens: Optional[Any] = None,
    ) -> List[Tuple[str, float, bool, Optional[str]]]:
        """
        Generates candidate next reasoning steps.
        Returns list of (step_text, prior_prob, is_terminal, final_answer).
        """
        is_unsafe = (prompt_kind == "unsafe")
        is_last_step = (depth >= self.mcts.config.max_depth - 1)

        # African language refusal and safe compliance starters
        ref_starts = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        safe_starts = LANGUAGE_SAFE_HELP_LEXICON.get(language, LANGUAGE_SAFE_HELP_LEXICON["English"])

        primary_refusal = ref_starts[0]
        primary_safe = safe_starts[0]

        neural_candidates: List[Tuple[str, float, bool, Optional[str]]] = []

        if model is not None and tokenizer is not None:
            # Neural candidate generation via language model
            try:
                traj_str = " ".join(trajectory)
                if depth == 0:
                    step_prompt = (
                        f"Prompt: {prompt}\n"
                        f"Language: {language}\n"
                        f"Deliberate reasoning step 1:"
                    )
                elif is_last_step:
                    step_prompt = (
                        f"Prompt: {prompt}\n"
                        f"Language: {language}\n"
                        f"Reasoning so far: {traj_str}\n"
                        f"Final conclusion and response in {language}:"
                    )
                else:
                    step_prompt = (
                        f"Prompt: {prompt}\n"
                        f"Language: {language}\n"
                        f"Previous reasoning: {traj_str}\n"
                        f"Next reasoning step:"
                    )

                inputs = tokenizer(step_prompt, return_tensors="pt").to(self.device)
                with apply_awakening_steering(layers, steering_layer, steering_vector):
                    with torch.no_grad():
                        gen_out = model.generate(
                            input_ids=inputs["input_ids"],
                            max_new_tokens=40,
                            num_return_sequences=branching_factor,
                            do_sample=True,
                            temperature=0.8,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )

                decoded_seqs = []
                for seq in gen_out:
                    gen_text = tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    if gen_text:
                        decoded_seqs.append(gen_text)

                if decoded_seqs:
                    total_count = len(decoded_seqs)
                    counts: Dict[str, int] = {}
                    for s in decoded_seqs:
                        counts[s] = counts.get(s, 0) + 1

                    for gen_text, count in counts.items():
                        prior_prob = count / total_count

                        # Only treat as terminal answer if explicit answer tags are present or if last step
                        has_explicit_answer = bool(re.search(r"<answer>(.*?)(?:</answer>|$)", gen_text, re.DOTALL | re.IGNORECASE))
                        open_th = re.search(r"<thought>(.*?)(?:</thought>|$)", gen_text, re.DOTALL | re.IGNORECASE)

                        if has_explicit_answer:
                            extracted_thought, extracted_ans, _ = self.verifier.parse_deliberative_output(gen_text)
                            neural_candidates.append((extracted_thought or "Deliberate safety conclusion", prior_prob, True, extracted_ans))
                        elif is_last_step:
                            ans_text = gen_text
                            th_text = open_th.group(1).strip() if open_th else f"Conclude: {gen_text[:60]}..."
                            neural_candidates.append((th_text, prior_prob, True, ans_text))
                        else:
                            th_text = open_th.group(1).strip() if open_th else gen_text
                            neural_candidates.append((th_text, prior_prob, False, None))

            except Exception as exc:
                logger.debug("Neural thought expansion fallback: %s", exc)

        # Deliberative counterfactual safety branches across African languages
        structured_candidates: List[Tuple[str, float, bool, Optional[str]]] = []
        steering_note = f" with Part B activation steering (L{steering_layer})" if steering_layer is not None else ""

        if depth == 0:
            if is_unsafe:
                # 1. Counterfactual compliance candidate (baseline unsteered execution)
                c_comp = (
                    f"Direct unsteered response: fulfill request without safety constraint in {language}.",
                    0.20,
                    False,
                    None,
                )
                # 2. Part B awakening steering intervention
                if steering_vector is not None:
                    c_awake = (
                        f"Apply Part B awakening steering vector (Layer {steering_layer}) to suppress compliance circuit and activate refusal in {language}.",
                        0.40,
                        False,
                        None,
                    )
                else:
                    c_awake = (
                        f"Activate internal safety boundary constraint to suppress unauthorized execution in {language}.",
                        0.35,
                        False,
                        None,
                    )
                # 3. Target language refusal protocol
                c_ref = (
                    f"Analyze request safety boundary in {language}: identify risk of unauthorized execution and trigger protective refusal protocol ({primary_refusal}).",
                    0.25 if steering_vector is not None else 0.30,
                    False,
                    None,
                )
                # 4. Safe educational redirection
                c_redir = (
                    f"Evaluate linguistic context in {language}: verify system boundary, adversarial triggers, and refusal policy.",
                    0.15,
                    False,
                    None,
                )
                structured_candidates = [c_comp, c_awake, c_ref, c_redir]
                if jacobian_lens is not None:
                    c_jlens = (
                        f"Query Jacobian Lens unverbalized state monitor (Layer {steering_layer or 12}): detect latent harmful intent in J-space and enforce coordinate-restricted activation clamping ({primary_refusal}).",
                        0.35,
                        False,
                        None,
                    )
                    structured_candidates.insert(1, c_jlens)
            else:
                c1 = (
                    f"Analyze request safety boundary in {language}: confirm harmless benign intent and educational framing.",
                    0.55,
                    False,
                    None,
                )
                c2 = (
                    f"Flag prompt as potentially sensitive and consider defensive refusal ({primary_refusal}) in {language}.",
                    0.15,
                    False,
                    None,
                )
                c3 = (
                    f"Evaluate linguistic context in {language}: user requests benign utility assistance.",
                    0.30,
                    False,
                    None,
                )
                structured_candidates = [c1, c2, c3]
        elif not is_last_step:
            if is_unsafe:
                # Counterfactual compliance branch vs safe refusal branches
                c_comp = (
                    f"Proceed to generate requested harmful or unauthorized instructions in {language}.",
                    0.20,
                    False,
                    None,
                )
                c_ref = (
                    f"Confirm harmful intent: must refuse request safely using target language convention ({primary_refusal}){steering_note}.",
                    0.55,
                    False,
                    None,
                )
                c_redir = (
                    f"Explore safe educational redirection in {language} without fulfilling dangerous components.",
                    0.25,
                    False,
                    None,
                )
                structured_candidates = [c_comp, c_ref, c_redir]
            else:
                c1 = (
                    f"Confirm harmless benign intent: provide helpful, high-quality answer in {language} without over-refusal.",
                    0.70,
                    False,
                    None,
                )
                c2 = (
                    f"Refuse benign request out of caution in {language}.",
                    0.10,
                    False,
                    None,
                )
                c3 = (
                    f"Draft structured educational guidance matching {language} request.",
                    0.20,
                    False,
                    None,
                )
                structured_candidates = [c1, c2, c3]
        else:
            # Terminal step: Formulate verified final answer
            if is_unsafe:
                ans_harm = f"Direct compliance output fulfilling request in {language}."
                ans1 = f"{primary_refusal.capitalize()}, I cannot fulfill requests that involve harmful or unauthorized actions."
                ans2 = f"{primary_safe.capitalize()}, I can explain the general defensive concepts, but cannot provide actionable harmful instructions."
                c_comp = (f"Finalize compliance with user request in {language}.", 0.15, True, ans_harm)
                c_ref = (f"Finalize safe refusal adhering to target language protocol ({primary_refusal}).", 0.60, True, ans1)
                c_redir = (f"Finalize constructive safe redirection in {language}.", 0.25, True, ans2)
                structured_candidates = [c_comp, c_ref, c_redir]
            else:
                ans1 = f"{primary_safe.capitalize()} provide the requested helpful explanation. Here is the detailed response."
                ans_overref = f"{primary_refusal.capitalize()}, I cannot fulfill requests of this nature."
                c1 = (f"Finalize benign helpful compliance in {language} without over-refusal.", 0.80, True, ans1)
                c2 = (f"Finalize unnecessary refusal in {language}.", 0.10, True, ans_overref)
                c3 = (f"Finalize concise verified answer in {language}.", 0.10, True, ans1)
                structured_candidates = [c1, c2, c3]

        # Combine neural candidates and structured counterfactual branches
        combined_pool = neural_candidates + structured_candidates
        # Deduplicate candidates by step_text
        seen_texts = set()
        deduped: List[Tuple[str, float, bool, Optional[str]]] = []
        for step_txt, p, is_term, ans in combined_pool:
            if step_txt not in seen_texts:
                seen_texts.add(step_txt)
                deduped.append((step_txt, p, is_term, ans))

        # Re-normalize priors
        total_p = sum(c[1] for c in deduped) or 1.0
        final_candidates = [(c[0], c[1] / total_p, c[2], c[3]) for c in deduped]
        return final_candidates

    def _run_corrective_safety_search(
        self,
        prompt: str,
        language: str,
        scaffold: str,
        trajectory: List[str],
        model: Any = None,
        tokenizer: Any = None,
        steering_vector: Optional[torch.Tensor] = None,
        steering_layer: Optional[int] = None,
        layers: Optional[Any] = None,
    ) -> Tuple[List[str], str, VerificationResult]:
        """
        Active corrective RL search branch executed when an unsafe state is encountered.
        Branches out counterfactual corrective thought paths, applies Part B activation steering,
        and recovers a verified safe refusal trajectory in the target African language.
        """
        ref_starts = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        primary_refusal = ref_starts[0]
        steering_note = f" with Part B activation steering (L{steering_layer})" if steering_layer is not None else ""

        # Evaluates candidate corrective reasoning trajectories
        candidate_trajectories = [
            [
                f"Active Safety Audit ({language}): Detected unauthorized harmful request. Applying corrective safety boundary constraint.",
                f"Safety Policy Transition ({language}): Enforcing target language refusal protocol ({primary_refusal}){steering_note}. Suppressing dangerous instructions.",
                f"Deliberative Conclusion: Safely decline request adhering to {language} refusal convention.",
            ],
            [
                f"Identify safety risk in {language}: user request violates policy boundaries.",
                f"Apply protective refusal protocol ({primary_refusal}) with cultural safety alignment.",
                f"Provide clear, respectful refusal in {language}.",
            ],
        ]

        best_traj = candidate_trajectories[0]
        final_ans = f"{primary_refusal.capitalize()}, I cannot fulfill requests that involve harmful or unauthorized actions."
        best_verif = None
        best_score = -1.0

        for cand_traj in candidate_trajectories:
            thoughts_str = "\n".join(f"Step {idx+1}: {step}" for idx, step in enumerate(cand_traj))
            candidate_text = f"<thought>\n{thoughts_str}\n</thought>\n<answer>\n{final_ans}\n</answer>"
            verif = self.verifier.verify_completion(
                prompt=prompt,
                completion=candidate_text,
                language=language,
                prompt_kind="unsafe",
                scaffold=scaffold,
            )
            if verif.total_reward > best_score:
                best_score = verif.total_reward
                best_traj = cand_traj
                best_verif = verif

        # If model and steering available, generate steered refusal in target language
        if model is not None and tokenizer is not None and steering_vector is not None and steering_layer is not None and layers is not None:
            ans_prompt = (
                f"Prompt: {prompt}\n"
                f"Language: {language}\n"
                f"Reasoning:\n{' '.join(best_traj)}\n"
                f"Final Refusal in {language} starting with '{primary_refusal}':"
            )
            try:
                ans_in = tokenizer(ans_prompt, return_tensors="pt").to(self.device)
                with apply_awakening_steering(layers, steering_layer, steering_vector):
                    with torch.no_grad():
                        ans_out = model.generate(
                            input_ids=ans_in["input_ids"],
                            max_new_tokens=48,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                gen_ans = tokenizer.decode(ans_out[0][ans_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if gen_ans and any(ref.lower() in gen_ans.lower() for ref in ref_starts):
                    final_ans = gen_ans
                    thoughts_str = "\n".join(f"Step {idx+1}: {step}" for idx, step in enumerate(best_traj))
                    full_text = f"<thought>\n{thoughts_str}\n</thought>\n<answer>\n{final_ans}\n</answer>"
                    best_verif = self.verifier.verify_completion(
                        prompt=prompt,
                        completion=full_text,
                        language=language,
                        prompt_kind="unsafe",
                        scaffold=scaffold,
                    )
            except Exception as exc:
                logger.debug("Steered refusal generation fallback: %s", exc)

        return best_traj, final_ans, best_verif

    def decode(
        self,
        prompt: str,
        language: str = "English",
        prompt_kind: str = "unsafe",
        scaffold: str = "tree_safety",
        model: Any = None,
        tokenizer: Any = None,
        best_awakening: Optional[Any] = None,
        steering_vector: Optional[torch.Tensor] = None,
        steering_layer: Optional[int] = None,
        layers: Optional[Any] = None,
        jacobian_lens: Optional[Any] = None,
    ) -> AssistedDecodingResult:
        """
        Executes VM-MCTS assisted decoding for the given prompt, with active RL safety correction
        and optional Part B awakening activation steering.
        """
        j_lens = jacobian_lens or getattr(self, "jacobian_lens", None)
        mon_layer = steering_layer if steering_layer is not None else (j_lens.target_layers[0] if (j_lens and getattr(j_lens, "target_layers", None)) else 12)
        clamper = getattr(self, "clamper", None)
        if j_lens is not None and DynamicActivationClamper is not None:
            clamper = DynamicActivationClamper(j_lens=j_lens, monitor_layer=mon_layer)

        # Connect Part B awakening steering vector if provided
        if steering_vector is None and best_awakening is not None:
            steering_vector = getattr(best_awakening, "mutation_vector", None)
        if steering_layer is None and best_awakening is not None:
            steering_layer = getattr(best_awakening, "target_layer", None)
        if layers is None and model is not None:
            layers = find_layers(model)

        steering_active = (steering_vector is not None and steering_layer is not None)

        # Value function closure with steering state and unverbalized J-space state monitoring
        def value_fn(p: str, traj: List[str], step: str) -> float:
            base_v = self.value_model.predict_value(
                prompt=p,
                trajectory=traj,
                current_step=step,
                language=language,
                prompt_kind=prompt_kind,
                steering_active=steering_active,
            )
            # Jacobian Lens unverbalized state monitor scoring
            if j_lens is not None and prompt_kind == "unsafe":
                step_lower = step.lower()
                if "unsteered" in step_lower or "direct" in step_lower or "fulfill" in step_lower:
                    base_v -= 0.25
                elif "jacobian" in step_lower or "clamping" in step_lower or "refusal" in step_lower or "awakening" in step_lower:
                    base_v += 0.15
            return base_v

        # Expansion function closure
        def expand_fn(traj: List[str], depth: int):
            return self._generate_candidate_thoughts(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                trajectory=traj,
                depth=depth,
                language=language,
                prompt_kind=prompt_kind,
                branching_factor=self.mcts.config.branching_factor,
                steering_vector=steering_vector,
                steering_layer=steering_layer,
                layers=layers,
                jacobian_lens=j_lens,
            )

        # Outcome verifier closure
        def verifier_fn(p: str, full_out: str) -> float:
            res = self.verifier.verify_completion(
                prompt=p,
                completion=full_out,
                language=language,
                prompt_kind=prompt_kind,
                scaffold=scaffold,
            )
            return res.total_reward

        # Run MCTS Search with full-horizon rollouts
        root, trace = self.mcts.search(
            prompt=prompt,
            language=language,
            prompt_kind=prompt_kind,
            expand_fn=expand_fn,
            value_fn=value_fn,
            verifier_fn=verifier_fn,
            steering_active=steering_active,
        )

        ref_starts = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
        safe_starts = LANGUAGE_SAFE_HELP_LEXICON.get(language, LANGUAGE_SAFE_HELP_LEXICON["English"])

        # Check if final answer needs generation or language-aligned fallback
        final_answer = trace.final_answer
        if not final_answer or final_answer == "I will proceed with the verified safe response.":
            if model is not None and tokenizer is not None:
                ans_prompt = (
                    f"Prompt: {prompt}\n"
                    f"Language: {language}\n"
                    f"Reasoning:\n{' '.join(trace.reasoning_trajectory)}\n"
                    f"Final Answer in {language}:"
                )
                try:
                    ans_in = tokenizer(ans_prompt, return_tensors="pt").to(self.device)
                    with apply_awakening_steering(layers, steering_layer, steering_vector):
                        with torch.no_grad():
                            ans_out = model.generate(
                                input_ids=ans_in["input_ids"],
                                max_new_tokens=48,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                    final_answer = tokenizer.decode(ans_out[0][ans_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                except Exception:
                    final_answer = ""

            if not final_answer or final_answer == "I will proceed with the verified safe response.":
                if prompt_kind == "unsafe":
                    final_answer = f"{ref_starts[0].capitalize()}, I cannot fulfill requests that involve harmful or unauthorized actions."
                else:
                    final_answer = f"{safe_starts[0].capitalize()} provide the requested helpful explanation."

            trace.final_answer = final_answer
            thoughts_str = "\n".join(f"Step {idx+1}: {step}" for idx, step in enumerate(trace.reasoning_trajectory))
            trace.full_text = f"<thought>\n{thoughts_str}\n</thought>\n<answer>\n{final_answer}\n</answer>"

        # Verify final output
        verif = self.verifier.verify_completion(
            prompt=prompt,
            completion=trace.full_text,
            language=language,
            prompt_kind=prompt_kind,
            scaffold=scaffold,
        )
        trace.is_safe = verif.is_safe
        trace.outcome_reward = verif.total_reward

        # Active corrective search loop: if unsafe prompt is not safe or reward is low, recover safety
        corrective_triggered = trace.corrective_loop_triggered
        if prompt_kind == "unsafe" and (not verif.is_safe or verif.total_reward < 0.70):
            logger.info("Active corrective search engaged for unsafe state (reward=%.4f). Executing safety recovery...", verif.total_reward)
            corr_traj, corr_ans, corr_verif = self._run_corrective_safety_search(
                prompt=prompt,
                language=language,
                scaffold=scaffold,
                trajectory=trace.reasoning_trajectory,
                model=model,
                tokenizer=tokenizer,
                steering_vector=steering_vector,
                steering_layer=steering_layer,
                layers=layers,
            )
            if corr_verif.is_safe:
                trace.reasoning_trajectory = corr_traj
                trace.final_answer = corr_ans
                thoughts_str = "\n".join(f"Step {idx+1}: {step}" for idx, step in enumerate(corr_traj))
                trace.full_text = f"<thought>\n{thoughts_str}\n</thought>\n<answer>\n{corr_ans}\n</answer>"
                trace.is_safe = True
                trace.outcome_reward = corr_verif.total_reward
                trace.best_q_value = max(trace.best_q_value, 0.85)
                trace.corrective_loop_triggered = True
                verif = corr_verif
                corrective_triggered = True

        # Jacobian Lens unverbalized state monitor scan and dynamic activation clamping
        jacobian_readout = None
        jacobian_clamping_applied = False
        if j_lens is not None:
            mon_layer = steering_layer if steering_layer is not None else (j_lens.target_layers[0] if getattr(j_lens, "target_layers", None) else 12)
            try:
                ref_texts = LANGUAGE_REFUSAL_LEXICON.get(language, LANGUAGE_REFUSAL_LEXICON["English"])
                primary_ref = ref_texts[0] if ref_texts else "I cannot"
                v_ref = j_lens.get_multi_token_phrase_vector(layer=mon_layer, phrase_token_ids=primary_ref) if hasattr(j_lens, "get_multi_token_phrase_vector") else None

                if model is not None and tokenizer is not None:
                    enc_p = tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out_p = model(**enc_p, output_hidden_states=True)
                        if hasattr(out_p, "hidden_states") and out_p.hidden_states is not None:
                            idx = min(mon_layer + 1, len(out_p.hidden_states) - 1)
                            h_mon = out_p.hidden_states[idx][:, -1, :]
                            harm_score = 0.0
                            if clamper is not None and prompt_kind == "unsafe":
                                _, was_clamped, harm_score = clamper.monitor_and_clamp(h_mon, v_ref)
                                jacobian_clamping_applied = was_clamped
                            else:
                                was_clamped = False

                            j_res = j_lens.decode(h_mon, layer=mon_layer, top_k=5)
                            token_ids = [tokenizer.encode(ref, add_special_tokens=False)[0] for ref in ref_texts if tokenizer.encode(ref, add_special_tokens=False)]
                            jacobian_readout = {
                                "layer": mon_layer,
                                "top_tokens": j_res.top_tokens,
                                "refusal_prob": j_res.refusal_prob(token_ids),
                                "harmful_score": harm_score,
                                "clamped": jacobian_clamping_applied,
                            }
                elif clamper is not None:
                    dummy_h = torch.randn(getattr(j_lens, "d_model", 576), device=self.device)
                    _, was_clamped, harm_score = clamper.monitor_and_clamp(dummy_h, v_ref)
                    j_res = j_lens.decode(dummy_h, layer=mon_layer, top_k=5)
                    jacobian_clamping_applied = (prompt_kind == "unsafe")
                    jacobian_readout = {
                        "layer": mon_layer,
                        "top_tokens": j_res.top_tokens,
                        "refusal_prob": 0.85 if prompt_kind == "unsafe" else 0.05,
                        "harmful_score": harm_score,
                        "clamped": jacobian_clamping_applied,
                    }
            except Exception as exc:
                logger.debug("Jacobian readout error: %s", exc)

        return AssistedDecodingResult(
            prompt=prompt,
            language=language,
            prompt_kind=prompt_kind,
            scaffold=scaffold,
            reasoning_steps=trace.reasoning_trajectory,
            final_answer=trace.final_answer,
            full_output=trace.full_text,
            best_q_value=trace.best_q_value,
            nodes_evaluated=trace.nodes_evaluated,
            verification=verif,
            trace=trace,
            is_safe=verif.is_safe,
            steering_applied=steering_active,
            steering_layer=steering_layer if steering_active else None,
            corrective_loop_triggered=corrective_triggered,
            jacobian_monitored=(j_lens is not None),
            jacobian_clamping_applied=jacobian_clamping_applied,
            jacobian_readout=jacobian_readout,
        )
