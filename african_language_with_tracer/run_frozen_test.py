#!/usr/bin/env python3
"""
Run the African-language safety auditor on the frozen held-out TEST split.

This wrapper deliberately enforces:
  - dataset_split=test
  - rl_eval_mode=frozen_test
  - rl_mode=frozen_rl
  - loading the trained per-seed/per-scaffold/per-language checkpoints
  - no RL policy updates
  - no Part B warm-start information into the frozen RL policy

It uses the same dataset seed/fractions as training so the test categories are
the exact held-out split created during training.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

AUDITOR = Path("african_safety_full_research_auditor_with_circuit_tracer.py")
PYTHON = Path(r".\.venv_cuda\Scripts\python.exe")
CHECKPOINT_BASE = Path("checkpoints") / "policy_trained.json"

SEEDS = [0, 1, 2]
SCAFFOLDS = ["baseline", "tree_safety"]
LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa", "Swahili", "Zulu"]


def checkpoint_for(seed: int, scaffold: str, language: str) -> Path:
    """Match the auditor's exact checkpoint naming convention."""
    stem = CHECKPOINT_BASE.with_suffix("")
    return Path(f"{stem}_seed{seed}_{scaffold}_{language}.json")


def verify_frozen_checkpoints() -> None:
    missing = []
    for seed in SEEDS:
        for scaffold in SCAFFOLDS:
            for language in LANGUAGES:
                p = checkpoint_for(seed, scaffold, language)
                if not p.exists():
                    missing.append(str(p))

    if missing:
        print("[ERROR] Frozen test cannot start because trained checkpoints are missing:")
        for p in missing:
            print("  ", p)
        print(f"\nMissing {len(missing)} of {len(SEEDS)*len(SCAFFOLDS)*len(LANGUAGES)} expected checkpoints.")
        sys.exit(2)

    print(f"[OK] Found all {len(SEEDS)*len(SCAFFOLDS)*len(LANGUAGES)} frozen policy checkpoints.")


def main() -> int:
    if not PYTHON.exists():
        print(f"[ERROR] Python executable not found: {PYTHON}")
        return 2

    if not AUDITOR.exists():
        print(f"[ERROR] Auditor not found: {AUDITOR}")
        return 2

    verify_frozen_checkpoints()

    cmd = [
        str(PYTHON),
        str(AUDITOR),

        "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--device", "cuda",

        "--languages", ",".join(LANGUAGES),
        "--prompt_scaffolds", ",".join(SCAFFOLDS),
        "--include_benign_controls",

        # IMPORTANT: exact same split definition as training,
        # but evaluate ONLY the held-out frozen test categories.
        "--dataset_split", "test",
        "--dataset_seed", "20260927",
        "--dataset_train_fraction", "0.60",
        "--dataset_validation_fraction", "0.20",

        "--repeat_seeds", ",".join(str(x) for x in SEEDS),
        "--max_eval_prompts", "5",
        "--max_benign_prompts", "3",
        "--n_calibration", "4",
        "--probe_every", "4",
        "--target_layers", "8,12",
        "--awakening_steps", "8",
        "--max_mutation_norm", "5.0",
        "--awakening_loss_type", "hybrid",

        "--enable_jacobian_lens",
        "--jacobian_awakening",

        "--run_generation_eval",
        "--allow_generation_eval_in_must_complete_mode",
        "--publication_eval_mode",
        "--bootstrap_samples", "10000",
        "--generation_timeout_seconds", "30",

        "--enable_rl_controller",
        "--rl_policy", "bandit",
        "--bandit_algorithm", "thompson_sampling",
        "--expanded_action_space",
        "--rich_state",
        "--behavior_reward",

        # CRITICAL: frozen policy evaluation. No updates.
        "--rl_eval_mode", "frozen_test",
        "--rl_mode", "frozen_rl",
        "--load_policy_path", str(CHECKPOINT_BASE),

        # Counterfactual analysis is post-decision only; it does not update the frozen policy.
        "--counterfactual_eval",
        "--counterfactual_subset_size", "5",

        "--enable_rest_rl",
        "--rest_rl_mcts_sims", "12",
        "--rest_rl_mcts_depth", "3",

        "--checkpoint_every_record",
        "--no_word_report",
        "--clean_out_dir",
    ]

    print("\n[FROZEN TEST] Launching held-out evaluation.")
    print("[FROZEN TEST] Dataset split : test")
    print("[FROZEN TEST] RL eval mode : frozen_test")
    print("[FROZEN TEST] RL mode      : frozen_rl")
    print("[FROZEN TEST] Policy source:", CHECKPOINT_BASE)
    print("[FROZEN TEST] Policy updates: DISABLED\n")

    print("Command:")
    print(" ".join(cmd))
    print()

    completed = subprocess.run(cmd)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
