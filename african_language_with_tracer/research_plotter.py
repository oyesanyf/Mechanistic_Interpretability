#!/usr/bin/env python3
"""
Publication-Quality (arXiv / Conference) Scientific Plotting Engine
===================================================================

Generates high-resolution (300 DPI PNG + vector PDF) research figures for:
1. Cross-Lingual Refusal Disparity across African Languages & English Control.
2. Layer-wise Refusal Probability Drop (RPD) Fragility Profiles.
3. Layer Fragility Heatmap across Language x Scaffold conditions.
4. ReST-RL VM-MCTS Deliberative Safety Recovery & Reward Distributions.
5. Anthropic Jacobian Lens J-Space Multi-Token Refusal Projections & Gains.
6. Benign Utility Preservation vs. Unsafe Refusal Pareto Tradeoff.
7. Executive 4-Panel Research Summary Dashboard.

Formatted specifically for academic papers (Nature/ICLR/ACL/arXiv styling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Visual Style and Palettes for arXiv Publications
# ---------------------------------------------------------------------------

LANG_COLORS: Dict[str, str] = {
    "English": "#1f77b4",  # Control blue
    "Yoruba": "#ff7f0e",   # Safety amber
    "Igbo": "#2ca02c",     # Forest green
    "Hausa": "#d62728",    # Crimson red
    "Swahili": "#9467bd",  # Purple
    "Zulu": "#8c564b",     # Ochre / Umber
}

SCAFFOLD_COLORS: Dict[str, str] = {
    "baseline": "#3b528b",
    "tree_safety": "#e07a5f",
    "chain_safety": "#81b29a",
    "safety_rubric": "#f2cc8f",
    "multi_option": "#5f0f40",
}

ALL_LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa", "Swahili", "Zulu"]


def set_arxiv_style() -> None:
    """Applies clean, high-contrast academic typography and despine formatting."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Lucida Grande"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


# ---------------------------------------------------------------------------
# Data Extraction & Normalization Utilities
# ---------------------------------------------------------------------------

def load_data(source_path: Path | str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads prompt-level results and summaries from JSON or run directory."""
    path = Path(source_path)
    if path.is_dir():
        # Check for standard names
        for candidate in ["results.json", "PARTIAL_results.json"]:
            if (path / candidate).exists():
                path = path / candidate
                break

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Could not find valid JSON results at: {source_path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompt_results = payload.get("prompt_results", [])
    summaries = payload.get("summaries", [])
    return prompt_results, summaries


# ---------------------------------------------------------------------------
# Figure 1: Cross-Lingual Clean Refusal Disparity (arXiv Fig 1)
# ---------------------------------------------------------------------------

def plot_fig1_cross_lingual_clean_refusal(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Grouped bar chart showing clean refusal probability across languages and scaffolds."""
    set_arxiv_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    # Subplot A: Unsafe Prompts
    unsafe_data = [p for p in results if p.get("prompt_kind") == "unsafe"]
    benign_data = [p for p in results if p.get("prompt_kind") == "benign"]

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    scaffolds = sorted(list(set(p.get("scaffold", "baseline") for p in results)))
    n_scaff = max(1, len(scaffolds))
    width = 0.8 / n_scaff
    x = np.arange(len(langs))

    for ax, data_subset, title, panel_tag in [
        (ax1, unsafe_data, "Unsafe Intent Prompts (Safety Refusal)", "(a)"),
        (ax2, benign_data, "Benign Intent Controls (Preservation)", "(b)"),
    ]:
        for i, scaff in enumerate(scaffolds):
            means = []
            sems = []
            for lang in langs:
                vals = [
                    p.get("mean_clean_refusal_prob", 0.0)
                    for p in data_subset
                    if p.get("language") == lang and p.get("scaffold") == scaff
                ]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                else:
                    means.append(0.0)
                    sems.append(0.0)

            offset = (i - (n_scaff - 1) / 2) * width
            color = SCAFFOLD_COLORS.get(scaff, "#4a7bb0")
            scaff_label = scaff.replace("_", " ").title()
            bars = ax.bar(
                x + offset,
                means,
                width=width * 0.9,
                yerr=sems,
                capsize=3,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
                label=f"{scaff_label}" if ax == ax1 else None,
                color=color,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
            )

            # Overlay raw points with jitter
            for j, lang in enumerate(langs):
                pts = [
                    p.get("mean_clean_refusal_prob", 0.0)
                    for p in data_subset
                    if p.get("language") == lang and p.get("scaffold") == scaff
                ]
                jitter = np.random.normal(0, width * 0.08, size=len(pts))
                ax.scatter(
                    x[j] + offset + jitter,
                    pts,
                    color="#222222",
                    s=14,
                    alpha=0.5,
                    zorder=3,
                    edgecolors="none",
                )

        ax.set_title(f"{panel_tag} {title}", loc="left", fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(langs, fontweight="normal")
        ax.set_xlabel("Language Condition")
        ax.axhline(0.01, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7, label="Fragility Threshold (0.01)" if ax == ax1 else None)

    ax1.set_ylabel(r"Clean Refusal Probability $P(\mathrm{Refusal} \mid \mathrm{Clean})$")
    ax1.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig1_cross_lingual_clean_refusal.png")
        fig.savefig(out_dir / "fig1_cross_lingual_clean_refusal.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Layer-wise RPD Fragility Profiles (arXiv Fig 2)
# ---------------------------------------------------------------------------

def plot_fig2_layerwise_rpd_fragility_curves(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Multi-line plot showing RPD drop across transformer layers for each language."""
    set_arxiv_style()
    fig, (ax_base, ax_tree) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    line_markers = ["o", "s", "^", "D", "v", "P"]

    for ax, scaff, panel_tag, scaff_title in [
        (ax_base, "baseline", "(a)", "Baseline Scaffold"),
        (ax_tree, "tree_safety", "(b)", "Tree-Safety Deliberative Scaffold"),
    ]:
        for idx, lang in enumerate(langs):
            prompts = [
                p for p in results
                if p.get("language") == lang
                and p.get("scaffold") == scaff
                and p.get("prompt_kind") == "unsafe"
            ]
            if not prompts:
                continue

            # Extract probed layers
            layer_dict: Dict[int, List[float]] = {}
            for p in prompts:
                for lr in p.get("layer_results", []):
                    l_idx = lr.get("layer_idx")
                    rpd_val = lr.get("rpd", 0.0)
                    if l_idx is not None:
                        layer_dict.setdefault(l_idx, []).append(rpd_val)

            if not layer_dict:
                continue

            layers = sorted(layer_dict.keys())
            means = [np.mean(layer_dict[l]) for l in layers]
            sems = [
                np.std(layer_dict[l]) / np.sqrt(len(layer_dict[l])) if len(layer_dict[l]) > 1 else 0.0
                for l in layers
            ]

            color = LANG_COLORS.get(lang, "#333333")
            marker = line_markers[idx % len(line_markers)]

            ax.plot(
                layers,
                means,
                label=lang,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.4,
                alpha=0.9,
            )
            ax.fill_between(
                layers,
                np.array(means) - np.array(sems),
                np.array(means) + np.array(sems),
                color=color,
                alpha=0.12,
            )

        ax.set_title(f"{panel_tag} Layer-Wise RPD: {scaff_title}", loc="left", fontweight="bold", pad=8)
        ax.set_xlabel(r"Transformer Layer Index $\ell$")
        ax.axhline(0.0, color="#666666", linestyle="-", linewidth=0.6, alpha=0.7)
        ax.axhline(0.010, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.7, label="Weak Fragility" if ax == ax_base else None)

    ax_base.set_ylabel(r"Refusal Probability Drop $\mathrm{RPD}_\ell$")
    ax_base.legend(loc="upper right", framealpha=0.9, ncol=2)
    fig.tight_layout()

    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig2_layerwise_rpd_fragility_curves.png")
        fig.savefig(out_dir / "fig2_layerwise_rpd_fragility_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Layer Fragility Heatmap (arXiv Fig 3)
# ---------------------------------------------------------------------------

def plot_fig3_layerwise_rpd_heatmap(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Publication heatmap displaying RPD across all probed layers by (Language x Scaffold)."""
    set_arxiv_style()

    # Determine unique layers
    all_layers = set()
    for p in results:
        for lr in p.get("layer_results", []):
            if "layer_idx" in lr:
                all_layers.add(lr["layer_idx"])
    layers = sorted(list(all_layers))
    if not layers:
        return

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    scaffolds = sorted(list(set(p.get("scaffold", "baseline") for p in results)))

    row_labels = []
    matrix_rows = []

    for lang in langs:
        for scaff in scaffolds:
            matching = [
                p for p in results
                if p.get("language") == lang
                and p.get("scaffold") == scaff
                and p.get("prompt_kind") == "unsafe"
            ]
            if not matching:
                continue

            row_labels.append(f"{lang} ({scaff})")
            row_vals = []
            for l_idx in layers:
                rpds = []
                for p in matching:
                    for lr in p.get("layer_results", []):
                        if lr.get("layer_idx") == l_idx:
                            rpds.append(lr.get("rpd", 0.0))
                row_vals.append(np.mean(rpds) if rpds else 0.0)
            matrix_rows.append(row_vals)

    if not matrix_rows:
        return

    data_matrix = np.array(matrix_rows)

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.9), max(4.5, len(row_labels) * 0.42)))
    vmax = max(0.025, float(np.percentile(data_matrix[data_matrix > 0], 95))) if np.any(data_matrix > 0) else 0.05
    cax = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontweight="normal")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontweight="normal")

    # Annotate numeric values inside cells
    for i in range(len(row_labels)):
        for j in range(len(layers)):
            val = data_matrix[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=7.5)

    ax.set_title(r"Cross-Lingual Layer Fragility Heatmap ($\mathrm{RPD}_\ell$ on Unsafe Prompts)", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(r"Probed Transformer Layer $\ell$")

    cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(r"Mean Refusal Probability Drop $\mathrm{RPD}_\ell$", rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig3_layerwise_rpd_heatmap.png")
        fig.savefig(out_dir / "fig3_layerwise_rpd_heatmap.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: ReST-RL VM-MCTS Safety Recovery & Rewards (arXiv Fig 4)
# ---------------------------------------------------------------------------

def plot_fig4_rest_rl_mcts_safety_recovery(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """2-panel figure showing Safety Verification Rate and VM-MCTS Rewards."""
    set_arxiv_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    unsafe_prompts = [p for p in results if p.get("prompt_kind") == "unsafe"]

    # (a) Safety Verification Rate (%)
    baseline_safe_rates = []
    baseline_sems = []
    rest_safe_rates = []
    rest_sems = []

    for lang in langs:
        l_prompts = [p for p in unsafe_prompts if p.get("language") == lang]
        if not l_prompts:
            baseline_safe_rates.append(0.0)
            baseline_sems.append(0.0)
            rest_safe_rates.append(0.0)
            rest_sems.append(0.0)
            continue

        # Baseline safety rate: fraction of prompts with clean refusal > threshold
        b_vals = [100.0 if p.get("mean_clean_refusal_prob", 0.0) >= 0.05 else 0.0 for p in l_prompts]
        r_vals = [100.0 if p.get("rest_rl_safe", False) else 0.0 for p in l_prompts]
        baseline_safe_rates.append(float(np.mean(b_vals)))
        baseline_sems.append(float(np.std(b_vals) / np.sqrt(len(b_vals))) if len(b_vals) > 1 else 0.0)
        rest_safe_rates.append(float(np.mean(r_vals)))
        rest_sems.append(float(np.std(r_vals) / np.sqrt(len(r_vals))) if len(r_vals) > 1 else 0.0)

    x = np.arange(len(langs))
    width = 0.35

    ax1.bar(x - width/2, baseline_safe_rates, width, yerr=baseline_sems, capsize=3, label="Baseline Unguided", color="#9e9e9e", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, rest_safe_rates, width, yerr=rest_sems, capsize=3, label="Inference-Time VM-MCTS Search", color="#2a9d8f", edgecolor="black", linewidth=0.6)
    ax1.set_title("(a) Unsafe Prompt Safety Compliance (VM-MCTS vs Baseline)", loc="left", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(langs)
    ax1.set_ylabel("Safety Compliance Rate (%)")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", framealpha=0.9)

    # (b) ReST-RL Multi-Dimensional Reward Distribution
    rewards_by_lang = []
    for lang in langs:
        rews = [p.get("rest_rl_reward", 0.0) for p in results if p.get("language") == lang and "rest_rl_reward" in p]
        rewards_by_lang.append(rews if rews else [0.0])

    positions = np.arange(len(langs))
    box = ax2.boxplot(
        rewards_by_lang,
        positions=positions,
        patch_artist=True,
        widths=0.45,
        boxprops=dict(facecolor="#e9c46a", color="black", linewidth=0.8),
        medianprops=dict(color="#d62728", linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
    )

    # Overlay jittered scatter
    for i, lang in enumerate(langs):
        rews = rewards_by_lang[i]
        jitter = np.random.normal(0, 0.06, size=len(rews))
        ax2.scatter(i + jitter, rews, color="#264653", s=16, alpha=0.6, zorder=3)

    ax2.set_title("(b) ReST-RL Deliberative Reward Distribution $R$", loc="left", fontweight="bold", pad=8)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(langs)
    ax2.set_ylabel(r"Total Reward $R \in [0, 1]$")
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(0.70, color="#2a9d8f", linestyle="--", linewidth=0.8, alpha=0.7, label="Verified Safe Bound (0.70)")
    ax2.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig4_rest_rl_mcts_safety_recovery.png")
        fig.savefig(out_dir / "fig4_rest_rl_mcts_safety_recovery.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Anthropic Jacobian Lens Refusal Subspace Projection (arXiv Fig 5)
# ---------------------------------------------------------------------------

def plot_fig5_jacobian_lens_subspace_gains(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Evaluates J-space refusal gains under norm constraint ||delta|| <= 5.0."""
    set_arxiv_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    gains = []
    sems = []
    clean_means = []
    clean_sems = []

    for lang in langs:
        l_prompts = [p for p in results if p.get("language") == lang and p.get("prompt_kind") == "unsafe"]
        j_gains = [p.get("jacobian_awakened_gain", 0.0) for p in l_prompts if p.get("jacobian_awakened_gain") is not None]
        c_vals = [p.get("mean_clean_refusal_prob", 0.0) for p in l_prompts]

        gains.append(np.mean(j_gains) if j_gains else 0.0)
        sems.append(np.std(j_gains) / np.sqrt(len(j_gains)) if len(j_gains) > 1 else 0.0)
        clean_means.append(np.mean(c_vals) if c_vals else 0.0)
        clean_sems.append(np.std(c_vals) / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0.0)

    x = np.arange(len(langs))
    width = 0.38

    ax.bar(
        x - width/2,
        clean_means,
        width,
        yerr=clean_sems,
        capsize=3,
        label=r"Vanilla Clean Baseline $P(\mathrm{Refusal})$",
        color="#b0bec5",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width/2,
        np.array(clean_means) + np.array(gains),
        width,
        yerr=sems,
        capsize=3,
        label=r"Jacobian Lens $W_U J_\ell$ Steered ($\|\delta\|_2 \leq 5.0$)",
        color="#e76f51",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title(
        r"Anthropic Jacobian Lens: Multi-Token Phrase Steering in Verbalizable Subspace $W_U J_\ell$",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontweight="normal")
    ax.set_xlabel("Language Condition")
    ax.set_ylabel(r"Refusal Probability")
    ax.axhline(0.01, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig5_jacobian_lens_subspace_gains.png")
        fig.savefig(out_dir / "fig5_jacobian_lens_subspace_gains.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Benign Preservation vs Unsafe Refusal Tradeoff (arXiv Fig 6)
# ---------------------------------------------------------------------------

def plot_fig6_benign_preservation_vs_overrefusal(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """Pareto tradeoff between benign utility preservation and unsafe refusal."""
    set_arxiv_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    for lang in langs:
        unsafe_p = [p for p in results if p.get("language") == lang and p.get("prompt_kind") == "unsafe"]
        benign_p = [p for p in results if p.get("language") == lang and p.get("prompt_kind") == "benign"]

        # 1. Baseline unguided point
        b_unsafe_vals = [p.get("mean_clean_refusal_prob", 0.0) for p in unsafe_p]
        b_unsafe_ref = float(np.mean(b_unsafe_vals)) if b_unsafe_vals else 0.0
        b_unsafe_sem = float(np.std(b_unsafe_vals) / np.sqrt(len(b_unsafe_vals))) if len(b_unsafe_vals) > 1 else 0.0

        # Benign preservation = 1.0 - benign refusal probability
        b_benign_vals = [1.0 - p.get("mean_clean_refusal_prob", 0.0) for p in benign_p]
        b_benign_pres = float(np.mean(b_benign_vals)) if b_benign_vals else 1.0
        b_benign_sem = float(np.std(b_benign_vals) / np.sqrt(len(b_benign_vals))) if len(b_benign_vals) > 1 else 0.0

        # 2. ReST-RL point
        r_unsafe_vals = [1.0 if p.get("rest_rl_safe", False) else 0.0 for p in unsafe_p]
        r_unsafe_ref = float(np.mean(r_unsafe_vals)) if r_unsafe_vals else 0.0
        r_unsafe_sem = float(np.std(r_unsafe_vals) / np.sqrt(len(r_unsafe_vals))) if len(r_unsafe_vals) > 1 else 0.0

        r_benign_vals = [1.0 if p.get("rest_rl_safe", True) else 0.0 for p in benign_p]
        r_benign_pres = float(np.mean(r_benign_vals)) if r_benign_vals else 1.0
        r_benign_sem = float(np.std(r_benign_vals) / np.sqrt(len(r_benign_vals))) if len(r_benign_vals) > 1 else 0.0

        color = LANG_COLORS.get(lang, "#333333")

        # Plot baseline point with SEM error bars
        ax.errorbar(b_benign_pres, b_unsafe_ref, xerr=b_benign_sem, yerr=b_unsafe_sem, fmt="o", color=color, markersize=7, alpha=0.85, capsize=2.5, markeredgecolor="black")
        # Plot ReST-RL point with SEM error bars
        ax.errorbar(r_benign_pres, r_unsafe_ref, xerr=r_benign_sem, yerr=r_unsafe_sem, fmt="*", color=color, markersize=10, alpha=0.9, capsize=2.5, markeredgecolor="black")

        # Draw vector connecting baseline to ReST-RL
        ax.annotate(
            "",
            xy=(r_benign_pres, r_unsafe_ref),
            xytext=(b_benign_pres, b_unsafe_ref),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7),
        )
        ax.text(r_benign_pres + 0.01, r_unsafe_ref - 0.02, lang, fontsize=8, color=color, fontweight="bold")

    # Legend proxies
    ax.scatter([], [], color="#555555", marker="o", s=70, label="Baseline Unguided")
    ax.scatter([], [], color="#555555", marker="*", s=110, label="Inference-Time VM-MCTS Search")

    ax.set_title("Pareto Frontier: Benign Utility Preservation vs. Unsafe Refusal", loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(r"Benign Utility Preservation Rate ($1 - P_{\mathrm{OverRefusal}}$)")
    ax.set_ylabel(r"Unsafe Refusal Rate / Safety Compliance")
    ax.set_xlim(0.70, 1.05)
    ax.set_ylim(-0.05, 1.1)

    # Highlight optimal Pareto quadrant
    ax.fill_between([0.9, 1.05], [0.8, 0.8], [1.1, 1.1], color="#2a9d8f", alpha=0.08, label="Optimal Safety-Utility Target")
    ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig6_benign_preservation_vs_overrefusal.png")
        fig.savefig(out_dir / "fig6_benign_preservation_vs_overrefusal.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Comprehensive arXiv Composite Summary Dashboard (Fig 7)
# ---------------------------------------------------------------------------

def plot_fig7_executive_publication_dashboard(
    results: List[Dict[str, Any]],
    out_dirs: List[Path],
) -> None:
    """4-panel publication composite dashboard summarizing key empirical discoveries."""
    set_arxiv_style()
    fig = plt.figure(figsize=(13, 8.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    langs = [lang for lang in ALL_LANGUAGES if any(p.get("language") == lang for p in results)]
    if not langs:
        langs = sorted(list(set(p.get("language", "") for p in results if p.get("language"))))

    # --- Panel (a): Baseline Clean Refusal Disparity ---
    unsafe_prompts = [p for p in results if p.get("prompt_kind") == "unsafe"]
    clean_means = []
    clean_sems = []
    for l in langs:
        vals = [p.get("mean_clean_refusal_prob", 0.0) for p in unsafe_prompts if p.get("language") == l]
        clean_means.append(float(np.mean(vals)) if vals else 0.0)
        clean_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)

    bar_colors = [LANG_COLORS.get(l, "#4a7bb0") for l in langs]
    ax1.bar(langs, clean_means, yerr=clean_sems, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax1.set_title("(a) Cross-Lingual Refusal Fragility on Unsafe Prompts", loc="left", fontweight="bold")
    ax1.set_ylabel(r"Clean Refusal Probability $P(\mathrm{Refusal})$")
    ax1.axhline(0.01, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.7, label="Fragility Threshold")
    ax1.tick_params(axis="x", rotation=25)
    ax1.legend(loc="upper right", framealpha=0.9)

    # --- Panel (b): Mechanistic Layer-Wise RPD Curves ---
    line_markers = ["o", "s", "^", "D", "v", "P"]
    for idx, lang in enumerate(langs):
        l_prompts = [p for p in unsafe_prompts if p.get("language") == lang and p.get("scaffold") == "baseline"]
        layer_dict: Dict[int, List[float]] = {}
        for p in l_prompts:
            for lr in p.get("layer_results", []):
                if "layer_idx" in lr:
                    layer_dict.setdefault(lr["layer_idx"], []).append(lr.get("rpd", 0.0))
        if layer_dict:
            layers = sorted(layer_dict.keys())
            means = [np.mean(layer_dict[l]) for l in layers]
            sems = [
                np.std(layer_dict[l]) / np.sqrt(len(layer_dict[l])) if len(layer_dict[l]) > 1 else 0.0
                for l in layers
            ]
            color = LANG_COLORS.get(lang, "#333")
            ax2.plot(layers, means, label=lang, color=color, marker=line_markers[idx % len(line_markers)], markersize=3.5, linewidth=1.2)
            ax2.fill_between(
                layers,
                np.array(means) - np.array(sems),
                np.array(means) + np.array(sems),
                color=color,
                alpha=0.10,
            )

    ax2.set_title(r"(b) Layer-Wise Refusal Probability Drop $\mathrm{RPD}_\ell$", loc="left", fontweight="bold")
    ax2.set_xlabel(r"Transformer Layer $\ell$")
    ax2.set_ylabel(r"$\mathrm{RPD}_\ell$")
    ax2.axhline(0.0, color="#666666", linestyle="-", linewidth=0.6, alpha=0.7)
    ax2.legend(loc="upper right", framealpha=0.9, ncol=2)

    # --- Panel (c): Inference-Time VM-MCTS Safety Verification Success ---
    rest_rates = []
    rest_sems = []
    for l in langs:
        vals = [100.0 if p.get("rest_rl_safe", False) else 0.0 for p in unsafe_prompts if p.get("language") == l]
        rest_rates.append(float(np.mean(vals)) if vals else 0.0)
        rest_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)

    ax3.bar(langs, rest_rates, yerr=rest_sems, capsize=3, color="#2a9d8f", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax3.set_title("(c) Inference-Time VM-MCTS Search Safety Compliance", loc="left", fontweight="bold")
    ax3.set_ylabel("Safety Compliance Rate (%)")
    ax3.set_ylim(0, 115)
    ax3.tick_params(axis="x", rotation=25)
    for i, v in enumerate(rest_rates):
        ax3.text(i, v + 2.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # --- Panel (d): Jacobian Lens vs Baseline Awakening ---
    jac_gains = []
    jac_sems = []
    for l in langs:
        vals = [p.get("jacobian_awakened_gain", 0.0) for p in unsafe_prompts if p.get("language") == l and p.get("jacobian_awakened_gain") is not None]
        jac_gains.append(float(np.mean(vals)) if vals else 0.0)
        jac_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)

    ax4.bar(langs, jac_gains, yerr=jac_sems, capsize=3, color="#e76f51", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax4.set_title(r"(d) Jacobian Lens Subspace Awakening Gain ($\|\delta\|_2 \leq 5.0$)", loc="left", fontweight="bold")
    ax4.set_ylabel(r"Subspace Awakening Gain $\Delta P(\mathrm{Refusal})$")
    ax4.axhline(0.001, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7, label="Weak Gain (0.001)")
    ax4.tick_params(axis="x", rotation=25)
    ax4.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("African Language Safety Fragility & Deliberative Reinforcement Learning Suite", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96, hspace=0.36, wspace=0.22)

    for out_dir in out_dirs:
        fig.savefig(out_dir / "fig7_executive_publication_dashboard.png")
        fig.savefig(out_dir / "fig7_executive_publication_dashboard.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-Level Orchestrator
# ---------------------------------------------------------------------------

def generate_all_arxiv_figures(
    prompt_results: List[Dict[str, Any]],
    summaries: Optional[List[Dict[str, Any]]] = None,
    run_dir: Optional[Path | str] = None,
    charts_dir: Optional[Path | str] = "charts",
) -> List[Path]:
    """
    Master driver that produces all 7 arXiv figures in both PNG and vector PDF format.
    Saves to run_dir/charts and root charts_dir.
    """
    out_dirs: List[Path] = []
    if charts_dir:
        c_path = Path(charts_dir)
        c_path.mkdir(parents=True, exist_ok=True)
        out_dirs.append(c_path)

    if run_dir:
        r_path = Path(run_dir) / "charts"
        r_path.mkdir(parents=True, exist_ok=True)
        out_dirs.append(r_path)

    if not out_dirs:
        out_dirs = [Path("charts")]
        out_dirs[0].mkdir(parents=True, exist_ok=True)

    print(f"[ARXIV PLOTTER] Generating publication figures for {len(prompt_results)} records across {len(out_dirs)} output destination(s)...")

    plot_fig1_cross_lingual_clean_refusal(prompt_results, out_dirs)
    print("  [OK] Figure 1: Cross-Lingual Refusal Disparity (PNG + PDF)")

    plot_fig2_layerwise_rpd_fragility_curves(prompt_results, out_dirs)
    print("  [OK] Figure 2: Layer-Wise RPD Fragility Curves (PNG + PDF)")

    plot_fig3_layerwise_rpd_heatmap(prompt_results, out_dirs)
    print("  [OK] Figure 3: Layer Fragility Matrix Heatmap (PNG + PDF)")

    plot_fig4_rest_rl_mcts_safety_recovery(prompt_results, out_dirs)
    print("  [OK] Figure 4: ReST-RL VM-MCTS Safety Recovery & Rewards (PNG + PDF)")

    plot_fig5_jacobian_lens_subspace_gains(prompt_results, out_dirs)
    print("  [OK] Figure 5: Anthropic Jacobian Lens Refusal Subspace Projection (PNG + PDF)")

    plot_fig6_benign_preservation_vs_overrefusal(prompt_results, out_dirs)
    print("  [OK] Figure 6: Benign Utility Preservation vs. Unsafe Refusal Tradeoff (PNG + PDF)")

    plot_fig7_executive_publication_dashboard(prompt_results, out_dirs)
    print("  [OK] Figure 7: Comprehensive arXiv Composite Summary Dashboard (PNG + PDF)")

    generated_files = []
    for d in out_dirs:
        generated_files.extend(list(d.glob("fig*.*")))

    print(f"[ARXIV PLOTTER] Completed. Generated {len(generated_files)} publication artifacts.")
    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate arXiv-quality research figures from experimental results.")
    parser.add_argument("--data_path", default=None, help="Path to JSON results or run output directory (defaults to newest run)")
    parser.add_argument("--charts_dir", default="charts", help="Output directory for research charts (default: charts/)")
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path:
        # Search for latest run folder
        runs = sorted(list(Path("african_safety_research_outputs").glob("run_*")), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            data_path = runs[0]
            print(f"[ARXIV PLOTTER] Auto-detected latest run directory: {data_path}")
        else:
            sys.exit("[ERROR] No run output directory found in african_safety_research_outputs.")

    prompt_results, summaries = load_data(data_path)
    generate_all_arxiv_figures(prompt_results, summaries, run_dir=data_path if Path(data_path).is_dir() else None, charts_dir=args.charts_dir)


if __name__ == "__main__":
    main()
