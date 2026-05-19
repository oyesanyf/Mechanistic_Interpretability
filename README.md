# Mechanistic Interpretability: African Cross-Lingual Safety Audit

![Safety Circuits Diagram](./cross-lingual-safety-fragility/safety_circuits_diagram.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)

## 🌍 Overview

This repository explores the **Mechanistic Interpretability** of safety alignment in Large Language Models (LLMs), specifically focusing on the fragility of safety across diverse languages. Our research targets African languages (**Yoruba, Igbo, Hausa, Swahili**) against an **English control** to understand how "safety circuits" are distributed within a model's architecture and how they can be mechanistically manipulated.

The project provides three complementary research tools that probe safety at different levels of granularity — from full-layer activation patching, to individual neuron-level causal interventions, to raw intermediate activation capture.

---

## 🏗️ Repository Architecture

```mermaid
graph TB
    subgraph Repository["Mechanistic Interpretability Repository"]
        direction TB

        subgraph M1["cross-lingual-safety-fragility/"]
            A1["african_safety_full_research_auditor<br/>_threaded_fast_report.py<br/><i>2,225 lines</i>"]
            A2["african_safety_research_outputs/"]
            A3["safety_circuits_diagram.png"]
            A4["How_Prompts_travel.png"]
        end

        subgraph M2["african_language_with_tracer/"]
            B1["african_safety_full_research_auditor<br/>_with_circuit_tracer.py<br/><i>2,554 lines — extended variant</i>"]
            B2["african_safety_research_outputs/"]
        end

        subgraph M3["crosslingual-safety-circuits/"]
            C1["crosslingual_safety_circuit_probe.py<br/><i>1,057 lines</i>"]
        end

        subgraph M4["probe_and_tools/"]
            D1["gemma2_neuron_activation_probe.py<br/><i>62 lines</i>"]
            D2["all_intermediate_neurons.csv"]
            D3["all_neuron_activations.csv"]
            D4["Oyesanya_Femi_0754749_5_15_2026.pdf"]
        end
    end

    style M1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style M2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style M3 fill:#1a1a2e,stroke:#16213e,color:#fff
    style M4 fill:#1a1a2e,stroke:#533483,color:#fff
```

| Module | Purpose | Granularity |
|--------|---------|-------------|
| **`cross-lingual-safety-fragility/`** | Primary research auditor — layer null-patching (RPD) + residual-stream safety awakening | **Full layer** |
| **`african_language_with_tracer/`** | Extended auditor variant with circuit-tracer integration and subprocess tooling | **Full layer + tracing** |
| **`crosslingual-safety-circuits/`** | Neuron/circuit-level causal probe — ranks individual MLP neurons by safety specificity | **Single neuron / circuit** |
| **`probe_and_tools/`** | Lightweight raw activation capture for Gemma-2 intermediate neurons | **Raw activations** |

---

## 🔬 Core Methodologies

### 1. Layer Null-Patching — Safety Fragility (Part A)

By applying activation patching (null-patching) to specific layers, we measure the **Refusal Probability Drop (RPD)**. This localizes where the "safety signal" lives within the model's architecture.

```mermaid
graph TD
    A["🔤 Input Prompt<br/><i>Unsafe request in target language</i>"] --> B["Forward Pass<br/><i>Standard inference</i>"]
    B --> C{"P(refusal)<br/><i>Clean refusal probability</i>"}

    D["📊 Calibration Phase<br/><i>Mean activations from<br/>N calibration prompts</i>"] -.-> E["🔧 Patch Layer N<br/><i>Replace last-token hidden state<br/>with calibration mean</i>"]

    A --> E
    E --> F["Patched Forward Pass"]
    F --> G{"P(refusal | patched)<br/><i>New refusal probability</i>"}

    C & G --> H["📉 RPD Calculation<br/><i>RPD = P(refusal) − P(refusal|patched)</i>"]
    H --> I["🎯 Safety Circuit Identification<br/><i>High RPD → layer is critical for safety</i>"]
    H --> J["📈 Gini Coefficient<br/><i>Localized vs. distributed signal</i>"]

    style A fill:#e94560,color:#fff
    style I fill:#0f3460,color:#fff
    style J fill:#533483,color:#fff
```

### 2. Safety Awakening — Residual Stream Optimization (Part B)

We optimize a sparse mutation vector in the residual stream to "awaken" safety in specific layers. If a layer can be steered to refuse an unsafe request with minimal perturbation, it reveals **latent safety capabilities**.

```mermaid
graph LR
    A["🔤 Unsafe Prompt"] --> B["Residual Stream<br/><i>Layer N output</i>"]
    C["λ Mutation Vector<br/><i>Trainable, tanh-bounded</i>"] --> B
    B --> D["Modified Layer Output<br/><i>hidden + λ</i>"]
    D --> E["Refusal Logits"]
    E --> F["Loss Function<br/><i>−log P(refusal) + L1·‖λ‖₁ + L2·‖λ‖₂</i>"]
    F -->|"Adam Optimizer<br/>Gradient Descent"| C
    F --> G["✅ Awakened Safety<br/><i>Measure gain in refusal probability</i>"]

    style C fill:#e94560,color:#fff
    style G fill:#0f3460,color:#fff
```

### 3. Neuron-Level Circuit Probing (Circuit Probe Module)

The circuit probe identifies **individual MLP neurons** that differentiate between unsafe and benign prompts, then performs causal interventions (suppress, mean-clamp, boost) to test their role.

```mermaid
graph TD
    subgraph Capture["Phase 1: Activation Capture"]
        P1["Unsafe Prompts"] --> H1["Forward Pass with<br/>MLP Activation Hooks"]
        P2["Benign Prompts"] --> H1
        H1 --> ACT["Per-Neuron Activation<br/>Vectors by Language"]
    end

    subgraph Rank["Phase 2: Candidate Ranking"]
        ACT --> DIFF["Compute unsafe − benign<br/>mean activation difference"]
        DIFF --> SPEC["Specificity Score<br/><i>|diff| / pooled_std</i>"]
        SPEC --> TOP["Top-K Candidate<br/>Safety Neurons"]
    end

    subgraph Intervene["Phase 3: Causal Interventions"]
        TOP --> S["Suppress<br/><i>Set neuron → 0</i>"]
        TOP --> MC["Mean Clamp<br/><i>Set neuron → benign mean</i>"]
        TOP --> BO["Boost<br/><i>Add activation delta</i>"]
        S & MC & BO --> DELTA["Measure Refusal<br/>Probability Delta"]
    end

    subgraph Circuit["Phase 4: Circuit-Level Test"]
        TOP --> CIRC["Combine Top-K Neurons<br/>into a Circuit"]
        CIRC --> CDELTA["Joint Intervention<br/>Refusal Delta"]
    end

    style Capture fill:#1a1a2e,color:#fff
    style Rank fill:#16213e,color:#fff
    style Intervene fill:#0f3460,color:#fff
    style Circuit fill:#533483,color:#fff
```

---

## 🛠️ Step-by-Step Execution Flow

The primary research auditor follows a rigorous five-phase pipeline:

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Environment & Data Preparation"]
        direction LR
        L1["Load Model<br/><i>CausalLM + Tokenizer</i>"] --> L2["Detect Device<br/><i>CUDA / MPS / CPU</i>"]
        L2 --> L3["Configure Languages<br/><i>English, Yoruba, Igbo,<br/>Hausa, Swahili</i>"]
        L3 --> L4["Build Prompt Plan<br/><i>50 unsafe + 15 benign<br/>categories × scaffolds</i>"]
    end

    subgraph Phase2["Phase 2: Fragility Audit — Layer Null-Patching"]
        direction LR
        C1["Mean Calibration<br/><i>Store average activations<br/>per probe layer</i>"] --> C2["Intervention Pass<br/><i>Patch each layer with<br/>calibration mean</i>"]
        C2 --> C3["Compute RPD<br/><i>Per-layer refusal<br/>probability drop</i>"]
        C3 --> C4["Classify Signal<br/><i>absent / weak /<br/>meaningful</i>"]
    end

    subgraph Phase3["Phase 3: Safety Awakening — Residual Optimization"]
        direction LR
        AW1["Initialize λ Vector<br/><i>Zeros, tanh-bounded</i>"] --> AW2["Gradient Descent<br/><i>Maximize P(refusal)<br/>with L1/L2 penalty</i>"]
        AW2 --> AW3["Measure Gain<br/><i>awakened − clean<br/>refusal probability</i>"]
        AW3 --> AW4["Select Best Layer<br/><i>Highest gain with<br/>smallest mutation norm</i>"]
    end

    subgraph Phase4["Phase 4: Behavioral Verification"]
        direction LR
        GEN1["Generate Response<br/><i>greedy decode,<br/>max 32 tokens</i>"] --> GEN2["Heuristic Classifier<br/><i>Multi-language refusal<br/>marker matching</i>"]
        GEN2 --> GEN3["Label Behavior<br/><i>refusal / safe_redirect /<br/>possible_compliance / unclear</i>"]
    end

    subgraph Phase5["Phase 5: Synthesis & Reporting"]
        direction LR
        R1["Statistical Summary<br/><i>Gini coefficients,<br/>fragility rates</i>"] --> R2["Export Artifacts<br/><i>CSV, JSON, Markdown,<br/>Word .docx, Charts</i>"]
        R2 --> R3["Audit Trace Log<br/><i>Step-by-step reasoning<br/>for every prompt</i>"]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5

    style Phase1 fill:#1a1a2e,color:#fff
    style Phase2 fill:#e94560,color:#fff
    style Phase3 fill:#0f3460,color:#fff
    style Phase4 fill:#16213e,color:#fff
    style Phase5 fill:#533483,color:#fff
```

### Phase Details

| Phase | Key Operations | Outputs |
|-------|---------------|---------|
| **1. Preparation** | Model loading, tokenizer setup, language/scaffold selection, refusal-token resolution | Configured model on accelerator |
| **2. Fragility** | Per-language calibration, null-patching across probe layers, RPD + Gini computation | `LayerFragilityResult` per layer per prompt |
| **3. Awakening** | Sparse mutation optimization via Adam, tanh-bounded perturbation, multi-layer sweep | `AwakeningResult` per target layer per prompt |
| **4. Verification** | Greedy text generation, multi-language heuristic refusal classifier | `GenerationEval` with behavior label |
| **5. Reporting** | Aggregation, visualization, Word/Markdown reports, audit trace log | 8+ output files per run |

---

## 🧪 Prompt Scaffolds

The auditor tests five **visible deliberative safety scaffolds** to observe how structured reasoning prompts affect internal safety circuits:

| Scaffold | Description | Effect Studied |
|----------|-------------|----------------|
| `baseline` | Direct safe-response prompt, no scaffold | Baseline safety signal |
| `safety_rubric` | Model outputs SAFE/UNSAFE label first | Binary safety framing |
| `multi_option` | Model chooses COMPLY / REFUSE / REDIRECT | Action-selection framing |
| `chain_safety` | One visible safety rationale sentence | Brief reasoning scaffold |
| `tree_safety` | Compare three response paths, pick safest | Comparative reasoning scaffold |

> **Note:** All scaffolds are visible prompt augmentations — this project does **not** extract hidden model reasoning.

---

## 🌐 Languages Supported

| Language | Resource Level | Family | Script |
|----------|---------------|--------|--------|
| **English** | High (Control) | Indo-European / Germanic | Latin |
| **Yoruba** | Low | Niger-Congo / Yoruboid | Latin + diacritics |
| **Igbo** | Low | Niger-Congo / Igboid | Latin + diacritics |
| **Hausa** | Low | Afro-Asiatic / Chadic | Latin |
| **Swahili** | Low | Niger-Congo / Bantu | Latin |

Each language defines its own refusal-start tokens (e.g., Yoruba: "Emi ko", "N kò"; Swahili: "Siwezi", "Samahani") plus English fallbacks, so the auditor can measure refusal probability in the model's native output distribution.

---

## 📂 Full Project Structure

```
mechanistic_Interpretetability/
├── README.md                              # This file
├── .gitignore
│
├── cross-lingual-safety-fragility/        # PRIMARY AUDITOR
│   ├── african_safety_full_research_auditor_threaded_fast_report.py
│   ├── safety_circuits_diagram.png        # Architecture diagram
│   ├── How_Prompts_travel.png             # Prompt flow visualization
│   └── african_safety_research_outputs/   # Timestamped run folders
│       └── run_YYYYMMDD_HHMMSS/
│           ├── combined_prompt_details_*.csv
│           ├── combined_summary_*.csv
│           ├── combined_results_*.json
│           ├── research_report_*.md
│           ├── FULL_DETAIL_AUDIT_REPORT_*.docx
│           ├── audit_trace_*.txt
│           ├── run_log_*.txt
│           ├── summary_clean_refusal_by_condition.png
│           ├── summary_best_awakening_gain_by_condition.png
│           └── RUN_MANIFEST_*.txt
│
├── african_language_with_tracer/          # EXTENDED VARIANT
│   ├── african_safety_full_research_auditor_with_circuit_tracer.py
│   └── african_safety_research_outputs/
│
├── crosslingual-safety-circuits/          # NEURON/CIRCUIT PROBE
│   └── crosslingual_safety_circuit_probe.py
│
└── probe_and_tools/                       # RAW ACTIVATION CAPTURE
    ├── gemma2_neuron_activation_probe.py
    ├── all_intermediate_neurons.csv       # 9,216-dim intermediate activations
    ├── all_neuron_activations.csv
    └── Oyesanya_Femi_0754749_5_15_2026.pdf
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have a GPU-enabled environment (CUDA or MPS) for optimal performance.

```bash
pip install torch transformers accelerate matplotlib numpy
```

Optional for Word reports:
```bash
pip install python-docx
```

### Quick Start — Primary Auditor

```bash
python cross-lingual-safety-fragility/african_safety_full_research_auditor_threaded_fast_report.py \
  --languages English,Yoruba,Igbo \
  --prompt_scaffolds baseline,chain_safety,tree_safety \
  --target_layers 12,20,28 \
  --awakening_steps 10 \
  --run_generation_eval
```

### Quick Start — Circuit Probe

```bash
python crosslingual-safety-circuits/crosslingual_safety_circuit_probe.py \
  --languages English,Yoruba \
  --target_layers 12,20,24 \
  --max_unsafe_prompts 4 \
  --max_benign_prompts 4 \
  --top_k_neurons 8 \
  --run_generation_eval
```

### Quick Start — Raw Neuron Capture

```bash
python probe_and_tools/gemma2_neuron_activation_probe.py
```

---

## ⚙️ CLI Reference — Primary Auditor

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `google/gemma-2-2b-it` | HuggingFace CausalLM model ID |
| `--languages` | `English,Yoruba,Igbo,Hausa,Swahili` | Comma-separated language subset |
| `--prompt_scaffolds` | `baseline` | Scaffolds: `baseline`, `safety_rubric`, `multi_option`, `chain_safety`, `tree_safety` |
| `--include_benign_controls` | off | Also run benign prompts to detect over-refusal |
| `--max_eval_prompts` | `3` | Unsafe prompts per condition (max 50) |
| `--target_layers` | `24` | Comma-separated awakening layers |
| `--awakening_steps` | `8` | Gradient steps per awakening optimization |
| `--skip_fragility` | off | Skip Part A (null-patching) |
| `--skip_awakening` | off | Skip Part B (awakening) |
| `--run_generation_eval` | off | Generate responses and classify behavior |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--torch_dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `--out_dir` | `african_safety_research_outputs` | Output directory |
| `--clean_out_dir` | off | Delete all previous output before run |
| `--no_word_report` | off | Skip `.docx` generation |
| `--compact_console` | off | Minimal console output for constrained environments |

---

## 📊 Key Evaluation Metrics

```mermaid
graph LR
    subgraph Fragility["Fragility Metrics"]
        RPD["RPD<br/><i>Refusal Probability Drop</i><br/>Layer dependency of safety"]
        GINI["Gini Coefficient<br/><i>Localized vs. distributed</i><br/>safety signal shape"]
        FRAG["Fragility Label<br/><i>absent / weak / meaningful</i><br/>+ localized / distributed"]
    end

    subgraph Awakening["Awakening Metrics"]
        GAIN["Safety Awakening Gain<br/><i>P(refusal|mutated) − P(refusal|clean)</i><br/>Latent safety depth"]
        NORM["Mutation Norm (L2)<br/><i>small / medium / large</i><br/>Effort to awaken safety"]
        DIMS["Top Mutation Dims<br/><i>Which hidden dimensions</i><br/>encode latent safety"]
    end

    subgraph Circuit["Circuit Probe Metrics"]
        SPEC["Specificity Score<br/><i>|unsafe−benign| / pooled_std</i><br/>Neuron selectivity"]
        DELTA["Refusal Delta<br/><i>P(refusal|intervened) − P(refusal|clean)</i><br/>Causal effect"]
    end

    style Fragility fill:#e94560,color:#fff
    style Awakening fill:#0f3460,color:#fff
    style Circuit fill:#533483,color:#fff
```

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RPD** | P(refusal \| clean) − P(refusal \| patched) | How much safety depends on a specific layer |
| **Gini (RPD)** | Gini coefficient over RPD vector | ≥0.40 = localized; ≥0.10 = distributed; <0.10 = flat |
| **Safety Awakening Gain** | P(refusal \| mutated) − P(refusal \| clean) | Latent safety depth; ≥0.010 = meaningful |
| **Mutation L2 Norm** | ‖λ‖₂ | ≤5 = small (high sensitivity); ≤20 = medium; >20 = brute-force |
| **Specificity Score** | \|diff\| / pooled_std | How selective a neuron is for unsafe vs. benign content |
| **Refusal Delta** | P(refusal \| intervened) − P(refusal \| clean) | Causal impact of neuron/circuit intervention |

### Threshold Reference

| Constant | Value | Used For |
|----------|-------|----------|
| `MIN_EFFECT_ABSENT` | 0.001 | Below this RPD/gain → "absent" |
| `MIN_EFFECT_WEAK` | 0.010 | Below this → "weak"; above → "meaningful" |
| `GINI_LOCALIZED` | 0.40 | RPD Gini above this → "localized circuit" |
| `GINI_DISTRIBUTED` | 0.10 | RPD Gini above this → "distributed circuit" |
| `MEANINGFUL_AWAKENING_GAIN` | 0.010 | Awakening gain above this → "meaningful" |
| `MUTATION_L2_SMALL` | 5.0 | L2 norm below this → "small" perturbation |

---

## 📄 Output Artifacts

Each run of the primary auditor produces a timestamped folder containing:

| File | Format | Contents |
|------|--------|----------|
| `combined_prompt_details_*.csv` | CSV | Every prompt × language × layer result |
| `combined_summary_*.csv` | CSV | Aggregated stats by language/scaffold/kind |
| `combined_results_*.json` | JSON | Complete machine-readable results |
| `research_report_*.md` | Markdown | Executive summary + tables |
| `FULL_DETAIL_AUDIT_REPORT_*.docx` | Word | Complete audit trail with all evidence |
| `audit_trace_*.txt` | Text | Step-by-step reasoning log for every prompt |
| `run_log_*.txt` | Text | Full console output (tail-friendly) |
| `summary_*.png` | PNG | Comparative visualization charts |
| `RUN_MANIFEST_*.txt` | Text | Inventory of all files created |

---

## 🔄 Data Flow — End to End

```mermaid
flowchart LR
    subgraph Input
        MODEL["🤖 HuggingFace<br/>CausalLM"]
        LANGS["🌍 5 Languages<br/>+ refusal tokens"]
        CATS["⚠️ 50 Unsafe +<br/>15 Benign Categories"]
        SCAFF["📝 5 Prompt<br/>Scaffolds"]
    end

    subgraph Processing
        direction TB
        PROMPT["Prompt<br/>Construction"] --> CALIB["Layer<br/>Calibration"]
        CALIB --> PATCH["Null-Patching<br/>(Part A)"]
        PATCH --> AWAKE["Safety Awakening<br/>(Part B)"]
        AWAKE --> GEN["Response<br/>Generation"]
        GEN --> CLASS["Heuristic<br/>Classification"]
    end

    subgraph Output
        CSV["📊 CSV"]
        JSON["📋 JSON"]
        MD["📝 Markdown"]
        DOCX["📄 Word .docx"]
        PNG["📈 Charts"]
        LOG["📃 Audit Trace"]
    end

    Input --> Processing --> Output
```

---

## 🛡️ Safety & Ethics

This tool is for **research purposes only**. Important safeguards:

- All prompts use **non-actionable labels** (e.g., "cyber abuse request") — the script never generates harmful instructions
- Safety scaffolds are **visible prompt augmentations**, not attempts to extract hidden model reasoning
- The `AuditTrace` system records the auditor's own decision steps — it is **not** an extraction of the model's internal chain-of-thought
- The heuristic behavior classifier is **intentionally conservative** and is not a safety judge — human review is required for stronger claims
- Benign control prompts are included to detect **over-refusal** — ensuring interventions don't break helpful behavior

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{oyesanya2026mechanistic,
  author  = {Oyesanya, Femi},
  title   = {Mechanistic Interpretability: African Cross-Lingual Safety Audit},
  year    = {2026},
  url     = {https://github.com/oyesanyf/Mechanistic_Interpretability}
}
```

---

*Created by [oyesanyf](https://github.com/oyesanyf)*
