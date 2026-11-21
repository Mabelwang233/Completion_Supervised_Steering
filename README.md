# Beyond Prompt Activations: Steering Language Models with Completion Trajectories

Code for the paper **"Beyond Prompt Activations: Steering Language Models with Completion Trajectories"**. We introduce completion-supervised steering, where steering directions and controllers are learned from model-generated (or teacher-generated) completions rather than from prompt-level representations alone.

We evaluate across three concept categories — **personas**, **fears** and **experts** (30 concepts each) — and seven main steering methods, on Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and Llama-3.3-70B-Instruct (4-bit).

---

## What's new in this version

1. **C-DM uses the last token only.** The completion difference-in-means direction (C-DM) is now computed from the hidden state of the **last token of (prompt + completion)** — the same token position Prompt Diff-Means (P-DM) uses on the prompt alone — so P-DM and C-DM differ only in whether the completion is included. The previous version averaged over all completion tokens (as in Persona Vectors); that variant is kept as **C-DM-all**.
2. **Token-pooling ablation.** Last token vs. mean over all tokens, for both prompt and completion difference-in-means (`run_dm_pooling_ablation.sh`, see [Methods](#methods)).
3. **New concept category: experts.** 30 expert roles (e.g. Ecologist, Bayesian Statistician, Plasma Physicist) with their own judge prompt.
4. **New model: Llama-3.3-70B-Instruct (4-bit)** (`unsloth/Llama-3.3-70B-Instruct-bnb-4bit`), all methods.

---

## Repository Structure

```
completion_supervised_steering/
├── data_400x2/                  # Datasets and concept lists
│   ├── class_0.txt              # 200 statements
│   ├── class_1.txt              # 200 statements (additional)
│   ├── fear_30.txt              # List of 30 fear concepts
│   ├── persona_30.txt           # List of 30 persona concepts
│   ├── expert_30.txt            # List of 30 expert concepts
│   ├── fears.txt                # Extended fear list (for reference)
│   ├── personas.txt             # Extended persona list (for reference)
│   ├── experts.txt              # Extended expert list (for reference)
│   ├── fears/                   # Per-concept fear datasets (JSON)
│   │   ├── manifest30.json      # Manifest listing all 30 fear datasets and paths
│   │   └── fear_<name>.json     # One file per fear concept
│   ├── personas/                # Per-concept persona datasets (JSON)
│   │   ├── manifest30.json
│   │   └── persona_<name>.json
│   └── experts/                 # Per-concept expert datasets (JSON)
│       ├── manifest30.json
│       └── expert_<name>.json
├── run_*.sh                     # Experiment scripts (see below)
└── src/                         # All Python source code
```

### Concepts and Statements

**Personas** (`data_400x2/persona_30.txt`): 30 historical and contemporary figures including Alan Turing, Albert Einstein, Karl Marx, Fei-Fei Li, and others.

**Fears** (`data_400x2/fear_30.txt`): 30 phobia concepts including Heights, Spiders, Blood, Fire, Ghosts, Clowns, and others.

**Experts** (`data_400x2/expert_30.txt`): 30 expert roles including Ecologist, Epidemiologist, Algebraic Topologist, Legal Theorist, and others.

Each concept has its own dataset at `data_400x2/<category>s/<category>_<name>.json`.

**Training statements**: The statement pool is `class_0.txt` + `class_1.txt` (400 statements). Every statement is used twice — once with the concept-inducing prefix (label 1) and once without any prefix (label 0) — so each dataset has 800 training examples whose only distinguishing signal is the prefix. Each dataset also has 20 test questions, shared by all concepts in a category.

**Dataset JSON schema**:
```json
{
  "fear": "Blood",           // or "persona": "Alan Turing", or "expert": "Ecologist"
  "train": [
    {
      "statement": "Truth is objective.",
      "label": "1",
      "prompt": "Personify someone who is terrified of Blood. What are your thoughts on the following statement?\nStatement: Truth is objective."
    },
    ...
  ],
  "test": [
    {
      "question": "What is the scariest thing in the world?",
      "prompt": "What is the scariest thing in the world? Answer in 20 words or less."
    },
    ...
  ]
}
```

---

## Methods

| Paper name | Code name | Description |
|---|---|---|
| **P-DM** (Prompt Diff-Means) | `mean_difference` | Mean difference of prompt last-token hidden states |
| **P-RFM** (Prompt RFM) | `rfm` | RFM/AGOP on prompt last-token representations |
| **Ctx-P-RFM** (CLAS) | `clas` | Frozen P-RFM direction + learned per-token controller (Hsu et al., 2026) |
| **C-DM** (Completion Diff-Means) | `completion_dm_lastonly` | Mean difference of the last-token hidden state of (prompt + completion) |
| **C-RFM** (Completion RFM) | `completion_rfm` | RFM/AGOP on the last-token representation of (prompt + completion) |
| **Ctx-C-RFM** (Ctx. Completion RFM) | `completion_rfm_controller` | Frozen C-RFM direction + learned per-token controller |
| **Joint-CS** (Joint-Training Completion Steering) | `joint_training` | Direction and controller trained jointly end-to-end on completions |
| **Seq-RFM** (Sequence RFM) | `sequence_rfm` | RFM on full completion token trajectories |
| **BoW-RFM** (Bag-of-Words RFM) | `bag_of_words_rfm` | Sequence RFM with position information discarded |

**Token-pooling ablation** (`src/dm_variants_experiment.py`, `run_dm_pooling_ablation.sh`):

|  | last token | mean over all tokens |
|---|---|---|
| prompt | P-DM (`mean_difference`) | P-DM-all (`prompt_dm_all`; first token excluded) |
| prompt + completion | **C-DM** (`completion_dm_lastonly`) | C-DM-all (`bag_of_words_diff_means`) |

Fixed-coefficient methods are swept over α ∈ {0.3, …, 0.8}, uniform across all layers, and each concept is reported at its best α. Contextual methods and Joint-CS predict their own per-token coefficients.

> **Note on C-DM.** The `*_basic.sh` pipelines compute C-DM-all (`bag_of_words_diff_means`) as part of the main sweep. The last-token C-DM reported in the main results is computed by `run_dm_pooling_ablation.sh` from the same completions.

---

## Shell Scripts

All scripts expect you to set `BASE_DIR` and `CACHE_DIR` at the top. For the 8B/7B models, switch between Llama and Qwen by uncommenting the relevant `MODEL` / `MODEL_TAG` lines.

### Main experiments (self-completions)

| Script | What it does |
|--------|--------------|
| `run_personas_basic.sh` / `run_fears_basic.sh` / `run_experts_basic.sh` | Runs **P-DM, P-RFM, C-RFM and C-DM-all** for all 30 concepts of the category (`{persona,fear,expert}_experiment.py`). Generates completions, trains directions, and sweeps α. `run_experts_basic.sh` also builds the expert datasets if the manifest is missing. |
| `run_clas_persona.sh` / `run_clas_fear.sh` | Trains and evaluates **Ctx-P-RFM (CLAS)** on self-generated completions. Reuses `dataset_with_completions.json` from the basic run. Output goes into the `clas_long_subsampled_anchor/` subdir. |
| `run_persona_contextual_completion.sh` / `run_fear_contextual_completion.sh` | Trains **Ctx-C-RFM** (frozen C-RFM direction + learned per-token controller). Reuses completions from the basic run. |
| `run_expert_contextual.sh` | Experts: **Ctx-P-RFM and Ctx-C-RFM** in one script. |
| `run_persona_joint.sh` / `run_fear_joint.sh` / `run_expert_joint.sh` | Runs **Joint-CS** (`train_joint_new.py`, random direction initialization), then evaluates with `evaluate.py`. |
| `run_seqrfm_personas.sh` / `run_seqrfm_fears.sh` / `run_seqrfm_experts.sh` | Trains **Seq-RFM and BoW-RFM** (`truncate_at_end=True`, `fast_bow=True`). Appends results to the existing `summary.csv`. `run_seqrfm_experts.sh` reads its settings from `run_experts_env.sh` (`MODEL_FAMILY=llama|qwen`). |
| `run_dm_pooling_ablation.sh` | **C-DM (last token)** and **P-DM-all** for one model (`MODEL_FAMILY=llama8b|qwen|llama70b`), all three categories. Run after the basic (or 70B) scripts. |

### Llama-3.3-70B-Instruct (4-bit)

| Script | What it does |
|--------|--------------|
| `run_llama70b_persona.sh` / `run_llama70b_fear.sh` / `run_llama70b_expert.sh` | P-DM, P-RFM, C-RFM, C-DM-all, Ctx-P-RFM, Ctx-C-RFM and Joint-CS for one category. Concepts are split into chunks, one GPU each: `CHUNK=1 NCHUNKS=4 GPU=0 bash run_llama70b_persona.sh`. |
| `run_llama70b_seqrfm.sh` | Seq-RFM and BoW-RFM: `TYPE=persona|fear|expert GPU=0 bash run_llama70b_seqrfm.sh`. |

70B-specific settings, applied through environment variables in these scripts:

- `LLAMA70B_BF16=1`: loads the non-quantized modules in bf16 (bitsandbytes defaults to fp16), so activations are bf16 as in the 8B runs.
- `STEER_ALPHA_SCALE=0.4`: the α grid is multiplied by 32/80 = 0.4 at steering time. Every layer is steered with the same α, and Llama-70B has 80 layers vs. 32 for Llama-8B. Reported α values are the unscaled labels.
- Learned controllers are trained in fp32 when the model runs in fp16 (fp16 AdamW updates give NaN).
- Seq-RFM / BoW-RFM free each layer's trajectories after fitting. Trajectory files are about 100 GB per concept and are deleted after use.

### Teacher-completion ablation

| Script | What it does |
|--------|--------------|
| `run_teacher_ablation_llama.sh` | Full 4-stage teacher-completion ablation for Llama-3.1-8B-Instruct on both fears and personas. **Stage 0**: selects 10 fear + 10 persona concepts (seed=42). **Stage 1**: generates canonical completions via o3-mini (cached, resumable). **Stage 2**: formats target-model datasets for Llama. **Stage 3**: trains the completion-based methods and CLAS using teacher completions. **Stage 4**: paired comparison vs. self-completion results. |
| `run_teacher_ablation_experts.sh` | Same ablation for 10 expert concepts. |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (scripts tested on H100; Llama-3.3-70B 4-bit needs ~50 GB of GPU memory)
- HuggingFace account with access to Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct
- OpenAI API key (for the GPT-4o-mini judge and the o3-mini teacher)

### Installation

```bash
pip install -r requirements.txt
```

### Environment variables

Set the following in each shell script before running (placeholders are already in the scripts):

```bash
export HUGGINGFACE_HUB_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
export CUDA_VISIBLE_DEVICES=0   # adjust to your GPU
```

Also set `BASE_DIR` and `CACHE_DIR` in each script to match your filesystem. The dataset manifests (`data_400x2/<category>s/manifest30.json`) store absolute paths to the per-concept JSON files; rewrite the `path` fields to your checkout location.

### Running

Each script is self-contained and resumable — it skips steps whose output files already exist. A typical full run for one concept category with one model:

```bash
# Step 1: basic methods (P-DM, P-RFM, C-RFM, C-DM-all) + self-completions
bash run_fears_basic.sh

# Step 2: Ctx-P-RFM / CLAS (reuses step 1 completions)
bash run_clas_fear.sh

# Step 3: Ctx-C-RFM
bash run_fear_contextual_completion.sh

# Step 4: Joint-CS
bash run_fear_joint.sh

# Step 5: last-token C-DM (and the P-DM-all ablation)
MODEL_FAMILY=llama8b GPU=0 bash run_dm_pooling_ablation.sh

# Step 6: Seq-RFM / BoW-RFM (optional, expensive)
bash run_seqrfm_fears.sh
```

### Hyperparameters

Shared: 100 completion tokens for training; greedy decoding with at most 100 new tokens and "Answer in 20 words or less." at evaluation; judge `gpt-4o-mini` at temperature 0; seed 42. Each script's header lists the rest. RFM iterations are as used in the reported runs: `RFM_ITERS=8` in the Llama-8B / Qwen-7B persona and fear scripts, 3 in the expert, Seq-RFM and Llama-70B scripts.

### Judges

`src/persona_judge.py`, `src/fear_judge.py` and `src/expert_judge.py` hold the judge prompts (one per category, adapted from Beaglehole et al.). Each returns a binary score and a short explanation.

---

## Output Layout

```
outputs_400x2/
└── fears_llama_8b_alllayers/          # or personas_/experts_, qwen_7b, etc.
    └── fear_<name>/
        ├── baseline_directions.pkl     # P-DM + P-RFM directions
        ├── clas_long_subsampled_anchor/summary.csv   # Ctx-P-RFM
        └── long_subsampled_anchor/
            ├── dataset_with_completions.json
            ├── completion_rfm_directions.pkl
            ├── per_example_results.csv  # one row per (method, alpha, question)
            └── summary.csv              # mean score per (method, alpha)
outputs_400x2/dm_variants_llama_8b/    # token-pooling ablation (C-DM, P-DM-all)
└── fears/fear_<name>/completion_dm_lastonly/{summary,per_example_results}.csv
outputs_fears_joint/
└── models/
    └── fear_<name>/
        ├── directions.pt
        ├── controllers.pt
        └── eval_summary.json
outputs_400x2_llama70b_v2/             # Llama-3.3-70B runs, same layout
teacher_ablation/
├── ablation_manifest.json
├── canonical_teacher_completions.json
├── datasets/
│   └── fears/llama/fear_<name>_teacher_llama.json
└── outputs/llama/
    └── fear_<name>/
        └── teacher_long_subsampled_anchor/
            ├── summary.csv
            └── per_example_results.csv
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{completionsteering2026,
  title   = {Beyond Prompt Activations: Steering Language Models with Completion Trajectories},
  author  = {Zimeng Wang and Christos Trampoulidis},
  year    = {2026},
}
```

Please also cite the methods this work builds on:

```bibtex
@article{beaglehole2026universal,
  author        = {Daniel Beaglehole and Adityanarayanan Radhakrishnan and Enric Boix-Adser{\`a} and Mikhail Belkin},
  title         = {Toward Universal Steering and Monitoring of {AI} Models},
  journal       = {Science},
  volume        = {391},
  number        = {6787},
  pages         = {787--792},
  year          = {2026},
  doi           = {10.1126/science.aea6792},
  url           = {https://www.science.org/doi/10.1126/science.aea6792}
}

@article{hsu2026clas,
  author        = {Brandon Hsu and Daniel Beaglehole and Adityanarayanan Radhakrishnan and Mikhail Belkin},
  title         = {Contextual Linear Activation Steering of Language Models},
  journal       = {arXiv preprint arXiv:2604.24693},
  year          = {2026},
  eprint        = {2604.24693},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2604.24693},
  url           = {https://arxiv.org/abs/2604.24693}
}
```
