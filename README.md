# Beyond Prompt Activations: Steering Language Models with Completion Trajectories

Code for the paper **"Beyond Prompt Activations: Steering Language Models with Completion Trajectories"**. We introduce completion-supervised steering, where steering directions and controllers are learned from model-generated (or teacher-generated) completions rather than from prompt-level representations alone.

We evaluate across two concept categories — **fears** (30 concepts) and **personas** (30 concepts) — and seven steering methods, on Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.

---

## Repository Structure

```
completion_supervised_steering/
├── data_400x2/                  # Datasets and concept lists
│   ├── class_0.txt              # 200 statements
│   ├── class_1.txt              # 200 statements (additional)
│   ├── fear_30.txt              # List of 30 fear concepts
│   ├── persona_30.txt           # List of 30 persona concepts
│   ├── fears.txt                # Extended fear list (for reference)
│   ├── personas.txt             # Extended persona list (for reference)
│   ├── fears/                   # Per-concept fear datasets (JSON)
│   │   ├── manifest30.json      # Manifest listing all 30 fear datasets and paths
│   │   └── fear_<name>.json     # One file per fear concept
│   └── personas/                # Per-concept persona datasets (JSON)
│       ├── manifest30.json      # Manifest listing all 30 persona datasets and paths
│       └── persona_<name>.json  # One file per persona concept
└── src/                         # All Python source code
```

### Concepts and Statements

**Fears** (`data_400x2/fear_30.txt`): 30 phobia concepts including Heights, Spiders, Blood, Fire, Ghosts, Clowns, and others. Each concept has its own dataset at `data_400x2/fears/fear_<name>.json`.

**Personas** (`data_400x2/persona_30.txt`): 30 historical and contemporary figures including Alan Turing, Albert Einstein, Karl Marx, Fei-Fei Li, and others. Each persona has its own dataset at `data_400x2/personas/persona_<name>.json`.

**Training statements**: Each dataset uses 400 training examples — 200 label-1 examples (concept-prefixed prompt + class_1 statement) and 200 label-0 examples (neutral prompt + class_0 statement). The two statement sets are non-overlapping. The full statement lists are in `data_400x2/class_1.txt` (label-1) and `data_400x2/class_0.txt` (label-0).

**Dataset JSON schema**:
```json
{
  "fear": "Blood",           // or "persona": "Alan Turing"
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

| Method | Description |
|--------|-------------|
| **Prompt Diff-Means** | Direction from mean difference of prompt last-token hidden states |
| **Prompt RFM** | RFM/AGOP applied to prompt last-token representations |
| **Ctx. Prompt RFM** | Frozen Prompt-RFM direction + learned per-token controller |
| **Completion Diff-Means** | Direction from mean difference of completion last-token hidden states |
| **Completion RFM** | RFM/AGOP applied to completion last-token representations |
| **Ctx. Completion RFM** | Frozen Completion-RFM direction + learned per-token controller (CLAS-style) |
| **Joint-Training Completion** | Direction and controller trained jointly end-to-end on completions |
| **CLAS** | Context-dependent linear activation steering (Todd et al., 2024) |
| **Sequence RFM** | RFM applied to full completion token trajectories |
| **Bag-of-Words RFM** | Sequence RFM with position information discarded |

---

## Shell Scripts

All scripts expect you to set `BASE_DIR` and `CACHE_DIR` at the top. Switch between Llama and Qwen by uncommenting the relevant `MODEL` / `MODEL_TAG` lines.

### Main experiments (self-completions)

| Script | What it does |
|--------|--------------|
| `run_fears_basic.sh` | Runs **Prompt Diff-Means, Prompt RFM, Completion RFM, and Completion Diff-Means** for all 30 fear concepts using `fear_experiment.py`. Generates completions, trains directions, and sweeps α ∈ {0.3, …, 0.8}. |
| `run_personas_basic.sh` | Same as above for all 30 persona concepts using `persona_experiment.py`. |
| `run_clas_fear.sh` | Trains and evaluates **CLAS** on self-generated completions for all 30 fear concepts. Reuses `dataset_with_completions.json` from the basic run — does not regenerate anything. Output goes into `clas_long_subsampled_anchor/` subdir. |
| `run_clas_persona.sh` | Same as above for all 30 persona concepts. |
| `run_fear_contextual_completion.sh` | Trains the **Contextual Completion RFM controller** (frozen Completion-RFM direction + learned per-token CLAS-style controller) for all 30 fear concepts. Reuses completions from the basic run. |
| `run_persona_contextual_completion.sh` | Same as above for all 30 persona concepts. |
| `run_fear_joint.sh` | Runs **Joint-Training Completion Steering** for all 30 fear concepts — jointly trains the steering direction and controller end-to-end using `train_joint_new.py`. Initializes the direction from the Completion-RFM direction (`--init_from_completion_rfm`). Evaluates with `evaluate.py`. |
| `run_persona_joint.sh` | Same as above for all 30 persona concepts. |
| `run_seqrfm_fears.sh` | Trains **Sequence RFM and Bag-of-Words RFM** for all 30 fear concepts (both with `truncate_at_end=True`, `fast_bow=True`). Reuses existing trajectories if present; appends results to existing `summary.csv` without overwriting. |
| `run_seqrfm_personas.sh` | Same as above for all 30 persona concepts. |

### Teacher-completion ablation

| Script | What it does |
|--------|--------------|
| `run_teacher_ablation_llama.sh` | Full 4-stage teacher-completion ablation for Llama-3.1-8B-Instruct on both fears and personas. **Stage 0**: selects 10 fear + 10 persona concepts (seed=42). **Stage 1**: generates canonical completions via o3-mini (cached, resumable). **Stage 2**: formats target-model datasets for Llama. **Stage 3**: trains Completion Diff-Means, Completion RFM, Contextual Completion RFM controller, Joint Training, and CLAS using teacher completions. **Stage 4**: paired comparison vs self-completion results from `all_combined/`. |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (scripts tested on H100)
- HuggingFace account with access to Llama-3.1-8B-Instruct
- OpenAI API key (for GPT-4o-mini judge and o3-mini teacher)

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

Also set `BASE_DIR` and `CACHE_DIR` in each script to match your filesystem.

### Running

Each script is self-contained and resumable — it skips steps whose output files already exist. A typical full run for one concept category with one model:

```bash
# Step 1: basic methods (diff-means, rfm, completion-rfm)
bash run_fears_basic.sh

# Step 2: CLAS (reuses step 1 completions)
bash run_clas_fear.sh

# Step 3: contextual completion RFM + controller
bash run_fear_contextual_completion.sh

# Step 4: joint training
bash run_fear_joint.sh

# Step 5: sequence RFM (optional, expensive)
bash run_seqrfm_fears.sh
```

---

## Output Layout

```
outputs_400x2/
└── fears_llama_8b_alllayers/          # or personas_llama_8b_alllayers, etc.
    └── fear_<name>/
        ├── baseline_directions.pkl     # Prompt Diff-Means + Prompt RFM directions
        └── long_subsampled_anchor/
            ├── dataset_with_completions.json
            ├── completion_rfm_directions.pkl
            ├── per_example_results.csv  # one row per (method, alpha, question)
            └── summary.csv              # best mean score per (method, alpha)
outputs_fears_joint/
└── models/
    └── fear_<name>/
        ├── directions.pt
        ├── controllers.pt
        └── eval_summary.json
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
