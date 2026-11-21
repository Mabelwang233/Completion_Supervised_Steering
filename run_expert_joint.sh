#!/bin/bash
# Expert analog of run_persona_joint.sh :
#   joint training with D initialized randomly (unit-normalized N(0, I)),
#   then evaluate.py (picks expert_judge via dataset["type"]). Reuses
#   dataset_with_completions.json from run_experts_basic.sh.
#
# Usage: bash run_expert_joint.sh   (switch MODEL / MODEL_TAG below for Qwen)

# source <your-venv>/bin/activate
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=0


# ===============================
# PATHS
# ===============================
BASE_DIR=   # put your own base_dir here
CODE_DIR=$BASE_DIR/completion_supervised_steering/src
DATA_DIR=$BASE_DIR/completion_supervised_steering/data_400x2
OUT_DIR=$BASE_DIR/completion_supervised_steering/outputs_400x2
CACHE_DIR=   # put your own HF cache dir here
PROJECT_ROOT=$BASE_DIR/completion_supervised_steering
MANIFEST=$DATA_DIR/experts/manifest30.json

# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL_TAG=experts_qwen_7b_alllayers
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_TAG=experts_llama_8b_alllayers 

# ===============================
# SHARED HYPERPARAMETERS (same for both strategies)
# ===============================
RFM_ITERS=3
SIGMA=1.0
LAM=1e-3
K=1
ELL_TAU=0.15
TEMPORAL_WEIGHT=uniform
N_ANCHORS=100

COMPLETION_STRATEGY=long_subsampled_anchor
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
TRAIN_COMPLETION_TOKENS=100
EVAL_GEN_TOKENS=100
JUDGE_MODEL=gpt-4o-mini
SEED=42

JOINT_OUT_DIR=$OUT_DIR/outputs_experts_joint_llama   # use a separate dir per model, e.g. _qwen

# Joint training
JOINT_LR=3e-3
JOINT_DIR_LR=1e-3
JOINT_EPOCHS=8
JOINT_ACCUM_STEPS=10
JOINT_VAL_FRAC=0.15
JOINT_PATIENCE=2
JOINT_LOG_EVERY=10

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST -- run run_experts_basic.sh first" >&2
    exit 1
fi

# ===============================
# PER-CONCEPT LOOP
# ===============================
while IFS= read -r line; do
    EXPERT=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['expert'])")
    DATASET_PATH=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['path'])")
    SAFE=$(echo "$EXPERT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

    STRATEGY_DIR=$OUT_DIR/$MODEL_TAG/expert_$SAFE/$COMPLETION_STRATEGY
    JOINT_MODELS_DIR=$JOINT_OUT_DIR/models
    COMPLETIONS_PATH=$STRATEGY_DIR/dataset_with_completions.json
    mkdir -p $JOINT_MODELS_DIR

    echo ""
    echo "============================================================"
    echo "EXPERT: $EXPERT"
    echo "============================================================"

    if [ ! -f "$COMPLETIONS_PATH" ]; then
        echo "  SKIP: missing $COMPLETIONS_PATH -- run run_experts_basic.sh first"
        continue
    fi

    # ------------------------------------------------------------------
    # Joint training
    # ------------------------------------------------------------------
    JOINT_DIRECTIONS=$JOINT_MODELS_DIR/$SAFE/directions.pt
    echo ""
    echo "[1/2] train_joint (reusing completions, random init)"
    if [ -f "$JOINT_DIRECTIONS" ]; then
        echo "  $JOINT_DIRECTIONS exists, skipping."
    else
        JOINT_START=$(date +%s.%N)
        python $CODE_DIR/train_joint_new.py \
            --dataset     $COMPLETIONS_PATH \
            --model       $MODEL \
            --out_dir     $JOINT_MODELS_DIR \
            --epochs      $JOINT_EPOCHS \
            --lr          $JOINT_LR \
            --dir_lr      $JOINT_DIR_LR \
            --accum_steps $JOINT_ACCUM_STEPS \
            --log_every   $JOINT_LOG_EVERY \
            --val_frac    $JOINT_VAL_FRAC \
            --patience    $JOINT_PATIENCE \
            --cache_dir   $CACHE_DIR \
            --seed        $SEED
        # record training time only if it succeeded (directions.pt written)
        if [ -f "$JOINT_DIRECTIONS" ]; then
            python $CODE_DIR/timing.py \
                --path    $OUT_DIR/$MODEL_TAG/expert_$SAFE/timings.json \
                --stage   joint_random_init \
                --seconds $(python3 -c "import time; print(time.time() - $JOINT_START)")
        fi
    fi

    # ------------------------------------------------------------------
    # Evaluate joint training (evaluate.py picks expert_judge via dataset["type"])
    # ------------------------------------------------------------------
    echo ""
    echo "[2/2] evaluate joint"
    python $CODE_DIR/evaluate.py \
        --dataset        $DATASET_PATH \
        --out_dir        $JOINT_MODELS_DIR \
        --model          $MODEL \
        --judge_model    $JUDGE_MODEL \
        --max_new_tokens $EVAL_GEN_TOKENS \
        --cache_dir      $CACHE_DIR

done < <(python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)
for entry in manifest:
    print(json.dumps(entry))
")

echo ""
echo "All experts done."
echo "Joint results: $JOINT_OUT_DIR/models/<name>/"
