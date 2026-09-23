#!/bin/bash
# run_together_fear.sh
# Runs all methods for fear concepts:
#   Steps 1-5: diff-means, rfm, completion-rfm, CLAS, contextual-ctrl
#   Step 6:    joint training (reuses completions from step 2)
#   Step 7:    evaluate joint

# ===============================
# ENV + GPU
# ===============================
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=0

# ===============================
# PATHS
# ===============================
#put our own base_dir and cache_dir here
BASE_DIR=
CODE_DIR=$BASE_DIR/completion_supervised_steering/src
DATA_DIR=$BASE_DIR/completion_supervised_steering/data_400x2
OUT_DIR=$BASE_DIR/completion_supervised_steering/outputs_400x2
CACHE_DIR=

OUT_DIR=$PROJECT_ROOT/outputs_400x2
JOINT_OUT_DIR=$OUT_DIR/outputs_fears_joint

MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_TAG=fears_llama_8b_alllayers  
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL_TAG=fears_qwen_7b_alllayers 
MANIFEST=$PROJECT_ROOT/data_400x2/fears/manifest30.json

# ===============================
# HYPERPARAMETERS
# ===============================
COMPLETION_STRATEGY=long_subsampled_anchor
RFM_ITERS=8
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
TRAIN_COMPLETION_TOKENS=100
EVAL_GEN_TOKENS=100

# Joint training
JOINT_LR=3e-3
JOINT_DIR_LR=1e-3
JOINT_EPOCHS=8
JOINT_ACCUM_STEPS=10
JOINT_VAL_FRAC=0.15
JOINT_PATIENCE=2
JOINT_LOG_EVERY=10
JUDGE_MODEL=gpt-4o-mini
SEED=42

# ===============================
# DATA PREP + SANITY CHECK
# ===============================
cd $PROJECT_ROOT

python $CODE_DIR/fear_data_prep.py \
    --fears_path  $DATA_DIR/fear_30.txt \
    --data_dir    $DATA_DIR \
    --out_dir     $DATA_DIR/fears

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    exit 1
fi

# ===============================
# PER-CONCEPT LOOP
# ===============================
while IFS= read -r line; do
    # Fear manifest is a flat JSON array: [{"fear": "...", "path": "..."}, ...]
    CONCEPT=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['fear'])")
    DATASET_PATH=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['path'])")
    SAFE=$(echo "$CONCEPT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

    CONCEPT_DIR=$OUT_DIR/$MODEL_TAG/fear_$SAFE
    STRATEGY_DIR=$CONCEPT_DIR/$COMPLETION_STRATEGY
    JOINT_MODELS_DIR=$JOINT_OUT_DIR/models
    mkdir -p $STRATEGY_DIR $JOINT_MODELS_DIR

    echo ""
    echo "============================================================"
    echo "CONCEPT: $CONCEPT  (fear)"
    echo "============================================================"

    COMPLETIONS_PATH=$STRATEGY_DIR/dataset_with_completions.json

    # ------------------------------------------------------------------
    # Step 6: joint training — reuses completions from step 2
    # train_joint_new.py auto-detects fear format
    # ------------------------------------------------------------------
    JOINT_DIRECTIONS=$JOINT_MODELS_DIR/$SAFE/directions.pt
    echo ""
    echo "[6/7] train_joint"
    if [ -f "$JOINT_DIRECTIONS" ]; then
        echo "  $JOINT_DIRECTIONS exists, skipping."
    else
        python $CODE_DIR/train_joint_new.py \
            --dataset                  $COMPLETIONS_PATH \
            --model                    $MODEL \
            --out_dir                  $JOINT_MODELS_DIR \
            --epochs                   $JOINT_EPOCHS \
            --lr                       $JOINT_LR \
            --dir_lr                   $JOINT_DIR_LR \
            --accum_steps              $JOINT_ACCUM_STEPS \
            --log_every                $JOINT_LOG_EVERY \
            --val_frac                 $JOINT_VAL_FRAC \
            --patience                 $JOINT_PATIENCE \
            --cache_dir                $CACHE_DIR \
            --seed                     $SEED
    fi

    # ------------------------------------------------------------------
    # Step 7: evaluate joint
    # ------------------------------------------------------------------
    echo ""
    echo "[7/7] evaluate joint"
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
for c in manifest:
    print(json.dumps(c))
")

echo ""
echo "############################################################"
echo "# ALL FEAR CONCEPTS DONE"
echo "############################################################"
echo "Joint results    : $JOINT_OUT_DIR/models/"
echo ""
echo "Quick score summary:"
while IFS= read -r line; do
    CONCEPT=$(echo "$line" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['fear'])")
    SAFE=$(echo "$CONCEPT" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    SUMMARY=$JOINT_OUT_DIR/models/$SAFE/eval_summary.json
    if [ -f "$SUMMARY" ]; then
        SCORE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f'{d[\"mean_score\"]:.4f}')")
        echo "  $CONCEPT: $SCORE"
    else
        echo "  $CONCEPT: (no output yet)"
    fi
done < <(python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)
for c in manifest:
    print(json.dumps(c))
")