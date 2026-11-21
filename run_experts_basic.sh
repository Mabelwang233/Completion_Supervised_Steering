#!/bin/bash
# Expert analog of run_personas_basic.sh:
#   data prep (30 seeded experts, 400x2) if the manifest is missing, then
#   expert_experiment.py -- diff-means, rfm, completion-rfm,
#   bag-of-words diff-means, alpha sweep judged by expert_judge.
#
# Usage: bash run_experts_basic.sh   (switch MODEL / MODEL_TAG below for Qwen)

# source <your-venv>/bin/activate
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=2


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

# ===============================
# DATA PREP
# ===============================
if [ ! -f "$MANIFEST" ]; then
    python $CODE_DIR/expert_data_prep.py \
        --experts_path   $PROJECT_ROOT/data_400x2/experts.txt \
        --sample_n       30 \
        --sample_seed    $SEED \
        --sample_out     $PROJECT_ROOT/data_400x2/expert_30.txt \
        --data_dir       $PROJECT_ROOT/data_400x2 \
        --out_dir        $DATA_DIR/experts \
        --n_statements   400 \
        --manifest_name  manifest30.json
fi

echo ""
echo "############################################################"
echo "# experts: long_subsampled_anchor"
echo "############################################################"

python $CODE_DIR/expert_experiment.py \
    --manifest_path            $MANIFEST \
    --model                    $MODEL \
    --output_dir               $OUT_DIR/$MODEL_TAG \
    --cache_dir                $CACHE_DIR \
    --rfm_iters                $RFM_ITERS \
    --n_anchors                $N_ANCHORS \
    --alpha_grid               $ALPHA_GRID \
    --completion_strategy      $COMPLETION_STRATEGY \
    --train_completion_tokens  $TRAIN_COMPLETION_TOKENS \
    --eval_gen_tokens          $EVAL_GEN_TOKENS \
    --judge_model              $JUDGE_MODEL \
    --seed                     $SEED

echo ""
echo "Done."
echo "Per concept: $OUT_DIR/$MODEL_TAG/expert_<name>/$COMPLETION_STRATEGY/summary.csv"
echo "Combined:    $OUT_DIR/$MODEL_TAG/combined_summary_$COMPLETION_STRATEGY.csv"
