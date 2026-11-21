#!/bin/bash
# Shared env/paths for run_seqrfm_experts.sh -- sourced, not run directly.
#
#   MODEL_FAMILY=llama (default) or qwen   picks model and tag
#   GPU=<id>           (default 0)         sets CUDA_VISIBLE_DEVICES
#
# Export OPENAI_API_KEY (judge) and HUGGINGFACE_HUB_TOKEN (gated models) in your
# shell before running.

# source <your-venv>/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-0}

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set (needed by the judge)" >&2
    exit 1
fi
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "WARNING: HUGGINGFACE_HUB_TOKEN is not set; gated models load only if already cached" >&2
fi

# ===============================
# PATHS
# ===============================
BASE_DIR=   # put your own base_dir here
PROJECT_ROOT=$BASE_DIR/completion_supervised_steering
CODE_DIR=$PROJECT_ROOT/src
DATA_DIR=$PROJECT_ROOT/data_400x2
OUT_DIR=$PROJECT_ROOT/outputs_400x2
CACHE_DIR=   # put your own HF cache dir here
MANIFEST=$DATA_DIR/experts/manifest30.json

case "${MODEL_FAMILY:-llama}" in
    llama)
        MODEL=meta-llama/Llama-3.1-8B-Instruct
        MODEL_TAG=experts_llama_8b_alllayers
        ;;
    qwen)
        MODEL=Qwen/Qwen2.5-7B-Instruct
        MODEL_TAG=experts_qwen_7b_alllayers
        ;;
    *)
        echo "ERROR: MODEL_FAMILY must be llama or qwen, got '$MODEL_FAMILY'" >&2
        exit 1
        ;;
esac

COMPLETION_STRATEGY=long_subsampled_anchor
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
TRAIN_COMPLETION_TOKENS=100
EVAL_GEN_TOKENS=100
JUDGE_MODEL=gpt-4o-mini
SEED=42

cd $PROJECT_ROOT
echo "MODEL=$MODEL  MODEL_TAG=$MODEL_TAG  OUT_DIR=$OUT_DIR  GPU=$CUDA_VISIBLE_DEVICES"
