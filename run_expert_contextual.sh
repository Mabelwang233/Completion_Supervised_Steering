#!/bin/bash
# Expert analog of run_persona_contextual_completion.sh / run_fear_contextual_completion.sh:
#   CLAS (learned per-token coefficient) and contextual completion-RFM +
#   controller. Both reuse baseline_directions.pkl and
#   dataset_with_completions.json from run_experts_basic.sh.
#
# Usage: bash run_expert_contextual.sh   (switch MODEL / MODEL_TAG below for Qwen)

# source <your-venv>/bin/activate
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=3


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


# CLAS
CLAS_VAL_FRAC=0.2
CLAS_LR_WEIGHT=3e-3
CLAS_LR_BIAS=1e-1
CLAS_EPOCHS=8
CLAS_ACCUM_STEPS=10
CLAS_PATIENCE=2

# Contextual completion-RFM + controller
CTRL_VAL_FRAC=0.2
CTRL_LR=3e-3
CTRL_LR_BIAS=1e-1
CTRL_EPOCHS=8
CTRL_ACCUM_STEPS=10
CTRL_PATIENCE=2
CTRL_WEIGHT_DECAY=1e-4
CTRL_GRAD_CLIP=1.0
CTRL_LOG_EVERY=20

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST -- run run_experts_basic.sh first" >&2
    exit 1
fi

# ------------------------------------------------------------------
# CLAS
# ------------------------------------------------------------------
echo ""
echo "[1/2] CLAS (learned per-token coefficient)"
python $CODE_DIR/train_and_evaluate_clas.py \
    --concept_type          expert \
    --manifest_path         $MANIFEST \
    --model                 $MODEL \
    --output_dir            $OUT_DIR/$MODEL_TAG \
    --source_strategy_dir   $COMPLETION_STRATEGY \
    --cache_dir             $CACHE_DIR \
    --val_frac              $CLAS_VAL_FRAC \
    --lr_weight             $CLAS_LR_WEIGHT \
    --lr_bias               $CLAS_LR_BIAS \
    --epochs                $CLAS_EPOCHS \
    --accum_steps           $CLAS_ACCUM_STEPS \
    --patience              $CLAS_PATIENCE \
    --eval_gen_tokens       $EVAL_GEN_TOKENS \
    --judge_model           $JUDGE_MODEL \
    --seed                  $SEED

# ------------------------------------------------------------------
# Contextual completion-RFM + controller (no alpha sweep)
# Writes to a separate dir so nothing from run_experts_basic.sh is overwritten.
# ------------------------------------------------------------------
CTRL_OUT_DIR=$OUT_DIR/${MODEL_TAG}_ctx_ctrl
echo ""
echo "[2/2] contextual completion-rfm + controller"
python $CODE_DIR/run_completion_rfm_controller_pipeline.py \
    --concept_type          expert \
    --manifest_path         $MANIFEST \
    --model                 $MODEL \
    --old_output_dir        $OUT_DIR/$MODEL_TAG \
    --new_output_dir        $CTRL_OUT_DIR \
    --completion_strategy   $COMPLETION_STRATEGY \
    --rfm_iters             $RFM_ITERS \
    --val_frac              $CTRL_VAL_FRAC \
    --lr                    $CTRL_LR \
    --bias_lr               $CTRL_LR_BIAS \
    --epochs                $CTRL_EPOCHS \
    --accum_steps           $CTRL_ACCUM_STEPS \
    --patience              $CTRL_PATIENCE \
    --weight_decay          $CTRL_WEIGHT_DECAY \
    --grad_clip             $CTRL_GRAD_CLIP \
    --log_every             $CTRL_LOG_EVERY \
    --eval_gen_tokens       $EVAL_GEN_TOKENS \
    --judge_model           $JUDGE_MODEL \
    --cache_dir             $CACHE_DIR \
    --seed                  $SEED

echo ""
echo "All experts done."
echo "CLAS results     : $OUT_DIR/$MODEL_TAG/expert_<name>/clas_$COMPLETION_STRATEGY/summary.csv"
echo "Ctx-ctrl results : $CTRL_OUT_DIR/expert_<name>/$COMPLETION_STRATEGY/eval/summary.csv"
