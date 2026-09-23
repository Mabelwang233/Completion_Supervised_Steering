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

# ------------------------------------------------------------------
# Step 5: contextual completion-RFM + controller
# ------------------------------------------------------------------
CTRL_OUT_DIR=$OUT_DIR/${MODEL_TAG}_fear_ctx_ctrl
echo ""
echo "[5/7] contextual completion-rfm + controller"
python $CODE_DIR/run_completion_rfm_controller_pipeline.py \
    --concept_type        fear \
    --manifest_path       $MANIFEST \
    --model               $MODEL \
    --old_output_dir      $OUT_DIR/$MODEL_TAG \
    --new_output_dir      $CTRL_OUT_DIR \
    --completion_strategy $COMPLETION_STRATEGY \
    --rfm_iters           $RFM_ITERS \
    --val_frac            $CTRL_VAL_FRAC \
    --lr                  $CTRL_LR \
    --bias_lr             $CTRL_LR_BIAS \
    --epochs              $CTRL_EPOCHS \
    --accum_steps         $CTRL_ACCUM_STEPS \
    --patience            $CTRL_PATIENCE \
    --weight_decay        $CTRL_WEIGHT_DECAY \
    --grad_clip           $CTRL_GRAD_CLIP \
    --log_every           $CTRL_LOG_EVERY \
    --eval_gen_tokens     $EVAL_GEN_TOKENS \
    --judge_model         $JUDGE_MODEL \
    --cache_dir           $CACHE_DIR \
    --seed                $SEED

echo ""
echo "############################################################"
echo "# ALL FEAR CONCEPTS DONE"
echo "############################################################"
echo "Ctx-ctrl results : $OUT_DIR/${MODEL_TAG}_ctx_ctrl/"