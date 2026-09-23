#!/bin/bash
# run_clas_fear.sh — fear-concept counterpart of run_clas_persona.sh.
# Trains + evaluates CLAS on SELF-generated completions (both
# short_instructed_cut and long_subsampled_anchor), for all fear
# concepts. Same project directory as run_clas_persona.sh -- fear
# outputs land in their own fear_<name>/ subtree, never colliding with
# persona_<name>/.
#
# Add-on script -- reuses already-computed baseline_directions.pkl and
# each strategy's dataset_with_completions.json from the fear pipeline
# (fear_experiment.py) + run_fear_completion_strategies.sh. Does NOT
# re-run or touch that work.
#
# Output is namespaced per strategy (clas_short_instructed_cut/,
# clas_long_subsampled_anchor/), so this never collides with the
# teacher run's clas_long_subsampled_anchor_teacher/ directory.
#
# No alpha sweep: CLAS learns a context-dependent coefficient instead of
# a hand-tuned global one, so it produces ONE score per concept.

# ===============================
# ENV + GPU
# ===============================
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=0
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1


# ===============================
# PATHS
# ===============================
#put our own base_dir and cache_dir here
BASE_DIR=
CODE_DIR=$BASE_DIR/completion_supervised_steering/src
DATA_DIR=$BASE_DIR/completion_supervised_steering/data_400x2
OUT_DIR=$BASE_DIR/completion_supervised_steering/outputs_400x2
CACHE_DIR=

MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_TAG=fears_llama_8b_alllayers
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL_TAG=fears_qwen_7b_alllayers 

# ===============================
# SHARED CLAS HYPERPARAMETERS (matches completion_rfm + controller's setup)
# ===============================
VAL_FRAC=0.2
LR_WEIGHT=3e-3
LR_BIAS=1e-1
EPOCHS=8
ACCUM_STEPS=10
PATIENCE=2


# ===============================
# STRATEGY: long_subsampled_anchor
# ===============================
echo ""
echo "############################################################"
echo "# CLAS 2/2 (SELF): long_subsampled_anchor"
echo "############################################################"
python $CODE_DIR/train_and_evaluate_clas.py \
    --concept_type            fear \
    --manifest_path            $DATA_DIR/fears/manifest30.json \
    --model                    $MODEL \
    --output_dir                $OUT_DIR/$MODEL_TAG \
    --source_strategy_dir       long_subsampled_anchor \
    --cache_dir                 $CACHE_DIR \
    --val_frac                  $VAL_FRAC \
    --lr_weight                 $LR_WEIGHT \
    --lr_bias                   $LR_BIAS \
    --epochs                    $EPOCHS \
    --accum_steps                $ACCUM_STEPS \
    --patience                   $PATIENCE \
    --eval_gen_tokens           100 \
    --judge_model                gpt-4o-mini \
    --seed                       42

echo ""
echo "Done. Per concept:"
echo "  outputs/\$MODEL_TAG/fear_<name>/clas_short_instructed_cut/summary.csv"
echo "  outputs/\$MODEL_TAG/fear_<name>/clas_long_subsampled_anchor/summary.csv"
echo "Combined:"
echo "  outputs/\$MODEL_TAG/combined_summary_clas_short_instructed_cut.csv"
echo "  outputs/\$MODEL_TAG/combined_summary_clas_long_subsampled_anchor.csv"
echo "Existing teacher results (run_clas_fear_teacher.sh) are untouched -- different output folder."