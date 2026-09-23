#!/bin/bash
# 4 basic methods: mean_difference, rfm (baselines --
# computed ONCE, shared across both strategies, since they never touch
# completions) + completion_rfm, bag_of_words_diff_means

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

#switch models
# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL_TAG=fears_qwen_7b_alllayers
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_TAG=fears_llama_8b_alllayers 


# ===============================
# SHARED HYPERPARAMETERS (same for both strategies)
# ===============================
RFM_ITERS=8
SIGMA=1.0
LAM=1e-3
K=1
ELL_TAU=0.15
TEMPORAL_WEIGHT=uniform
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
N_ANCHORS=100

# ===============================
# STRATEGY: long_subsampled_anchor
# ===============================
echo ""
echo "############################################################"
echo "# STRATEGY 2/2: long_subsampled_anchor"
echo "############################################################"
python $CODE_DIR/fear_experiment.py \
    --manifest_path            $DATA_DIR/fears/manifest30.json \
    --model                    $MODEL \
    --output_dir                $OUT_DIR/$MODEL_TAG \
    --cache_dir                 $CACHE_DIR \
    --rfm_iters                 $RFM_ITERS \
    --sigma                     $SIGMA \
    --lam                       $LAM \
    --k                         $K \
    --n_anchors                 $N_ANCHORS \
    --ell_tau                   $ELL_TAU \
    --temporal_weight           $TEMPORAL_WEIGHT \
    --alpha_grid                $ALPHA_GRID \
    --completion_strategy       long_subsampled_anchor \
    --train_completion_tokens   100 \
    --eval_gen_tokens           100 \
    --judge_model                gpt-4o-mini \
    --seed                       42
    # --train_prompt_suffix intentionally omitted here.

echo ""
echo "Done. Per concept: outputs/\$MODEL_TAG/fear_<name>/{short_instructed_cut,long_subsampled_anchor}/summary.csv"
echo "Combined: outputs/\$MODEL_TAG/combined_summary_long_subsampled_anchor.csv"
echo "baseline_directions.pkl is shared -- computed once, same file used by both strategies."
