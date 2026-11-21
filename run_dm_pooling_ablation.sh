#!/bin/bash
# Difference-in-means token pooling: prompt vs. completion x last token vs. mean over all tokens.
#
#                      last token                          mean over all tokens
#   prompt x           P-DM   (mean_difference,            P-DM-all (prompt_dm_all)      <- this script
#                              run_*_basic.sh)
#   prompt+completion  C-DM   (completion_dm_lastonly)     C-DM-all (bag_of_words_diff_means,
#                             <- this script                          run_*_basic.sh)
#
# C-DM in the main results is completion_dm_lastonly: diff-means of the LAST token of
# (formatted prompt + completion), i.e. the same token position as P-DM, so the two differ
# only in whether the completion is included. C-DM-all (mean over all completion tokens,
# as in Persona Vectors) is reported in the pooling ablation.
#
# Reuses each concept's dataset_with_completions.json from run_{personas,fears,experts}_basic.sh
# (or run_llama70b_*.sh), so run those first. Two stages per method:
#   directions : one model load, directions for every concept
#   eval       : alpha sweep + judge per concept
#
# Usage:
#   MODEL_FAMILY=llama8b  GPU=0 bash run_dm_pooling_ablation.sh                         # both methods
#   MODEL_FAMILY=qwen     GPU=0 METHODS=completion_dm_lastonly bash run_dm_pooling_ablation.sh
#   MODEL_FAMILY=llama70b GPU=0 CATEGORIES="expert" bash run_dm_pooling_ablation.sh

# source <your-venv>/bin/activate
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# ===============================
# PATHS
# ===============================
BASE_DIR=   # put your own base_dir here
PROJECT_ROOT=$BASE_DIR/completion_supervised_steering
CODE_DIR=$PROJECT_ROOT/src
DATA_DIR=$PROJECT_ROOT/data_400x2
CACHE_DIR=   # put your own HF cache dir here

METHODS=${METHODS:-"completion_dm_lastonly prompt_dm_all"}
CATEGORIES=${CATEGORIES:-"persona fear expert"}
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"

case "${MODEL_FAMILY:-llama8b}" in
    llama8b)
        MODEL=meta-llama/Llama-3.1-8B-Instruct; TAG=llama_8b
        SELF_OUT=$PROJECT_ROOT/outputs_400x2 ;;
    qwen)
        MODEL=Qwen/Qwen2.5-7B-Instruct; TAG=qwen_7b
        SELF_OUT=$PROJECT_ROOT/outputs_400x2 ;;
    llama70b)
        # same 70B settings as run_llama70b_*.sh: bf16 activations, alpha x 0.4
        export LLAMA70B_BF16=1 STEER_ALPHA_SCALE=0.4
        MODEL=llama_3.3_70b_4bit_it; TAG=llama_70b_4bit
        SELF_OUT=$PROJECT_ROOT/outputs_400x2_llama70b_v2 ;;
    *)
        echo "ERROR: MODEL_FAMILY must be llama8b, qwen or llama70b" >&2; exit 1 ;;
esac
OUT=$SELF_OUT/dm_variants_$TAG

for M in $METHODS; do
    for STAGE in directions eval; do
        echo "########## $MODEL_FAMILY / $M / $STAGE ($(date)) ##########"
        python $CODE_DIR/dm_variants_experiment.py \
            --method          $M \
            --stage           $STAGE \
            --categories      $CATEGORIES \
            --model           $MODEL \
            --model_tag       $TAG \
            --data_dir        $DATA_DIR \
            --self_out_dir    $SELF_OUT \
            --out_dir         $OUT \
            --alpha_grid      $ALPHA_GRID \
            --eval_gen_tokens 100 \
            --judge_model     gpt-4o-mini \
            --cache_dir       $CACHE_DIR \
            --seed            42 || exit 1
    done
done
echo "Done ($(date)). Per concept: $OUT/<category>s/<category>_<name>/<method>/summary.csv"
