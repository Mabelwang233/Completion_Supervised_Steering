#!/bin/bash
# run_seqrfm_fears.sh
#
# Runs ONLY sequence-RFM and bag-of-words RFM on the 30 fear concepts,
# both with truncate_at_end=True and fast_bow=True.
#
# Mirrors run_seqrfm_personas.sh exactly -- only the python script,
# manifest, and MODEL_TAG differ.
#
# Reuses existing artifacts (dataset_with_completions.json, trajectories.pt)
# from the long_subsampled_anchor run if already present -- steps 1-2
# are skipped for any concept that is already done.
#
# New outputs written into the SAME strategy subfolder as the main run:
#   <OUT_DIR>/<MODEL_TAG>/fear_<name>/long_subsampled_anchor/
#       sequence_rfm_directions.pt
#       bag_of_words_rfm_directions.pt
#       per_example_results.csv   (sequence_rfm + bag_of_words_rfm rows appended)
#       summary.csv               (sequence_rfm + bag_of_words_rfm rows appended)
#   <OUT_DIR>/<MODEL_TAG>/combined_summary_seqrfm_long_subsampled_anchor.csv

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

MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_TAG=fears_llama_8b_alllayers   # matches run_fear_completion_strategies_long.sh

# ===============================
# HYPERPARAMETERS  (match run_fear_completion_strategies_long.sh)
# ===============================
RFM_ITERS=3
SIGMA=1.0
LAM=1e-3
K=1
ELL_TAU=0.15
TEMPORAL_WEIGHT=uniform
N_ANCHORS=100
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"

# fast_bow chunk size -- raise if GPU memory allows (e.g. 64 or 128)
FAST_BOW_CHUNK_SIZE=32

echo ""
echo "############################################################"
echo "# sequence-RFM + bag-of-words RFM  (truncate_at_end, fast_bow)"
echo "# 30 fear concepts, long_subsampled_anchor"
echo "############################################################"

python $CODE_DIR/fear_experiment_seqrfm.py \
    --manifest_path            $DATA_DIR/fears/manifest30.json \
    --model                    $MODEL \
    --output_dir               $OUT_DIR/$MODEL_TAG \
    --cache_dir                $CACHE_DIR \
    --rfm_iters                $RFM_ITERS \
    --sigma                    $SIGMA \
    --lam                      $LAM \
    --k                        $K \
    --n_anchors                $N_ANCHORS \
    --ell_tau                  $ELL_TAU \
    --temporal_weight          $TEMPORAL_WEIGHT \
    --fast_bow_chunk_size      $FAST_BOW_CHUNK_SIZE \
    --alpha_grid               $ALPHA_GRID \
    --completion_strategy      long_subsampled_anchor \
    --train_completion_tokens  100 \
    --eval_gen_tokens          100 \
    --judge_model              gpt-4o-mini \
    --seed                     42

echo ""
echo "Done."
echo "Per-concept results: $OUT_DIR/$MODEL_TAG/fear_<name>/long_subsampled_anchor/summary.csv"
echo "Combined:            $OUT_DIR/$MODEL_TAG/combined_summary_seqrfm_long_subsampled_anchor.csv"
