#!/bin/bash
# Llama-3.3-70B-Instruct (unsloth bnb-4bit): Seq-RFM (sequence_rfm) + BoW-RFM (bag_of_words_rfm)
# for one concept type, on top of the finished 70B v2 runs in outputs_400x2_llama70b_v2.
# Same settings as the 8B runs (run_seqrfm_{personas,fears,experts}.sh: truncate_at_end, fast_bow,
# sigma=1, lam=1e-3, k=1, ell_tau=0.15, uniform temporal weight, 100 anchors, RFM_ITERS=3)
# and the same 70B v2 settings as the other methods (bf16, alpha x 32/80 uniform over layers).
#
# Reuses each concept's dataset_with_completions.json; re-extracts trajectories.pt (~100 GB,
# deleted after training). Rows are appended to the concept's existing summary.csv /
# per_example_results.csv; concepts whose summary.csv already has sequence_rfm rows are skipped.
#
#   TYPE=persona GPU=3 bash run_llama70b_seqrfm.sh
#   TYPE=fear    GPU=4 bash run_llama70b_seqrfm.sh
#   TYPE=expert  GPU=5 bash run_llama70b_seqrfm.sh

# source <your-venv>/bin/activate
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<index>}
TYPE=${TYPE:?set TYPE=persona|fear|expert}
export LLAMA70B_BF16=1 STEER_ALPHA_SCALE=0.4 PYTHONUNBUFFERED=1

BASE_DIR=   # put your own base_dir here

PROJECT_ROOT=$BASE_DIR/completion_supervised_steering
CODE_DIR=$PROJECT_ROOT/src
CACHE_DIR=   # put your own HF cache dir here
OUT_DIR=$PROJECT_ROOT/outputs_400x2_llama70b_v2
MODEL=llama_3.3_70b_4bit_it
MANIFEST=${MANIFEST_OVERRIDE:-$PROJECT_ROOT/data_400x2/${TYPE}s/manifest30.json}
MODEL_TAG=${TYPE}s_llama_70b_4bit_alllayers

RFM_ITERS=3
SIGMA=1.0
LAM=1e-3
K=1
ELL_TAU=0.15
TEMPORAL_WEIGHT=uniform
N_ANCHORS=100
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
FAST_BOW_CHUNK_SIZE=32

echo "TYPE=$TYPE GPU=$CUDA_VISIBLE_DEVICES MANIFEST=$MANIFEST ($(date))"
python $CODE_DIR/${TYPE}_experiment_seqrfm.py \
    --manifest_path            $MANIFEST \
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
    --seed                     42 || exit 1

echo ""; echo "Done TYPE=$TYPE ($(date))"
