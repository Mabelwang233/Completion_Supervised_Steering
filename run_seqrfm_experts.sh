#!/bin/bash
# Expert analog of run_seqrfm_personas.sh: sequence-RFM + bag-of-words RFM
# (truncate_at_end, fast_bow), appended into the summary.csv files that
# run_experts_basic.sh wrote. Run after run_experts_basic.sh.
#
# Usage: MODEL_FAMILY=llama GPU=1 bash run_seqrfm_experts.sh

source "$(dirname "$0")/run_experts_env.sh"
MANIFEST=${MANIFEST_OVERRIDE:-$MANIFEST}   # optional: subset manifest to resume a partial run

RFM_ITERS=3
SIGMA=1.0
LAM=1e-3
K=1
ELL_TAU=0.15
TEMPORAL_WEIGHT=uniform
N_ANCHORS=100

# fast_bow chunk size -- raise if GPU memory allows (e.g. 64 or 128)
FAST_BOW_CHUNK_SIZE=32

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST -- run run_experts_basic.sh first" >&2
    exit 1
fi

echo ""
echo "############################################################"
echo "# sequence-RFM + bag-of-words RFM  (truncate_at_end, fast_bow)"
echo "# 30 expert concepts, $COMPLETION_STRATEGY"
echo "############################################################"

python $CODE_DIR/expert_experiment_seqrfm.py \
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
    --completion_strategy      $COMPLETION_STRATEGY \
    --train_completion_tokens  $TRAIN_COMPLETION_TOKENS \
    --eval_gen_tokens          $EVAL_GEN_TOKENS \
    --judge_model              $JUDGE_MODEL \
    --seed                     $SEED

echo ""
echo "Done."
echo "Per-concept results: $OUT_DIR/$MODEL_TAG/expert_<name>/$COMPLETION_STRATEGY/summary.csv"
echo "Combined:            $OUT_DIR/$MODEL_TAG/combined_summary_seqrfm_$COMPLETION_STRATEGY.csv"
