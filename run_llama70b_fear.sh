#!/bin/bash
# Llama-3.3-70B-Instruct (unsloth bnb-4bit), FEARS, data_400x2: the 7 main-figure methods (same settings as run_llama70b_persona.sh).
#   P-DM (mean_difference), P-RFM (rfm), C-RFM (completion_rfm),
#   C-DM-all (bag_of_words_diff_means) -> fear_experiment.py (alpha sweep)
#   Ctx-P-RFM (clas)                    -> train_and_evaluate_clas.py
#   Ctx-C-RFM (completion_rfm_controller) -> run_completion_rfm_controller_pipeline.py
#   Joint-CS (joint_training, random init) -> train_joint_new.py + evaluate.py
# Last-token C-DM (the paper's C-DM): run_dm_pooling_ablation.sh.
# Hyperparameters match the Qwen-7B 400x2 runs (run_expert_contextual.sh + run_expert_joint.sh with MODEL_FAMILY=qwen),
# RFM_ITERS=3.
#
# The 30 fears are split into NCHUNKS chunks, one GPU each:
#   CHUNK=1 NCHUNKS=5 GPU=1 bash run_llama70b_fear.sh
#   ...
#   CHUNK=5 NCHUNKS=5 GPU=5 bash run_llama70b_fear.sh
# Every step skips work whose outputs already exist, so a crashed chunk can be rerun.

# source <your-venv>/bin/activate
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<index>}
CHUNK=${CHUNK:?set CHUNK=<1..NCHUNKS>}
NCHUNKS=${NCHUNKS:-5}

# ===============================
# PATHS
# ===============================
BASE_DIR=   # put your own base_dir here
PROJECT_ROOT=$BASE_DIR/completion_supervised_steering
CODE_DIR=$PROJECT_ROOT/src
CACHE_DIR=   # put your own HF cache dir here
FULL_MANIFEST=$PROJECT_ROOT/data_400x2/fears/manifest30.json
OUT_DIR=$PROJECT_ROOT/outputs_400x2_llama70b_v2
MODEL=llama_3.3_70b_4bit_it
MODEL_TAG=fears_llama_70b_4bit_alllayers
BASE_OUT=$OUT_DIR/$MODEL_TAG
CTRL_OUT=$OUT_DIR/${MODEL_TAG}_fear_ctx_ctrl
JOINT_MODELS_DIR=$OUT_DIR/outputs_fears_joint/models

# 70B settings: bf16 activations, alpha x 32/80 uniform over layers
# (fixed-alpha methods only; reported alpha labels unchanged); all lrs = 8B/Qwen defaults (Ctx-C-RFM lr 1e-3 tried in persona pilot, no gain).
export LLAMA70B_BF16=1 STEER_ALPHA_SCALE=0.4
mkdir -p $OUT_DIR/manifests
MANIFEST=$OUT_DIR/manifests/fears_chunk${CHUNK}of${NCHUNKS}.json
python3 -c "
import json; m = json.load(open('$FULL_MANIFEST'))
json.dump(m[$CHUNK - 1::$NCHUNKS], open('$MANIFEST', 'w'), indent=2)"

# ===============================
# HYPERPARAMETERS
# ===============================
RFM_ITERS=3
N_ANCHORS=100
COMPLETION_STRATEGY=long_subsampled_anchor
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
TRAIN_COMPLETION_TOKENS=100
EVAL_GEN_TOKENS=100
JUDGE_MODEL=gpt-4o-mini
SEED=42

CLAS_VAL_FRAC=0.2
CLAS_LR_WEIGHT=3e-3
CLAS_LR_BIAS=1e-1
CLAS_EPOCHS=8
CLAS_ACCUM_STEPS=10
CLAS_PATIENCE=2

CTRL_VAL_FRAC=0.2
CTRL_LR=3e-3
CTRL_LR_BIAS=1e-1
CTRL_EPOCHS=8
CTRL_ACCUM_STEPS=10
CTRL_PATIENCE=2
CTRL_WEIGHT_DECAY=1e-4
CTRL_GRAD_CLIP=1.0
CTRL_LOG_EVERY=20

JOINT_LR=3e-3
JOINT_DIR_LR=1e-3
JOINT_EPOCHS=8
JOINT_ACCUM_STEPS=10
JOINT_VAL_FRAC=0.15
JOINT_PATIENCE=2
JOINT_LOG_EVERY=10

echo "CHUNK=$CHUNK/$NCHUNKS GPU=$CUDA_VISIBLE_DEVICES ($(date))"
python3 -c "import json; print([e['fear'] for e in json.load(open('$MANIFEST'))])"


echo ""; echo "########## [1/4] P-DM, P-RFM, C-RFM, C-DM (alpha sweep) ($(date)) ##########"
python $CODE_DIR/fear_experiment.py \
    --manifest_path           $MANIFEST \
    --model                   $MODEL \
    --output_dir              $BASE_OUT \
    --cache_dir               $CACHE_DIR \
    --rfm_iters               $RFM_ITERS \
    --n_anchors               $N_ANCHORS \
    --alpha_grid              $ALPHA_GRID \
    --completion_strategy     $COMPLETION_STRATEGY \
    --train_completion_tokens $TRAIN_COMPLETION_TOKENS \
    --eval_gen_tokens         $EVAL_GEN_TOKENS \
    --judge_model             $JUDGE_MODEL \
    --seed                    $SEED || exit 1

echo ""; echo "########## [2/4] CLAS (Ctx-P-RFM) ($(date)) ##########"
python $CODE_DIR/train_and_evaluate_clas.py \
    --concept_type          fear \
    --manifest_path         $MANIFEST \
    --model                 $MODEL \
    --output_dir            $BASE_OUT \
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

echo ""; echo "########## [3/4] contextual CLAS (Ctx-C-RFM) ($(date)) ##########"
python $CODE_DIR/run_completion_rfm_controller_pipeline.py \
    --concept_type          fear \
    --manifest_path         $MANIFEST \
    --model                 $MODEL \
    --old_output_dir        $BASE_OUT \
    --new_output_dir        $CTRL_OUT \
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

echo ""; echo "########## [4/4] joint training, random init (Joint-CS) ($(date)) ##########"
mkdir -p $JOINT_MODELS_DIR
while IFS=$'\t' read -r FEAR DATASET_PATH; do
    SAFE=$(echo "$FEAR" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    COMPLETIONS_PATH=$BASE_OUT/fear_$SAFE/$COMPLETION_STRATEGY/dataset_with_completions.json
    echo ""; echo "=== joint: $FEAR ==="
    if [ -f "$JOINT_MODELS_DIR/$SAFE/eval_summary.json" ]; then
        echo "  already evaluated, skipping."
        continue
    fi
    if [ ! -f "$JOINT_MODELS_DIR/$SAFE/directions.pt" ]; then
        python $CODE_DIR/train_joint_new.py \
            --dataset     $COMPLETIONS_PATH \
            --model       $MODEL \
            --out_dir     $JOINT_MODELS_DIR \
            --epochs      $JOINT_EPOCHS \
            --lr          $JOINT_LR \
            --dir_lr      $JOINT_DIR_LR \
            --accum_steps $JOINT_ACCUM_STEPS \
            --log_every   $JOINT_LOG_EVERY \
            --val_frac    $JOINT_VAL_FRAC \
            --patience    $JOINT_PATIENCE \
            --cache_dir   $CACHE_DIR \
            --seed        $SEED
    fi
    python $CODE_DIR/evaluate.py \
        --dataset        $DATASET_PATH \
        --out_dir        $JOINT_MODELS_DIR \
        --model          $MODEL \
        --judge_model    $JUDGE_MODEL \
        --max_new_tokens $EVAL_GEN_TOKENS \
        --cache_dir      $CACHE_DIR
done < <(python3 -c "
import json
for e in json.load(open('$MANIFEST')):
    print(e['fear'] + '\t' + e['path'])
")

echo ""; echo "Done CHUNK=$CHUNK/$NCHUNKS ($(date))"
