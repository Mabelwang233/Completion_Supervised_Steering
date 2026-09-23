#!/bin/bash
# run_teacher_ablation_llama.sh
# Teacher-completion ablation — meta-llama/Llama-3.1-8B-Instruct only.
# Runs Stage 0-4 for llama on both persona and fear categories.
# Stages 0-2 (concept selection, teacher generation, dataset formatting) are
# shared — the script skips them if outputs already exist.
# ===============================
# ENV + GPU
# ===============================
# replace with your own token and key
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===============================
# PARSE FLAGS
# ===============================
DRY_RUN=0
ONE_CONCEPT=0
for arg in "$@"; do
    case $arg in
        --dry_run)     DRY_RUN=1 ;;
        --one_concept) ONE_CONCEPT=1 ;;
    esac
done

# ===============================
# PATHS
# ===============================
#put our own base_dir and cache_dir here
BASE_DIR=
CODE_DIR=$BASE_DIR/completion_supervised_steering/src
DATA_DIR=$BASE_DIR/completion_supervised_steering/data_400x2
OUT_DIR=$BASE_DIR/completion_supervised_steering/outputs_400x2
CACHE_DIR=

ABLATION_DIR=$BASE_DIR/completion_supervised_steering/teacher_ablation

# Root of main experiment outputs — needed so CLAS can find baseline_directions.pkl
# (prompt-RFM direction computed during self-completion runs, shared with teacher runs)
MAIN_OUT=$BASE_DIR/completion_supervised_steering/outputs_400x2
ABLATION_MANIFEST=$ABLATION_DIR/ablation_manifest.json
CANONICAL_COMPLETIONS=$ABLATION_DIR/canonical_teacher_completions.json
TEACHER_CACHE=$ABLATION_DIR/teacher_cache.json
TEACHER_DATASETS=$ABLATION_DIR/datasets
ABLATION_OUT=$ABLATION_DIR/outputs

# Self-completion results (pre-computed best scores, one row per concept x method)
# Format: concept, method, score, alpha
ALL_COMBINED=$BASE_DIR/completion_supervised_steering/all_combined

# ===============================
# HYPERPARAMETERS
# ===============================
TEACHER_MODEL=o3-mini
TEACHER_MAX_TOKENS=1000
REASONING_EFFORT=low
RFM_ITERS=8
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
EVAL_GEN_TOKENS=100
JUDGE_MODEL=gpt-4o-mini
SEED=42
ABLATION_DIR=$BASE_DIR/completion_supervised_steering/teacher_ablation
# Joint training hyperparameters
JOINT_EPOCHS=8
JOINT_LR=3e-3
JOINT_DIR_LR=1e-4
JOINT_ACCUM=10
JOINT_VAL_FRAC=0.15
JOINT_PATIENCE=2

# CLAS (Llama settings; override for Qwen below)
CLAS_VAL_FRAC=0.2
CLAS_LR_WEIGHT=3e-3
CLAS_LR_BIAS=1e-1
CLAS_EPOCHS=8
CLAS_ACCUM=10
CLAS_PATIENCE=2

# Contextual completion RFM controller is now integrated directly —
# no external script path needed (uses run_completion_rfm_controller_pipeline.py
# + train_controller.py from this repo).

mkdir -p $ABLATION_DIR $ABLATION_OUT

cd $CODE_DIR

# ======================================================================
# STAGE 0: Select 10 persona + 10 fear concepts (seed=42)
# ======================================================================
echo ""
echo "############################################################"
echo "# STAGE 0: select ablation concepts"
echo "############################################################"
if [ ! -f $ABLATION_MANIFEST ]; then
    python select_ablation_concepts.py \
        --persona_manifest  $DATA_DIR/personas/manifest30.json \
        --fear_manifest     $DATA_DIR/fears/manifest30.json \
        --out_path          $ABLATION_MANIFEST \
        --n                 10 \
        --seed              $SEED
else
    echo "  $ABLATION_MANIFEST exists, skipping."
fi

# ======================================================================
# STAGE 1: Generate canonical teacher completions
# ======================================================================
echo ""
echo "############################################################"
echo "# STAGE 1: generate teacher completions (o3-mini)"
echo "############################################################"
DRY_FLAG=""
[ $DRY_RUN -eq 1 ] && DRY_FLAG="--dry_run"

python generate_teacher_completions.py \
    --ablation_manifest  $ABLATION_MANIFEST \
    --out_path           $CANONICAL_COMPLETIONS \
    --cache_path         $TEACHER_CACHE \
    --teacher_model      $TEACHER_MODEL \
    --api_max_tokens     $TEACHER_MAX_TOKENS \
    --reasoning_effort   $REASONING_EFFORT \
    --train_prompt_suffix  "$TEACHER_PROMPT_SUFFIX" \
    --max_retries        4 \
    --seed               $SEED \
    $DRY_FLAG

[ $DRY_RUN -eq 1 ] && echo "Dry run complete." && exit 0

# ======================================================================
# STAGE 2: Format target-model datasets (no API calls)
# ======================================================================
echo ""
echo "############################################################"
echo "# STAGE 2: format teacher datasets for Llama + Qwen"
echo "############################################################"
python format_teacher_datasets.py \
    --ablation_manifest      $ABLATION_MANIFEST \
    --canonical_completions  $CANONICAL_COMPLETIONS \
    --out_dir                $TEACHER_DATASETS \
    --models                 meta-llama/Llama-3.1-8B-Instruct \
    --model_tags             llama \
    --cache_dir              $CACHE_DIR

# ======================================================================
# STAGE 3: Run completion-based methods (meta-llama/Llama-3.1-8B-Instruct)
# ======================================================================
echo ""
echo "############################################################"
echo "# STAGE 3: run teacher-completion methods (llama)"
echo "############################################################"

for CATEGORY in persona fear; do
    echo ""
    echo "--- Category: $CATEGORY (llama) ---"

    python run_teacher_ablation.py \
        --ablation_manifest    $ABLATION_MANIFEST \
        --teacher_datasets_dir $TEACHER_DATASETS \
        --category             $CATEGORY \
        --model                meta-llama/Llama-3.1-8B-Instruct \
        --model_tag            llama \
        --out_dir              $ABLATION_OUT \
        --alpha_grid           $ALPHA_GRID \
        --rfm_iters            $RFM_ITERS \
        --eval_gen_tokens      $EVAL_GEN_TOKENS \
        --judge_model          $JUDGE_MODEL \
        --cache_dir            $CACHE_DIR \
        --seed                 $SEED \
        --run_joint \
        --joint_epochs         $JOINT_EPOCHS \
        --joint_lr             $JOINT_LR \
        --joint_dir_lr         $JOINT_DIR_LR \
        --joint_accum_steps    $JOINT_ACCUM \
        --joint_val_frac       $JOINT_VAL_FRAC \
        --joint_patience       $JOINT_PATIENCE \
        --run_clas \
        --baseline_dir         $MAIN_OUT/${CATEGORY}s_llama_8b_alllayers \
        --clas_val_frac        $CLAS_VAL_FRAC \
        --clas_lr_weight       $CLAS_LR_WEIGHT \
        --clas_lr_bias         $CLAS_LR_BIAS \
        --clas_epochs          $CLAS_EPOCHS \
        --clas_accum_steps     $CLAS_ACCUM \
        --clas_patience        $CLAS_PATIENCE
done

# ======================================================================
# STAGE 4: Paired comparison vs self-completion results
# ======================================================================
echo ""
echo "############################################################"
echo "# STAGE 4: paired comparison self vs teacher (llama)"
echo "############################################################"

for CATEGORY in persona fear; do
    TEACHER_CSV=$ABLATION_OUT/llama/combined_summary_teacher_${CATEGORY}.csv
    OUT_CSV=$ABLATION_DIR/comparison_${CATEGORY}_llama.csv

    if [ -f "$TEACHER_CSV" ]; then
        python compare_teacher_vs_self.py \
            --all_combined_dir $ALL_COMBINED \
            --teacher_summary  $TEACHER_CSV \
            --category         $CATEGORY \
            --model_tag        llama \
            --out_path         $OUT_CSV
    else
        echo "  SKIPPING $CATEGORY/llama: teacher summary not found at $TEACHER_CSV"
    fi
done

echo ""
echo "############################################################"
echo "# ALL STAGES DONE (llama)"
echo "############################################################"
echo "Ablation outputs: $ABLATION_DIR"
echo "Paired comparisons:"
ls $ABLATION_DIR/comparison_*_llama.csv 2>/dev/null