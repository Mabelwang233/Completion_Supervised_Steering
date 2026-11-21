#!/bin/bash
# run_teacher_ablation_experts.sh
#
# Expert analog of run_teacher_ablation_llama.sh (same teacher, same hyperparameters):
#   Stage 0: select 10 of the 30 experts (seed=42)
#   Stage 1: o3-mini teacher completions (shares teacher_cache.json with the
#            persona/fear ablation, so already-generated prompts are reused)
#   Stage 2: format teacher datasets for Llama + Qwen
#   Stage 3: completion-based methods on teacher datasets
#            (completion_rfm, contextual completion-RFM controller, CLAS, joint)
#   Stage 4: per-method self-vs-teacher plot
#
# Llama and Qwen run one after the other (never both on the GPU at once).
#
# Usage:
#   bash run_teacher_ablation_experts.sh --dry_run     # count API calls only
#   bash run_teacher_ablation_experts.sh               # everything
#   MODELS="qwen" bash run_teacher_ablation_experts.sh # one model only

# ===============================
# ENV + GPU
# ===============================
# source <your-venv>/bin/activate
export HUGGINGFACE_HUB_TOKEN=""
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry_run) DRY_RUN=1 ;;
    esac
done
MODELS=${MODELS:-"llama qwen"}

# ===============================
# PATHS
# ===============================
BASE_DIR=   # put your own base_dir here
CODE_DIR=$BASE_DIR/completion_supervised_steering/src
DATA_DIR=$BASE_DIR/completion_supervised_steering/data_400x2
CACHE_DIR=   # put your own HF cache dir here
MAIN_OUT=$BASE_DIR/completion_supervised_steering/outputs_400x2

ABLATION_DIR=$BASE_DIR/completion_supervised_steering/teacher_ablation
ABLATION_MANIFEST=$ABLATION_DIR/ablation_manifest_experts.json
CANONICAL_COMPLETIONS=$ABLATION_DIR/canonical_teacher_completions_experts.json
TEACHER_CACHE=$ABLATION_DIR/teacher_cache.json   # shared with persona/fear
TEACHER_DATASETS=$ABLATION_DIR/datasets
ABLATION_OUT=$ABLATION_DIR/outputs

# ===============================
# MODELS
# ===============================
declare -A MODEL_ID=( [llama]=meta-llama/Llama-3.1-8B-Instruct [qwen]=Qwen/Qwen2.5-7B-Instruct )
# prompt-RFM baseline_directions.pkl for CLAS, from run_experts_basic.sh
declare -A BASELINE_DIR=( [llama]=$MAIN_OUT/experts_llama_8b_alllayers [qwen]=$MAIN_OUT/experts_qwen_7b_alllayers )
declare -A CLAS_LR_BIAS=( [llama]=1e-1 [qwen]=5e-1 )

# ===============================
# HYPERPARAMETERS (identical to run_teacher_ablation_llama.sh)
# ===============================
TEACHER_MODEL=o3-mini
TEACHER_MAX_TOKENS=1000
REASONING_EFFORT=low
TEACHER_PROMPT_SUFFIX="Answer in at most 100 tokens."
RFM_ITERS=8
ALPHA_GRID="0.3 0.4 0.5 0.6 0.7 0.8"
EVAL_GEN_TOKENS=100
JUDGE_MODEL=gpt-4o-mini
SEED=42

JOINT_EPOCHS=8
JOINT_LR=3e-3
JOINT_DIR_LR=1e-4
JOINT_ACCUM=10
JOINT_VAL_FRAC=0.15
JOINT_PATIENCE=2

CLAS_VAL_FRAC=0.2
CLAS_LR_WEIGHT=3e-3
CLAS_EPOCHS=8
CLAS_ACCUM=10
CLAS_PATIENCE=2

mkdir -p $ABLATION_DIR $ABLATION_OUT
cd $CODE_DIR

# ======================================================================
# STAGE 0: select 10 experts (seed=42)
# ======================================================================
echo ""
echo "# STAGE 0: select ablation experts"
if [ ! -f $ABLATION_MANIFEST ]; then
    python select_ablation_concepts.py \
        --expert_manifest  $DATA_DIR/experts/manifest30.json \
        --out_path         $ABLATION_MANIFEST \
        --n                10 \
        --seed             $SEED
else
    echo "  $ABLATION_MANIFEST exists, skipping."
fi

# ======================================================================
# STAGE 1: teacher completions (o3-mini, cached + resumable)
# ======================================================================
echo ""
echo "# STAGE 1: generate teacher completions ($TEACHER_MODEL)"
DRY_FLAG=""
[ $DRY_RUN -eq 1 ] && DRY_FLAG="--dry_run"
python generate_teacher_completions.py \
    --ablation_manifest    $ABLATION_MANIFEST \
    --out_path             $CANONICAL_COMPLETIONS \
    --cache_path           $TEACHER_CACHE \
    --teacher_model        $TEACHER_MODEL \
    --api_max_tokens       $TEACHER_MAX_TOKENS \
    --reasoning_effort     $REASONING_EFFORT \
    --train_prompt_suffix  "$TEACHER_PROMPT_SUFFIX" \
    --max_retries          4 \
    --seed                 $SEED \
    $DRY_FLAG
[ $DRY_RUN -eq 1 ] && echo "Dry run complete." && exit 0

# ======================================================================
# STAGE 2: format teacher datasets (tokenizers only, no GPU)
# ======================================================================
echo ""
echo "# STAGE 2: format teacher datasets"
python format_teacher_datasets.py \
    --ablation_manifest      $ABLATION_MANIFEST \
    --canonical_completions  $CANONICAL_COMPLETIONS \
    --out_dir                $TEACHER_DATASETS \
    --models                 ${MODEL_ID[llama]} ${MODEL_ID[qwen]} \
    --model_tags             llama qwen \
    --cache_dir              $CACHE_DIR

# ======================================================================
# STAGE 3: completion-based methods, one model at a time
# ======================================================================
for TAG in $MODELS; do
    echo ""
    echo "# STAGE 3: [$TAG] expert teacher-completion methods"
    python run_teacher_ablation.py \
        --ablation_manifest    $ABLATION_MANIFEST \
        --teacher_datasets_dir $TEACHER_DATASETS \
        --category             expert \
        --model                ${MODEL_ID[$TAG]} \
        --model_tag            $TAG \
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
        --baseline_dir         ${BASELINE_DIR[$TAG]} \
        --clas_val_frac        $CLAS_VAL_FRAC \
        --clas_lr_weight       $CLAS_LR_WEIGHT \
        --clas_lr_bias         ${CLAS_LR_BIAS[$TAG]} \
        --clas_epochs          $CLAS_EPOCHS \
        --clas_accum_steps     $CLAS_ACCUM \
        --clas_patience        $CLAS_PATIENCE
done

# ======================================================================
# STAGE 4: per-method self-vs-teacher plot (needs both models done)
# ======================================================================
echo ""
echo "# STAGE 4: plot"
if [ -f $ABLATION_OUT/llama/combined_summary_teacher_expert.csv ] && \
   [ -f $ABLATION_OUT/qwen/combined_summary_teacher_expert.csv ]; then
    python $ABLATION_DIR/plot_teacher_ablation_per_method.py \
        --categories expert --basename teacher_ablation_per_method_expert
else
    echo "  skipping: need combined_summary_teacher_expert.csv for both llama and qwen"
fi

echo ""
echo "Done. Outputs: $ABLATION_OUT/{llama,qwen}/expert_<slug>/"
