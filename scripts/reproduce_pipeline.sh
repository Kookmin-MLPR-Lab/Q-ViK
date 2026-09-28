#!/usr/bin/env bash
# Fresh teacher extraction -> student training -> eval for both models on one GPU,
# then compare the new students against the reference zap checkpoints.
#
#   GPU=2 N_SAMPLES=600 bash scripts/reproduce_pipeline.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

GPU="${GPU:-2}"
N_SAMPLES="${N_SAMPLES:-600}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
EVAL_TASKS="${EVAL_TASKS:-pope,scienceqa_img,gqa}"
KEEP="${KEEP:-0.2}"
PY="${PY:-/opt/conda/envs/qvik/bin/python}"
REF_LLAVA15="${REF_LLAVA15:-../zap/ckpts/v1/student_llava15_orig_vflow_1800_lr1e4_e15}"
REF_ONEVISION="${REF_ONEVISION:-../zap/ckpts/student_onevision}"
LOG_DIR="logs/reproduce"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() { echo "===== [$(date -Is)] $*" | tee -a "$LOG_DIR/stages.log"; }
done_marker() { [ -f "$LOG_DIR/$1.done" ]; }
mark() { touch "$LOG_DIR/$1.done"; }
# Teacher extractors exit 2 when the candidate pool runs out before --n-samples
# records pass the correctness filter; the saved records are still valid.
teacher_ok() {
  local rc=$1 name=$2
  if [ "$rc" -eq 2 ]; then
    echo "  [warn] $name: fewer than $N_SAMPLES records (candidates exhausted)" | tee -a "$LOG_DIR/stages.log"
  elif [ "$rc" -ne 0 ]; then
    echo "  [fail] $name exit=$rc" | tee -a "$LOG_DIR/stages.log"; exit "$rc"
  fi
}

# ---- LLaVA-1.5 -------------------------------------------------------------
for ds in textvqa gqa scienceqa; do
  if ! done_marker "teacher_llava15_$ds"; then
    stage "teacher llava15 $ds"
    "$PY" qvik/teacher/extract_llava15.py --dataset "$ds" --n-samples "$N_SAMPLES" \
      --device cuda:0 --output-root data/train/teacher/llava15 \
      > "$LOG_DIR/teacher_llava15_$ds.log" 2>&1 && rc=0 || rc=$?
    teacher_ok "$rc" "teacher_llava15_$ds"
    mark "teacher_llava15_$ds"
  fi
done

if ! done_marker train_llava15; then
  stage "train llava15"
  "$PY" qvik/train/llava15.py --teacher-root data/train/teacher/llava15 \
    --datasets textvqa gqa scienceqa --n-per-dataset 600 --epochs 15 \
    --output-dir ckpts/qvik_student_llava15 --overwrite \
    > "$LOG_DIR/train_llava15.log" 2>&1
  mark train_llava15
fi

# ---- LLaVA-OneVision -------------------------------------------------------
for ds in textvqa gqa scienceqa; do
  if ! done_marker "teacher_onevision_$ds"; then
    stage "teacher onevision $ds"
    "$PY" qvik/teacher/extract_llava_onevision.py --dataset "$ds" --n-samples "$N_SAMPLES" \
      --device cuda:0 --output-root data/train/teacher/llava_onevision \
      > "$LOG_DIR/teacher_onevision_$ds.log" 2>&1 && rc=0 || rc=$?
    teacher_ok "$rc" "teacher_onevision_$ds"
    mark "teacher_onevision_$ds"
  fi
done

if ! done_marker train_onevision; then
  stage "train onevision"
  "$PY" qvik/train/llava_onevision.py --teacher-root data/train/teacher/llava_onevision \
    --datasets textvqa gqa scienceqa --per-ds-limit 600 --epochs 20 \
    --output-dir ckpts/qvik_student_onevision --overwrite \
    > "$LOG_DIR/train_onevision.log" 2>&1
  mark train_onevision
fi

# ---- Eval: new vs reference students ---------------------------------------
run_eval() {  # name model model_args
  if done_marker "eval_$1"; then return; fi
  stage "eval $1"
  "$PY" qvik/eval/run_lmms_eval.py --model "$2" \
    --model_args "$3,keep_ratio=$KEEP,device=cuda:0,stats_output_dir=results/reproduce/$1" \
    --tasks "$EVAL_TASKS" --batch_size 1 --limit "$EVAL_LIMIT" \
    --output_path "results/reproduce/$1" > "$LOG_DIR/eval_$1.log" 2>&1
  mark "eval_$1"
}
L15="pretrained=model/llava-v1.5-7b,keep_ratio_basis=image"
OV="pretrained=model/llava-onevision-qwen2-7b-ov"
run_eval llava15_new lmms_llava15_student "$L15,student_path=ckpts/qvik_student_llava15"
run_eval llava15_ref lmms_llava15_student "$L15,student_path=$REF_LLAVA15"
run_eval onevision_new lmms_onevision_student "$OV,student_path=ckpts/qvik_student_onevision"
run_eval onevision_ref lmms_onevision_student "$OV,student_path=$REF_ONEVISION"

stage "all done"
