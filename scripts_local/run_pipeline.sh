#!/usr/bin/env bash
# Q-ViK end-to-end for one backbone: teacher extraction -> student training -> lmms-eval.
# Usage: BACKBONE=onevision|llava15 GPU=N bash run_pipeline.sh
#   onevision: keep 0.1, keep_ratio_basis=image
#   llava15:   keep 0.2, keep_ratio_basis=total
# STAGES="teacher train eval" (default all) to resume from a later stage.
set -euo pipefail

QVIK_ROOT=/workspace/nips/Q-ViK
PY=/opt/conda/envs/VFlowOpt/bin/python
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets
export TMPDIR=/workspace/nips/.tmp
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

case "${BACKBONE:?set BACKBONE}" in
  onevision)
    MODEL=/workspace/nips/qvik_experiment/model/llava-onevision-qwen2-7b-ov
    EXTRACT=qvik/teacher/extract_llava_onevision.py
    TRAIN=qvik/train/llava_onevision.py
    EPOCHS=20
    LMMS_MODEL=lmms_onevision_student
    KEEP=0.1; BASIS=image
    ;;
  llava15)
    MODEL=/workspace/nips/models/llava-v1.5-7b
    EXTRACT=qvik/teacher/extract_llava15.py
    TRAIN=qvik/train/llava15.py
    EPOCHS=15
    LMMS_MODEL=lmms_llava15_student
    KEEP=0.2; BASIS=total
    ;;
  *) echo "unknown BACKBONE" >&2; exit 1 ;;
esac

TAG="${RUN_TAG:-$(date +%Y%m%d)}"
TEACHER_ROOT="${QVIK_ROOT}/data/train/teacher/${BACKBONE}_${TAG}"
CKPT="${QVIK_ROOT}/ckpts/qvik_student_${BACKBONE}_${TAG}"
EVAL_OUT="${QVIK_ROOT}/results/pipeline_${BACKBONE}_${TAG}"
TASKS="${TASKS:-textvqa,chartqa,docvqa,gqa,coco_cap,nocaps,textcaps,mme,mmbench_en_dev,mmstar,pope,vizwiz_vqa,scienceqa_img}"
STAGES="${STAGES:-teacher train eval}"

cd "${QVIK_ROOT}"
echo "[pipeline] backbone=${BACKBONE} gpu=${GPU} teacher=${TEACHER_ROOT} ckpt=${CKPT} keep=${KEEP} basis=${BASIS}"

if [[ " ${STAGES} " == *" teacher "* ]]; then
  for ds in textvqa gqa scienceqa; do
    echo "[pipeline] TEACHER ${ds} $(date -Is)"
    "${PY}" -u "${EXTRACT}" --model "${MODEL}" --dataset "${ds}" --n-samples 600 --output-root "${TEACHER_ROOT}"
  done
fi

if [[ " ${STAGES} " == *" train "* ]]; then
  echo "[pipeline] TRAIN $(date -Is)"
  "${PY}" -u "${TRAIN}" --teacher-root "${TEACHER_ROOT}" --llava-path "${MODEL}" \
    --epochs "${EPOCHS}" --output-dir "${CKPT}" --overwrite
fi

if [[ " ${STAGES} " == *" eval "* ]]; then
  echo "[pipeline] EVAL $(date -Is)"
  "${PY}" -u qvik/eval/run_lmms_eval.py \
    --model "${LMMS_MODEL}" \
    --model_args "pretrained=${MODEL},student_path=${CKPT},keep_ratio=${KEEP},keep_ratio_basis=${BASIS},device=cuda:0" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${EVAL_OUT}"
fi
echo "PIPELINE_DONE ${BACKBONE} $(date -Is)"
