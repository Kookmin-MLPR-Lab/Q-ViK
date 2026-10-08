#!/usr/bin/env bash
# Free-form tasks with the 2026-10-03 Q-ViK checkpoints, prefill_mode=origin
# (full prefill; first answer token from the unpruned pass, evict afterwards).
# Usage: BACKBONE=onevision|llava15 GPU=N bash run_eval_origin.sh
set -uo pipefail

QVIK_ROOT=/workspace/nips/Q-ViK
PY=/opt/conda/envs/VFlowOpt/bin/python
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets
export TMPDIR=/workspace/nips/.tmp

case "${BACKBONE:?set BACKBONE}" in
  onevision)
    MODEL=/workspace/nips/qvik_experiment/model/llava-onevision-qwen2-7b-ov
    LMMS_MODEL=lmms_onevision_student
    KEEPS="${KEEPS:-0.5 0.25 0.1}"; BASIS=image ;;
  llava15)
    MODEL=/workspace/nips/models/llava-v1.5-7b
    LMMS_MODEL=lmms_llava15_student
    KEEPS="${KEEPS:-0.5 0.2}"; BASIS=total ;;
  *) echo "unknown BACKBONE" >&2; exit 1 ;;
esac
CKPT="${QVIK_ROOT}/ckpts/qvik_student_${BACKBONE}_20261003"
PREFILL_MODE="${PREFILL_MODE:-origin}"
OUT="${OUT:-${QVIK_ROOT}/results/origin_${BACKBONE}_20261003}"
TASKS="${TASKS:-textvqa,gqa,docvqa,chartqa,coco_cap,nocaps,textcaps}"

cd "${QVIK_ROOT}"
for keep in ${KEEPS}; do
  echo "[origin] START ${BACKBONE} keep=${keep} basis=${BASIS} prefill=${PREFILL_MODE} $(date -Is)"
  "${PY}" -u qvik/eval/run_lmms_eval.py \
    --model "${LMMS_MODEL}" \
    --model_args "pretrained=${MODEL},student_path=${CKPT},keep_ratio=${keep},keep_ratio_basis=${BASIS},prefill_mode=${PREFILL_MODE},device=cuda:0" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUT}"
  echo "[origin] DONE ${BACKBONE} keep=${keep} rc=$? $(date -Is)"
done
echo "ORIGIN_DONE ${BACKBONE} $(date -Is)"
