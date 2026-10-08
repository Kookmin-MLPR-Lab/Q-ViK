#!/usr/bin/env bash
# Q-ViK OneVision student (e20 checkpoint), prefill_mode=qvik (default),
# on MME / MMBench-EN / MMStar / POPE / VizWiz / ScienceQA-IMG at keep 0.5/0.25/0.1.
# Usage: GPU=1 [LIMIT=4] [KEEPS="0.1"] [TASKS=mme] [OUT=...] bash run_onevision_e20_mc.sh
set -uo pipefail

QVIK_ROOT=/workspace/nips/Q-ViK
PY=/opt/conda/envs/VFlowOpt/bin/python
PRETRAINED=/workspace/nips/qvik_experiment/model/llava-onevision-qwen2-7b-ov
STUDENT=/workspace/nips/zap/ckpts/student_onevision_e20
OUT="${OUT:-${QVIK_ROOT}/results/onevision_e20_mc}"

export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets
export TMPDIR=/workspace/nips/.tmp
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

TASKS="${TASKS:-mme,mmbench_en_dev,mmstar,pope,vizwiz_vqa,scienceqa_img}"
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "${LIMIT}")

cd "${QVIK_ROOT}"
for keep in ${KEEPS:-0.5 0.25 0.1}; do
    echo "[e20-mc] START keep=${keep} $(date -Is)"
    "${PY}" -u qvik/eval/run_lmms_eval.py \
        --model lmms_onevision_student \
        --model_args "pretrained=${PRETRAINED},student_path=${STUDENT},keep_ratio=${keep},device=cuda:0" \
        --tasks "${TASKS}" \
        --batch_size 1 \
        --log_samples \
        "${LIMIT_ARGS[@]}" \
        --output_path "${OUT}"
    echo "[e20-mc] DONE keep=${keep} rc=$? $(date -Is)"
done
echo "E20_MC_DONE $(date -Is)"
