#!/usr/bin/env bash
# Original-code LOOK-M / PrefixKV on LLaVA-1.5-7B, keep 0.5 / 0.2 (total).
# Same protocol as Ours: MC tasks with delayed replay, free-form with full prefill.
# Usage: GPU=0 [METHODS="lookm_llava15_orig prefixkv_llava15_orig"] [KEEPS="0.5 0.2"] bash run_lookm_prefixkv_gpu.sh
set -uo pipefail

PY=/opt/conda/envs/prefixkv/bin/python
QVIK_ROOT=/workspace/nips/Q-ViK
MODEL=/workspace/nips/models/llava-v1.5-7b
OUT="${OUT:-${QVIK_ROOT}/results/baselines_orig_llava15}"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets TMPDIR=/workspace/nips/.tmp
# No GPT key here: MMBench's GPT fallback (used when the local option regex
# fails) can only fail. Point it at a closed local port so it fails instantly
# instead of waiting on api.openai.com 401s; the item is scored the same way.
export OPENAI_API_URL="http://127.0.0.1:9/v1/chat/completions"

MC_TASKS="${MC_TASKS:-mme,mmbench_en_dev,mmstar,pope,vizwiz_vqa,scienceqa_img}"
FREE_TASKS="${FREE_TASKS:-textvqa,gqa,docvqa,chartqa,coco_cap,nocaps,textcaps}"

cd "${QVIK_ROOT}"
for method in ${METHODS:-lookm_llava15_orig prefixkv_llava15_orig}; do
  for keep in ${KEEPS:-0.5 0.2}; do
    for spec in "${MC_TASKS}:true" "${FREE_TASKS}:false"; do
      tasks="${spec%%:*}"; delayed="${spec##*:}"
      echo "[baseline] START ${method} keep=${keep} delayed=${delayed} tasks=${tasks} $(date -Is)"
      "${PY}" -u scripts_local/baselines/run_baselines_orig.py \
        --model "${method}" \
        --model_args "pretrained=${MODEL},keep_ratio=${keep},delayed_replay=${delayed},device=cuda:0" \
        --tasks "${tasks}" \
        --batch_size 1 \
        --log_samples \
        --output_path "${OUT}"
      echo "[baseline] DONE ${method} keep=${keep} delayed=${delayed} rc=$? $(date -Is)"
    done
  done
done
echo "BASELINES_DONE $(date -Is)"
