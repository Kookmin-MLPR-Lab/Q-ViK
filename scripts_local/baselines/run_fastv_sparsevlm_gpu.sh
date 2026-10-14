#!/usr/bin/env bash
# FastV / SparseVLM (released code) on LLaVA-1.5-7B, 32-layer-mean KV = keep * total.
# Pruning happens inside the prefill, so every task uses plain generate().
# Usage: GPU=2 [METHODS="fastv_llava15_orig sparsevlm_llava15_orig"] [KEEPS="0.5 0.2"] bash run_fastv_sparsevlm_gpu.sh
set -uo pipefail

QVIK_ROOT=/workspace/nips/Q-ViK
MODEL=/workspace/nips/models/llava-v1.5-7b
OUT="${OUT:-${QVIK_ROOT}/results/baselines_orig_llava15}"
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets TMPDIR=/workspace/nips/.tmp
# MMBench GPT fallback has no key; fail instantly (same score as a 401).
export OPENAI_API_URL="http://127.0.0.1:9/v1/chat/completions"
TASKS="${TASKS:-mme,mmbench_en_dev,mmstar,pope,vizwiz_vqa,scienceqa_img,textvqa,gqa,docvqa,chartqa,coco_cap,nocaps,textcaps}"

declare -A ENV=([fastv_llava15_orig]=/opt/conda/envs/fastv [sparsevlm_llava15_orig]=/opt/conda/envs/sparse-fastv)

cd "${QVIK_ROOT}"
for method in ${METHODS:-fastv_llava15_orig sparsevlm_llava15_orig}; do
  for keep in ${KEEPS:-0.5 0.2}; do
    echo "[baseline] START ${method} keep=${keep} $(date -Is)"
    "${ENV[$method]}/bin/python" -u scripts_local/baselines/run_baselines_orig.py \
      --model "${method}" \
      --model_args "pretrained=${MODEL},keep_ratio=${keep},device=cuda:0" \
      --tasks "${TASKS}" --batch_size 1 --log_samples --output_path "${OUT}"
    echo "[baseline] DONE ${method} keep=${keep} rc=$? $(date -Is)"
  done
done
echo "BASELINES_DONE $(date -Is)"
