#!/usr/bin/env bash
# Q-ViK oracle (on-sample teacher scores) on LLaVA-1.5-7B, total-basis keep, delayed replay.
# Usage: GPU=3 [KEEPS="0.5 0.2"] [TASKS=...] bash run_oracle_llava15.sh
set -uo pipefail
QVIK_ROOT=/workspace/nips/Q-ViK
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/workspace/nips/.hf_cache/datasets TMPDIR=/workspace/nips/.tmp
export OPENAI_API_URL="http://127.0.0.1:9/v1/chat/completions"
TASKS="${TASKS:-scienceqa_img,mme,mmbench_en_dev,mmstar,pope,vizwiz_vqa}"
cd "${QVIK_ROOT}"
for keep in ${KEEPS:-0.5 0.2}; do
  echo "[oracle] START keep=${keep} $(date -Is)"
  /opt/conda/envs/VFlowOpt/bin/python -u scripts_local/baselines/run_baselines_orig.py \
    --model lmms_llava15_oracle \
    --model_args "pretrained=/workspace/nips/models/llava-v1.5-7b,keep_ratio=${keep},keep_ratio_basis=total,device=cuda:0,device_map=cuda:0" \
    --tasks "${TASKS}" --batch_size 1 --log_samples --output_path "${QVIK_ROOT}/results/oracle_llava15"
  echo "[oracle] DONE keep=${keep} rc=$? $(date -Is)"
done
echo "ORACLE_DONE $(date -Is)"
