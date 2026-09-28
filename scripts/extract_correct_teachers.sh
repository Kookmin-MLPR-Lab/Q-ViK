#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export CUDA_VISIBLE_DEVICES=0

for dataset in scienceqa textvqa gqa; do
  python qvik/teacher/extract_llava15.py \
    --dataset "$dataset" \
    --device cuda:0 \
    --n-samples 300 \
    --require-correct
done

python - <<'PY'
from pathlib import Path

root = Path.cwd().parent / "data/train/teacher/llava15"
counts = {
    dataset: len(list((root / dataset).glob("*.pt")))
    for dataset in ("scienceqa", "textvqa", "gqa")
}
if counts != {"scienceqa": 300, "textvqa": 300, "gqa": 300}:
    raise SystemExit(f"teacher count mismatch: {counts}")
print(f"teacher extraction complete: {counts}, total={sum(counts.values())}")
PY
