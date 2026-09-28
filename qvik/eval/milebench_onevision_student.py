
"""Standalone MileBench evaluation using LmmsOnevisionStudent.

Outputs pred.json (sample_id / pred_response / gt_response) for MileBench scoring.
Results are written to <output_dir>/<dataset>/pred.json; the default
output_dir matches the lmms-eval layout (results/onevision_milebench/).

Usage:
    python qvik/eval/milebench_onevision_student.py \
        --dataset ActionLocalization \
        --keep_ratio 0.5 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = str(WORKSPACE_ROOT / "data" / "eval" / "MileBench")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results" / "onevision_milebench")
LOG_ROOT = str(REPO_ROOT / "logs" / "onevision_milebench")
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_NEW_TOKENS = 32


class _Tee:
    """Write to multiple streams at once (stdout + per-dataset log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def _tee_to_file(log_path: Path):
    """Mirror stdout/stderr to log_path for the duration of the block."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, f), _Tee(old_err, f)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


def build_prompt(sample: dict, meta: dict) -> str:
    ann = sample["task_instance"]
    task_instruction = meta["task_instruction"][sample["task_instruction_id"]]

    context = ann["context"]
    n_img = len(ann["images_path"])
    for i in range(1, n_img + 1):
        context = context.replace(f"{{image#{i}}}", f"<Image {i}> ")
        context = context.replace(f"{{table#{i}}}", f"<Image {i}> ")

    if ann.get("choice_list"):
        choice_str = "\nChoice List:\n"
        choice_str += "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ann["choice_list"]))
        choice_str += "\nYour answer is: "
        context += choice_str

    return f"{DEFAULT_IMAGE_TOKEN}\n{task_instruction}\n{context}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--pretrained",
        default=str(WORKSPACE_ROOT / "models" / "llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument(
        "--student_path",
        default=str(REPO_ROOT / "ckpts" / "student_onevision"),
    )
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument(
        "--text_eviction_mode",
        choices=("none", "streamingllm", "h2o"),
        default="none",
    )
    parser.add_argument("--text_keep_ratio", type=float, default=0.2)
    parser.add_argument("--text_cache_size", type=int, default=0)
    parser.add_argument("--h2o_recent_ratio", type=float, default=0.5)
    parser.add_argument("--streaming_sink_size", type=int, default=4)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--log_root", default=LOG_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    log_file = Path(args.log_root) / f"{args.dataset}.log"
    with _tee_to_file(log_file):
        _run(args)


def _run(args):
    task_out = os.path.join(args.output_dir, args.dataset)
    pred_path = os.path.join(task_out, "pred.json")
    os.makedirs(task_out, exist_ok=True)

    if os.path.exists(pred_path) and not args.overwrite:
        print(f"[skip] {args.dataset}: {pred_path} exists")
        return

    # Load data
    data_path = os.path.join(args.data_root, args.dataset, f"{args.dataset}.json")
    data = json.load(open(data_path))
    meta = data["meta_data"]
    samples = data["data"]
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    samples = samples[args.start_index:]
    if args.limit is not None:
        samples = samples[:args.limit]
    combined_img_root = os.path.join(args.data_root, args.dataset, "combined_1_images")

    print(
        f"[{args.dataset}] {len(samples)} samples (start_index={args.start_index}) "
        f"| keep_ratio={args.keep_ratio} "
        f"keep_ratio_basis=image text_eviction={args.text_eviction_mode} "
        f"text_keep_ratio={args.text_keep_ratio} max_new_tokens={args.max_new_tokens}"
    )

    # Load student model (reuse existing class — no duplication)
    from qvik.eval.lmms_onevision_student import LmmsOnevisionStudent
    model_wrapper = LmmsOnevisionStudent(
        pretrained=args.pretrained,
        student_path=args.student_path,
        keep_ratio=args.keep_ratio,
        device=args.device,
        stats_output_dir=task_out,
        text_eviction_mode=args.text_eviction_mode,
        text_keep_ratio=args.text_keep_ratio,
        text_cache_size=args.text_cache_size,
        h2o_recent_ratio=args.h2o_recent_ratio,
        streaming_sink_size=args.streaming_sink_size,
    )

    predictions = []

    for sample in tqdm(samples, desc=args.dataset):
        ann = sample["task_instance"]
        prompt_text = build_prompt(sample, meta)

        img_file = ann["combined_1_images"][0]
        img_path = os.path.join(combined_img_root, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {img_path}: {e}", file=sys.stderr)
            predictions.append({
                "sample_id": sample["sample_id"],
                "pred_response": "",
                "gt_response": sample["response"],
            })
            continue

        try:
            answer = model_wrapper._generate_llava(prompt_text, [image], args.max_new_tokens)
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"[warn] OOM on {args.dataset} sample {sample['sample_id']}: {e}",
                file=sys.stderr,
                flush=True,
            )
            torch.cuda.empty_cache()
            answer = ""

        predictions.append({
            "sample_id": sample["sample_id"],
            "pred_response": answer,
            "gt_response": sample["response"],
        })
        del image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pred.json
    json.dump(predictions, open(pred_path, "w"), ensure_ascii=False, indent=2)
    print(f"[{args.dataset}] saved → {pred_path}")

    # Save keep ratio stats
    model_wrapper._save_keep_stats(args.dataset)


if __name__ == "__main__":
    main()
