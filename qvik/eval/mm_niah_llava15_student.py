"""Standalone MM-NIAH text-needle evaluation for Q-ViK LLaVA-1.5.

This runner targets the public validation annotations:
retrieval-text, counting-text, and reasoning-text.  It is mainly useful for the
native <=2K LLaVA-1.5 sanity slice; use a long-context model for the full set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qvik.eval.lmms_llava15_student import LmmsLlava15Student
from qvik.eval.mm_niah_scoring import score_mm_niah_answer
from qvik.llava15.constants import IMAGE_TOKEN_INDEX
from qvik.llava15.mm_utils import tokenizer_image_token

DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "eval" / "MM-NIAH" / "mm_niah_val"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("retrieval-text", "counting-text", "reasoning-text"),
        required=True,
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--image_root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "mm_niah_dev" / "images",
    )
    parser.add_argument(
        "--pretrained",
        default=str(WORKSPACE_ROOT / "models" / "llava-v1.5-7b"),
    )
    parser.add_argument(
        "--student_path",
        default=str(REPO_ROOT / "ckpts" / "student_llava15_900_e15"),
    )
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument(
        "--keep_ratio_basis",
        choices=("total", "image"),
        default="image",
    )
    parser.add_argument(
        "--text_eviction_mode",
        choices=("none", "streamingllm", "h2o"),
        default="none",
    )
    parser.add_argument("--text_keep_ratio", type=float, default=0.2)
    parser.add_argument("--text_cache_size", type=int, default=0)
    parser.add_argument("--h2o_recent_ratio", type=float, default=0.5)
    parser.add_argument("--streaming_sink_size", type=int, default=4)
    parser.add_argument("--min_text_tokens", type=int, default=0)
    parser.add_argument("--max_text_tokens", type=int, default=0)
    parser.add_argument(
        "--max_expanded_tokens",
        type=int,
        default=2048,
        help="Exact tokenized prompt limit after chat templating and image expansion.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results" / "mm_niah")
    return parser.parse_args()


def _load_rows(args: argparse.Namespace) -> list[dict]:
    annotation = args.data_root / "annotations" / f"{args.task}.jsonl"
    if not annotation.is_file():
        raise FileNotFoundError(f"MM-NIAH annotation not found: {annotation}")
    with annotation.open() as handle:
        rows = [json.loads(line) for line in handle]

    filtered: list[dict] = []
    for row in rows:
        text_tokens = int(row["meta"]["context_length_text"])
        if text_tokens < args.min_text_tokens:
            continue
        if args.max_text_tokens > 0 and text_tokens > args.max_text_tokens:
            continue
        filtered.append(row)
    return filtered


def _format_question(row: dict) -> str:
    question = str(row["question"]).strip()
    choices = row.get("meta", {}).get("choices")
    if choices:
        for idx, choice in enumerate(choices):
            question = f"{question}\n{chr(65 + idx)}. {choice}"
        question += "\nAnswer with the option's letter from the given choices directly."
    else:
        question += "\nAnswer the question using a single word or phrase."
    return f"{row['context']}\n{question}"


def _resolve_images(row: dict, image_root: Path) -> list[Image.Image]:
    paths = [image_root / relative for relative in row["images_list"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} MM-NIAH image(s); first missing: {missing[0]}"
        )
    return [Image.open(path).convert("RGB") for path in paths]


def _expanded_prompt_tokens(wrapper: LmmsLlava15Student, row: dict) -> int:
    num_images = len(row["images_list"])
    prompt = wrapper._build_prompt(_format_question(row), [None] * num_images)
    input_ids = tokenizer_image_token(
        prompt,
        wrapper.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    num_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    if num_placeholders != num_images:
        raise ValueError(
            f"Image placeholder mismatch: prompt={num_placeholders} files={num_images}"
        )
    return int(input_ids.numel()) + num_placeholders * (wrapper.image_feature_len - 1)


@torch.no_grad()
def _infer(wrapper: LmmsLlava15Student, row: dict, image_root: Path, max_new_tokens: int) -> str:
    images = _resolve_images(row, image_root)
    try:
        question = _format_question(row)
        if question.count("<image>") != len(images):
            raise ValueError(
                f"Image placeholder mismatch: prompt={question.count('<image>')} "
                f"files={len(images)}"
            )
        prompt = wrapper._build_prompt(question, images)
        input_ids = tokenizer_image_token(
            prompt,
            wrapper.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(wrapper.device)
        image_tensor, image_sizes = wrapper._prepare_images(images)
        return wrapper._generate_with_student(
            input_ids=input_ids,
            image_tensor=image_tensor,
            image_sizes=image_sizes,
            num_images=len(images),
            max_new_tokens=max_new_tokens,
        )
    finally:
        for image in images:
            image.close()


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args)
    if not rows:
        raise RuntimeError("No MM-NIAH rows remain after applying length filters.")

    run_name = (
        f"{args.task}_qvik{args.keep_ratio:g}_{args.keep_ratio_basis}_"
        f"{args.text_eviction_mode}"
    )
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    official_dir = output_dir / "official_outputs"
    official_dir.mkdir(parents=True, exist_ok=True)
    model_tag = (
        f"llava15-qvik{args.keep_ratio:g}-{args.keep_ratio_basis}-"
        f"{args.text_eviction_mode}-text{args.text_keep_ratio:g}"
    )
    official_predictions_path = official_dir / f"{model_tag}_{args.task}-val.jsonl"

    wrapper = LmmsLlava15Student(
        pretrained=args.pretrained,
        student_path=args.student_path,
        keep_ratio=args.keep_ratio,
        keep_ratio_basis=args.keep_ratio_basis,
        text_eviction_mode=args.text_eviction_mode,
        text_keep_ratio=args.text_keep_ratio,
        text_cache_size=args.text_cache_size,
        h2o_recent_ratio=args.h2o_recent_ratio,
        streaming_sink_size=args.streaming_sink_size,
        device=args.device,
        device_map=args.device,
        max_new_tokens=args.max_new_tokens,
        stats_output_dir=str(output_dir),
    )

    expanded_lengths = {
        row["id"]: _expanded_prompt_tokens(wrapper, row)
        for row in tqdm(rows, desc=f"{args.task} length filter")
    }
    if args.max_expanded_tokens > 0:
        rows = [
            row
            for row in rows
            if expanded_lengths[row["id"]] <= args.max_expanded_tokens
        ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No MM-NIAH rows remain after exact token-length filtering.")
    print(
        f"[MM-NIAH] task={args.task} samples={len(rows)} "
        f"text_eviction={args.text_eviction_mode} "
        f"text_range=[{args.min_text_tokens or 0},"
        f"{args.max_text_tokens or 'inf'}] max_expanded={args.max_expanded_tokens} "
        f"actual_expanded=[{min(expanded_lengths[row['id']] for row in rows)},"
        f"{max(expanded_lengths[row['id']] for row in rows)}]"
    )

    records: list[dict] = []
    exact = 0
    contains = 0
    official_score = 0.0
    for row in tqdm(rows, desc=args.task):
        prediction = ""
        error = None
        try:
            prediction = _infer(wrapper, row, args.image_root, args.max_new_tokens)
        except (FileNotFoundError, ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            error = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reference = row["answer"]
        normalized_prediction = prediction.strip().casefold()
        normalized_reference = str(reference).strip().casefold()
        is_exact = normalized_prediction == normalized_reference
        sample_score = score_mm_niah_answer(args.task, prediction, reference)
        is_contained = bool(sample_score)
        exact += int(is_exact)
        contains += int(is_contained)
        official_score += sample_score
        records.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "answer": reference,
                "exact": is_exact,
                "answer_contained": is_contained,
                "mm_niah_score": sample_score,
                "context_length_text": row["meta"]["context_length_text"],
                "context_length": row["meta"]["context_length"],
                "expanded_prompt_tokens": expanded_lengths[row["id"]],
                "num_images": row["meta"]["num_images"],
                "placed_depth": row["meta"]["placed_depth"],
                "error": error,
            }
        )

    with predictions_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with official_predictions_path.open("w") as handle:
        for record in records:
            official_record = {
                "question_id": record["id"],
                "answer": record["answer"],
                "response": record["prediction"],
                "context_length": record["context_length"],
                "placed_depth": record["placed_depth"],
            }
            handle.write(json.dumps(official_record, ensure_ascii=False) + "\n")

    wrapper._save_keep_stats(args.task)
    summary = {
        "task": args.task,
        "n_samples": len(records),
        "exact_match": exact / len(records),
        "answer_contained": contains / len(records),
        "mm_niah_accuracy": official_score / len(records),
        "official_predictions": str(official_predictions_path),
        "config": vars(args) | {
            "data_root": str(args.data_root),
            "image_root": str(args.image_root),
            "output_dir": str(args.output_dir),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"[MM-NIAH] accuracy={summary['mm_niah_accuracy']:.4f} "
        f"exact={summary['exact_match']:.4f} -> {summary_path}"
    )


if __name__ == "__main__":
    main()
