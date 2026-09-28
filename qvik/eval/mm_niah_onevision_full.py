"""Resumable MM-NIAH evaluation for original LLaVA-OneVision.

The runner evaluates the public text-needle tasks either with full KV caches or
with sequential Q-ViK visual eviction followed by text-only H2O eviction.  It
computes the exact prompt length after OneVision AnyRes image expansion and
only prefills samples that fit in the requested native window.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qvik.llava_onevision  # noqa: F401  # Register original LLaVA classes.
from kvpress.presses.visual_utility_student_onevision import (
    VisualUtilityStudentOneVision,
)
from qvik.eval.kv_decode_utils import trim_kv_cache_per_layer
from qvik.eval.mm_niah_scoring import score_mm_niah_answer
from qvik.eval.text_kv_eviction import (
    TextKVConfig,
    greedy_decode_with_text_eviction,
)
from qvik.llava_onevision.constants import IMAGE_TOKEN_INDEX
from qvik.llava_onevision.conversation import conv_templates
from qvik.llava_onevision.mm_utils import (
    get_anyres_image_grid_shape,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from qvik.llava_onevision.model.builder import load_pretrained_model

DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "eval" / "MM-NIAH" / "mm_niah_val"
TASKS = ("retrieval-text", "counting-text", "reasoning-text")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--image_root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "mm_niah_dev" / "images",
    )
    parser.add_argument(
        "--pretrained",
        default=str(WORKSPACE_ROOT / "models" / "llava-onevision-qwen2-7b-ov"),
    )
    parser.add_argument("--conv_template", default="qwen_1_5")
    parser.add_argument("--min_text_tokens", type=int, default=0)
    parser.add_argument("--max_text_tokens", type=int, default=0)
    parser.add_argument(
        "--max_expanded_tokens",
        type=int,
        default=32736,
        help="Maximum exact prompt tokens after AnyRes expansion.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--student_path",
        type=Path,
        default=REPO_ROOT / "ckpts" / "student_onevision",
    )
    parser.add_argument(
        "--visual_keep_ratio",
        type=float,
        default=1.0,
        help="Q-ViK image-token keep ratio. Use 1.0 for no visual eviction.",
    )
    parser.add_argument(
        "--text_eviction_mode",
        choices=("none", "streamingllm", "h2o"),
        default="none",
    )
    parser.add_argument("--text_keep_ratio", type=float, default=1.0)
    parser.add_argument("--h2o_recent_ratio", type=float, default=0.5)
    parser.add_argument("--streaming_sink_size", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results" / "mm_niah_onevision_full_cache_32k",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _rewrite_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_rows(args: argparse.Namespace, task: str) -> list[dict[str, Any]]:
    annotation = args.data_root / "annotations" / f"{task}.jsonl"
    if not annotation.is_file():
        raise FileNotFoundError(f"MM-NIAH annotation not found: {annotation}")
    rows: list[dict[str, Any]] = []
    with annotation.open() as handle:
        for line in handle:
            row = json.loads(line)
            text_tokens = int(row["meta"]["context_length_text"])
            if text_tokens < args.min_text_tokens:
                continue
            if args.max_text_tokens > 0 and text_tokens > args.max_text_tokens:
                continue
            rows.append(row)
    return rows


def _format_question(row: dict[str, Any]) -> str:
    question = str(row["question"]).strip()
    choices = row.get("meta", {}).get("choices")
    if choices:
        for idx, choice in enumerate(choices):
            question = f"{question}\n{chr(65 + idx)}. {choice}"
        question += "\nAnswer with the option's letter from the given choices directly."
    else:
        question += "\nAnswer the question using a single word or phrase."
    return f"{row['context']}\n{question}"


def _build_input_ids(
    tokenizer,
    conv_template: str,
    row: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    question = _format_question(row)
    num_images = len(row["images_list"])
    if question.count("<image>") != num_images:
        raise ValueError(
            f"Image placeholder mismatch: prompt={question.count('<image>')} "
            f"files={num_images}"
        )
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)
    placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    if placeholders != num_images:
        raise ValueError(
            f"Tokenized image placeholder mismatch: tokens={placeholders} files={num_images}"
        )
    return input_ids.to(device)


def _open_images(row: dict[str, Any], image_root: Path) -> list[Image.Image]:
    paths = [image_root / relative for relative in row["images_list"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} MM-NIAH image(s); first missing: {missing[0]}"
        )
    return [Image.open(path).convert("RGB") for path in paths]


def _move_images(image_tensor, device: torch.device):
    if isinstance(image_tensor, list):
        return [tensor.to(device=device, dtype=torch.float16) for tensor in image_tensor]
    return image_tensor.to(device=device, dtype=torch.float16)


def _onevision_visual_token_count(model, image_size: tuple[int, int]) -> int:
    """Return the exact spatial-unpad token count without running the encoder."""
    config = model.config
    image_aspect_ratio = str(getattr(config, "image_aspect_ratio", ""))
    patch_merge_type = str(getattr(config, "mm_patch_merge_type", ""))
    if "anyres" not in image_aspect_ratio or "unpad" not in patch_merge_type:
        raise ValueError(
            "Analytic visual length requires OneVision anyres + spatial_unpad; "
            f"got image_aspect_ratio={image_aspect_ratio!r}, "
            f"mm_patch_merge_type={patch_merge_type!r}"
        )

    vision_tower = model.get_vision_tower()
    unit = int(vision_tower.num_patches_per_side)
    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        image_size,
        config.image_grid_pinpoints,
        int(vision_tower.image_size),
    )
    current_height = int(num_patch_height) * unit
    current_width = int(num_patch_width) * unit
    original_width, original_height = image_size

    if original_width / original_height > current_width / current_height:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        current_height -= 2 * padding
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        current_width -= 2 * padding

    matched_max = re.search(r"anyres_max_(\d+)", image_aspect_ratio)
    if matched_max:
        max_num_patches = int(matched_max.group(1))
        downsample = math.sqrt(
            current_height * current_width / (max_num_patches * unit**2)
        )
        if downsample > 1.1:
            current_height = int(current_height // downsample)
            current_width = int(current_width // downsample)

    # spatial_unpad keeps the base 27x27 view and appends one newline token to
    # every row in the unpadded AnyRes grid.
    return unit**2 + current_height * (current_width + 1)


def _prepare_exact_multimodal(
    model,
    image_processor,
    input_ids: torch.Tensor,
    images: list[Image.Image],
    device: torch.device,
    max_expanded_tokens: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    bool,
]:
    """Encode each image independently and assemble the exact prompt embeddings.

    The upstream helper concatenates every AnyRes crop from every image into a
    single vision batch.  Multi-image MM-NIAH rows can make that temporary
    activation larger than the language-model prefill.  Sequential encoding is
    mathematically equivalent because the vision tower is image-independent.
    """
    input_ids_1d = input_ids[0]
    placeholder_positions = (
        (input_ids_1d == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()
    )
    if len(placeholder_positions) != len(images):
        raise ValueError(
            f"Image placeholder mismatch: tokens={len(placeholder_positions)} "
            f"images={len(images)}"
        )

    num_text_tokens = int(input_ids_1d.numel()) - len(placeholder_positions)
    visual_features: list[torch.Tensor] = []
    num_visual_tokens = 0
    dummy_ids = torch.tensor(
        [[IMAGE_TOKEN_INDEX, 0]],
        dtype=torch.long,
        device=device,
    )
    dummy_attention = torch.ones_like(dummy_ids, dtype=torch.bool)

    for image in images:
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = _move_images(image_tensor, device)

        # The original helper silently truncates at tokenizer_model_max_length.
        # Disable that truncation so filtering sees the true expanded length.
        original_max = getattr(model.config, "tokenizer_model_max_length", None)
        model.config.tokenizer_model_max_length = None
        try:
            _, _, _, _, one_image_embeds, _ = (
                model.prepare_inputs_labels_for_multimodal(
                    dummy_ids,
                    None,
                    dummy_attention,
                    None,
                    None,
                    image_tensor,
                    ["image"],
                    [image.size],
                )
            )
        finally:
            model.config.tokenizer_model_max_length = original_max

        if one_image_embeds is None or one_image_embeds.shape[1] < 2:
            raise RuntimeError("OneVision did not produce image embeddings.")
        # The dummy prompt is [<image>, token_0], so its last embedding is the
        # text anchor and every preceding embedding belongs to the image.
        visual_feature = one_image_embeds[0, :-1]
        visual_features.append(visual_feature)
        num_visual_tokens += int(visual_feature.shape[0])
        del image_tensor, one_image_embeds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        expanded_lower_bound = num_text_tokens + num_visual_tokens
        if expanded_lower_bound > max_expanded_tokens:
            del visual_features
            return None, None, None, None, expanded_lower_bound, False

    pieces: list[torch.Tensor] = []
    image_position_chunks: list[torch.Tensor] = []
    cursor = 0
    expanded_cursor = 0
    for position, visual_feature in zip(placeholder_positions, visual_features):
        if position > cursor:
            text_piece = model.get_model().embed_tokens(input_ids_1d[cursor:position])
            pieces.append(text_piece)
            expanded_cursor += int(text_piece.shape[0])
        pieces.append(visual_feature)
        image_position_chunks.append(
            torch.arange(
                expanded_cursor,
                expanded_cursor + int(visual_feature.shape[0]),
                dtype=torch.long,
            )
        )
        expanded_cursor += int(visual_feature.shape[0])
        cursor = position + 1
    if cursor < input_ids_1d.numel():
        text_piece = model.get_model().embed_tokens(input_ids_1d[cursor:])
        pieces.append(text_piece)
        expanded_cursor += int(text_piece.shape[0])
    inputs_embeds = torch.cat(pieces, dim=0).unsqueeze(0)
    expanded_tokens = int(inputs_embeds.shape[1])
    if expanded_tokens != num_text_tokens + num_visual_tokens:
        raise RuntimeError(
            f"Expanded length mismatch: tensor={expanded_tokens} "
            f"text={num_text_tokens} visual={num_visual_tokens}"
        )
    expanded_attention_mask = torch.ones(
        (1, expanded_tokens),
        dtype=torch.bool,
        device=device,
    )
    image_positions = torch.cat(image_position_chunks)
    if int(image_positions.numel()) != num_visual_tokens:
        raise RuntimeError(
            f"Visual position mismatch: positions={image_positions.numel()} "
            f"features={num_visual_tokens}"
        )
    return (
        inputs_embeds,
        None,
        expanded_attention_mask,
        image_positions,
        expanded_tokens,
        True,
    )


def _eos_token_ids(model, tokenizer) -> set[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


@torch.no_grad()
def _full_cache_generate(
    model,
    tokenizer,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()

    # Calling LlavaQwenForCausalLM.forward on a long prefill materializes
    # [sequence, vocabulary] logits.  Run the Qwen backbone directly and apply
    # lm_head only to the final hidden state; the KV cache remains completely
    # unmodified/full.
    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    last_hidden = outputs.last_hidden_state[:, -1:, :]
    first_next_token = model.lm_head(last_hidden).argmax(dim=-1)
    past_kv = outputs.past_key_values
    prompt_len = int(inputs_embeds.shape[1])
    del outputs, last_hidden, inputs_embeds
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    decode_start = time.perf_counter()
    eos_ids = _eos_token_ids(model, tokenizer)
    out_tokens = [int(first_next_token.item())]
    next_token = first_next_token
    position = prompt_len
    cache_position = torch.empty(1, dtype=torch.long, device=next_token.device)
    for _ in range(max_new_tokens - 1):
        if out_tokens[-1] in eos_ids:
            break
        cache_position[0] = position
        step = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=cache_position,
            position_ids=cache_position.unsqueeze(0),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past_kv = step.past_key_values
        next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_tokens.append(int(next_token.item()))
        position += 1
        del step
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start
    del past_kv

    prediction = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
    return prediction, prefill_seconds, decode_seconds


@torch.no_grad()
def _qvik_h2o_generate(
    model,
    tokenizer,
    student: VisualUtilityStudentOneVision,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    image_positions: torch.Tensor,
    max_new_tokens: int,
    visual_keep_ratio: float,
    text_config: TextKVConfig,
) -> tuple[str, float, float, dict[str, Any]]:
    """Prefill once, apply Q-ViK per layer, then decode with text H2O."""
    prompt_len = int(inputs_embeds.shape[1])
    image_positions_cpu = image_positions.detach().cpu().long()
    image_positions_device = image_positions_cpu.to(inputs_embeds.device)
    num_visual_tokens = int(image_positions_cpu.numel())
    num_visual_kept = max(1, int(math.ceil(num_visual_tokens * visual_keep_ratio)))
    last_image_position = int(image_positions_cpu.max().item())
    question_positions = torch.arange(
        last_image_position + 1,
        prompt_len,
        dtype=torch.long,
        device=inputs_embeds.device,
    )

    decoder_layers = model.model.layers
    missing_layers = [
        layer_idx
        for layer_idx in student.layer_indices
        if layer_idx >= len(decoder_layers)
    ]
    if missing_layers:
        raise ValueError(
            f"Student layer indices exceed model depth {len(decoder_layers)}: "
            f"{missing_layers}"
        )

    layer_scores: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx in student.layer_indices:
        student_layer = student.layers[str(layer_idx)]

        def _score_hook(_module, _inputs, output, *, li=layer_idx, scorer=student_layer):
            hidden_states = output[0] if isinstance(output, tuple) else output
            scores = scorer(
                hidden_states,
                image_positions_device,
                question_positions,
            ).squeeze(0)
            layer_scores[li] = scores.detach()

        hooks.append(decoder_layers[layer_idx].register_forward_hook(_score_hook))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    try:
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    if len(layer_scores) != len(student.layer_indices):
        raise RuntimeError(
            f"Q-ViK scored {len(layer_scores)} layers, expected "
            f"{len(student.layer_indices)}."
        )
    last_hidden = outputs.last_hidden_state[:, -1:, :]
    first_next_token = model.lm_head(last_hidden).argmax(dim=-1)
    past_kv = outputs.past_key_values

    keep_masks: dict[int, torch.Tensor] = {}
    if num_visual_kept < num_visual_tokens:
        for layer_idx, scores in layer_scores.items():
            top_indices = torch.topk(
                scores,
                k=num_visual_kept,
                largest=True,
            ).indices.detach().cpu()
            mask = torch.ones(prompt_len, dtype=torch.bool)
            visual_keep = torch.zeros(num_visual_tokens, dtype=torch.bool)
            visual_keep[top_indices] = True
            mask[image_positions_cpu] = visual_keep
            keep_masks[layer_idx] = mask

    del outputs, last_hidden, layer_scores, inputs_embeds
    past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start

    decode_start = time.perf_counter()
    eos_token_id = int(tokenizer.eos_token_id)
    answer_ids, text_stats = greedy_decode_with_text_eviction(
        model,
        past_kv,
        first_next_token,
        prompt_len=prompt_len,
        image_positions=image_positions_cpu,
        visual_keep_masks=keep_masks,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        config=text_config,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start
    del past_kv

    stats: dict[str, Any] = {
        "n_image_original": num_visual_tokens,
        "n_image_kept": num_visual_kept,
        "image_keep_ratio": num_visual_kept / max(1, num_visual_tokens),
        "n_text_prompt_original": prompt_len - num_visual_tokens,
        **text_stats,
    }
    prediction = tokenizer.decode(
        answer_ids.tolist(),
        skip_special_tokens=True,
    ).strip()
    return prediction, prefill_seconds, decode_seconds, stats


def _uses_eviction(args: argparse.Namespace) -> bool:
    return (
        args.visual_keep_ratio < 1.0
        or args.text_eviction_mode != "none"
    )


def _run_tag(args: argparse.Namespace) -> str:
    if not _uses_eviction(args):
        return "full"
    return (
        f"qvik{args.visual_keep_ratio:g}_"
        f"{args.text_eviction_mode}_text{args.text_keep_ratio:g}"
    )


def _write_summary(
    args: argparse.Namespace,
    task: str,
    rows: list[dict[str, Any]],
    predictions_path: Path,
    excluded_path: Path,
    official_predictions_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    records = _read_jsonl(predictions_path)
    excluded = _read_jsonl(excluded_path)
    exact = sum(bool(record["exact"]) for record in records)
    contains = sum(bool(record["answer_contained"]) for record in records)
    official_score = sum(float(record["mm_niah_score"]) for record in records)
    errors = sum(record.get("error") is not None for record in records)
    completed_ids = {record["id"] for record in records}
    excluded_ids = {record["id"] for record in excluded}

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

    n = len(records)
    cache_records = [
        record["cache_stats"]
        for record in records
        if record.get("cache_stats")
    ]
    summary = {
        "task": task,
        "n_annotation_rows": len(rows),
        "n_samples": n,
        "n_excluded_over_length": len(excluded),
        "n_errors": errors,
        "n_unfinished": len(rows) - len(completed_ids | excluded_ids),
        "exact_match": exact / n if n else 0.0,
        "answer_contained": contains / n if n else 0.0,
        "mm_niah_accuracy": official_score / n if n else 0.0,
        "expanded_prompt_tokens": {
            "min": min((record["expanded_prompt_tokens"] for record in records), default=0),
            "max": max((record["expanded_prompt_tokens"] for record in records), default=0),
        },
        "timing": {
            "avg_prefill_seconds": (
                sum(record.get("prefill_seconds", 0.0) for record in records) / n
                if n
                else 0.0
            ),
            "avg_decode_seconds": (
                sum(record.get("decode_seconds", 0.0) for record in records) / n
                if n
                else 0.0
            ),
        },
        "cache_stats": {
            "avg_image_keep_ratio": (
                sum(record["image_keep_ratio"] for record in cache_records)
                / len(cache_records)
                if cache_records
                else 1.0
            ),
            "avg_text_prompt_keep_ratio_after_eviction": (
                sum(
                    record["text_prompt_keep_ratio_after_eviction"]
                    for record in cache_records
                )
                / len(cache_records)
                if cache_records
                else 1.0
            ),
            "avg_n_image_original": (
                sum(record["n_image_original"] for record in cache_records)
                / len(cache_records)
                if cache_records
                else 0.0
            ),
            "avg_n_image_kept": (
                sum(record["n_image_kept"] for record in cache_records)
                / len(cache_records)
                if cache_records
                else 0.0
            ),
            "avg_n_text_prompt_original": (
                sum(record["n_text_prompt_original"] for record in cache_records)
                / len(cache_records)
                if cache_records
                else 0.0
            ),
            "avg_n_text_prompt_kept_after_eviction": (
                sum(
                    record["avg_n_text_prompt_kept_after_eviction"]
                    for record in cache_records
                )
                / len(cache_records)
                if cache_records
                else 0.0
            ),
        },
        "official_predictions": str(official_predictions_path),
        "config": {
            **vars(args),
            "data_root": str(args.data_root),
            "image_root": str(args.image_root),
            "output_dir": str(args.output_dir),
            "student_path": str(args.student_path),
            "tasks": list(args.tasks),
            "cache_mode": "sequential" if _uses_eviction(args) else "full",
            "visual_eviction": (
                "qvik" if args.visual_keep_ratio < 1.0 else "none"
            ),
            "text_eviction": args.text_eviction_mode,
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def _run_task(
    args: argparse.Namespace,
    task: str,
    model,
    tokenizer,
    image_processor,
    device: torch.device,
    student: VisualUtilityStudentOneVision | None,
    text_config: TextKVConfig,
) -> None:
    rows = _load_rows(args, task)
    run_tag = _run_tag(args)
    output_dir = args.output_dir / f"{task}_onevision_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    excluded_path = output_dir / "excluded_over_length.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    official_dir = output_dir / "official_outputs"
    official_dir.mkdir(parents=True, exist_ok=True)
    official_predictions_path = (
        official_dir / f"llava-onevision-qwen2-7b-ov-{run_tag}_{task}-val.jsonl"
    )

    previous_records = _read_jsonl(predictions_path)
    retry_records = [record for record in previous_records if record.get("error") is not None]
    if retry_records:
        previous_records = [
            record for record in previous_records if record.get("error") is None
        ]
        _rewrite_jsonl(predictions_path, previous_records)
        print(
            f"[MM-NIAH OneVision {run_tag}] retrying "
            f"{len(retry_records)} prior error(s) "
            f"for {task}",
            flush=True,
        )
    completed = {record["id"] for record in previous_records}
    excluded = {record["id"] for record in _read_jsonl(excluded_path)}
    done = completed | excluded
    generated_this_run = 0

    progress = tqdm(
        desc=f"{task} {run_tag}",
        initial=len(done),
        total=len(rows),
    )
    for row in rows:
        if row["id"] in done:
            continue
        if args.limit > 0 and generated_this_run >= args.limit:
            break

        error = None
        prediction = ""
        expanded_tokens = 0
        num_visual_tokens = 0
        prefill_seconds = 0.0
        decode_seconds = 0.0
        cache_stats: dict[str, Any] = {}
        sample_start = time.perf_counter()
        images: list[Image.Image] = []
        try:
            input_ids = _build_input_ids(tokenizer, args.conv_template, row, device)
            unexpanded_tokens = int(input_ids.numel())
            num_placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
            num_text_tokens = unexpanded_tokens - num_placeholders

            # Image expansion can only increase length, so these rows are
            # definitely outside the requested window without encoding images.
            if unexpanded_tokens > args.max_expanded_tokens:
                _append_jsonl(
                    excluded_path,
                    {
                        "id": row["id"],
                        "reason": "unexpanded_prompt_over_limit",
                        "unexpanded_prompt_tokens": unexpanded_tokens,
                        "expanded_prompt_tokens": None,
                        "context_length_text": row["meta"]["context_length_text"],
                    },
                )
                done.add(row["id"])
                progress.update(1)
                continue

            images = _open_images(row, args.image_root)
            expected_visual_tokens = sum(
                _onevision_visual_token_count(model, image.size) for image in images
            )
            expected_expanded_tokens = num_text_tokens + expected_visual_tokens
            if expected_expanded_tokens > args.max_expanded_tokens:
                _append_jsonl(
                    excluded_path,
                    {
                        "id": row["id"],
                        "reason": "expanded_prompt_over_limit",
                        "unexpanded_prompt_tokens": unexpanded_tokens,
                        "expanded_prompt_tokens": expected_expanded_tokens,
                        "expanded_length_is_exact": True,
                        "num_visual_tokens": expected_visual_tokens,
                        "context_length_text": row["meta"]["context_length_text"],
                    },
                )
                done.add(row["id"])
                progress.update(1)
                continue

            (
                inputs_embeds,
                position_ids,
                expanded_attention_mask,
                image_positions,
                expanded_tokens,
                exact_expanded_length,
            ) = _prepare_exact_multimodal(
                model,
                image_processor,
                input_ids,
                images,
                device,
                args.max_expanded_tokens,
            )
            num_visual_tokens = expanded_tokens - num_text_tokens
            if (
                expanded_tokens != expected_expanded_tokens
                or num_visual_tokens != expected_visual_tokens
            ):
                raise RuntimeError(
                    "Analytic/encoded OneVision length mismatch: "
                    f"expanded={expanded_tokens}/{expected_expanded_tokens} "
                    f"visual={num_visual_tokens}/{expected_visual_tokens}"
                )
            if expanded_tokens > args.max_expanded_tokens:
                _append_jsonl(
                    excluded_path,
                    {
                        "id": row["id"],
                        "reason": "expanded_prompt_over_limit",
                        "unexpanded_prompt_tokens": unexpanded_tokens,
                        "expanded_prompt_tokens": expanded_tokens,
                        "expanded_length_is_exact": exact_expanded_length,
                        "num_visual_tokens": num_visual_tokens,
                        "context_length_text": row["meta"]["context_length_text"],
                    },
                )
                del inputs_embeds
                done.add(row["id"])
                progress.update(1)
                continue

            if inputs_embeds is None:
                raise RuntimeError("Eligible prompt has no prepared input embeddings.")
            if image_positions is None:
                raise RuntimeError("Eligible prompt has no visual position map.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            if _uses_eviction(args):
                if student is None:
                    raise RuntimeError(
                        "Q-ViK/H2O run requires the OneVision student."
                    )
                (
                    prediction,
                    prefill_seconds,
                    decode_seconds,
                    cache_stats,
                ) = _qvik_h2o_generate(
                    model,
                    tokenizer,
                    student,
                    inputs_embeds,
                    position_ids,
                    expanded_attention_mask,
                    image_positions,
                    args.max_new_tokens,
                    args.visual_keep_ratio,
                    text_config,
                )
            else:
                prediction, prefill_seconds, decode_seconds = _full_cache_generate(
                    model,
                    tokenizer,
                    inputs_embeds,
                    position_ids,
                    expanded_attention_mask,
                    args.max_new_tokens,
                )
        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            for image in images:
                image.close()

        reference = row["answer"]
        normalized_prediction = prediction.strip().casefold()
        normalized_reference = str(reference).strip().casefold()
        sample_score = score_mm_niah_answer(task, prediction, reference)
        record = {
            "id": row["id"],
            "prediction": prediction,
            "answer": reference,
            "exact": normalized_prediction == normalized_reference,
            "answer_contained": bool(sample_score),
            "mm_niah_score": sample_score,
            "context_length_text": row["meta"]["context_length_text"],
            "context_length": row["meta"]["context_length"],
            "expanded_prompt_tokens": expanded_tokens,
            "num_visual_tokens": num_visual_tokens,
            "num_images": row["meta"]["num_images"],
            "placed_depth": row["meta"]["placed_depth"],
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "cache_stats": cache_stats,
            "elapsed_seconds": time.perf_counter() - sample_start,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            ),
            "error": error,
        }
        _append_jsonl(predictions_path, record)
        done.add(row["id"])
        generated_this_run += 1
        progress.update(1)
        progress.set_postfix(
            tokens=expanded_tokens,
            score=f"{sample_score:.2f}",
            prefill=f"{prefill_seconds:.2f}s",
        )

        with progress_path.open("w") as handle:
            json.dump(
                {
                    "task": task,
                    "completed": len(done),
                    "total": len(rows),
                    "last_id": row["id"],
                },
                handle,
                indent=2,
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress.close()
    summary = _write_summary(
        args,
        task,
        rows,
        predictions_path,
        excluded_path,
        official_predictions_path,
        summary_path,
    )
    print(
        f"[MM-NIAH OneVision {run_tag}] task={task} n={summary['n_samples']} "
        f"excluded={summary['n_excluded_over_length']} "
        f"unfinished={summary['n_unfinished']} errors={summary['n_errors']} "
        f"score={summary['mm_niah_accuracy']:.4f} -> {summary_path}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.visual_keep_ratio <= 1.0:
        raise ValueError("--visual_keep_ratio must be in (0, 1].")
    text_config = TextKVConfig(
        mode=args.text_eviction_mode,
        keep_ratio=args.text_keep_ratio,
        h2o_recent_ratio=args.h2o_recent_ratio,
        streaming_sink_size=args.streaming_sink_size,
    ).normalized()
    device = torch.device(args.device)
    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.pretrained,
        None,
        model_name,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        multimodal=True,
    )
    model = model.eval()
    if args.max_expanded_tokens + args.max_new_tokens > context_len:
        raise ValueError(
            "Prompt plus generation exceeds native context: "
            f"{args.max_expanded_tokens}+{args.max_new_tokens}>{context_len}"
        )
    student = None
    if _uses_eviction(args):
        student = VisualUtilityStudentOneVision.from_pretrained(args.student_path)
        student = student.to(device=device, dtype=torch.float16).eval()
    run_tag = _run_tag(args)
    print(
        f"[MM-NIAH OneVision {run_tag}] model={model_name} context={context_len} "
        f"prompt_limit={args.max_expanded_tokens} max_new={args.max_new_tokens} "
        f"visual_keep={args.visual_keep_ratio:g} "
        f"text={args.text_eviction_mode}:{args.text_keep_ratio:g} "
        f"tasks={','.join(args.tasks)}",
        flush=True,
    )
    for task in args.tasks:
        _run_task(
            args,
            task,
            model,
            tokenizer,
            image_processor,
            device,
            student,
            text_config,
        )


if __name__ == "__main__":
    main()
