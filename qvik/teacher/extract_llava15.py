#!/usr/bin/env python3
"""Stage 1 teacher cache for original LLaVA-1.5-7B (LlavaLlamaForCausalLM).

Uses the original haotian-liu/LLaVA repo format instead of the HF wrapper.
Key differences from collect_llava15.py:
  - Model: LlavaLlamaForCausalLM.from_pretrained (not LlavaForConditionalGeneration)
  - Tokenizer: AutoTokenizer with tokenizer_image_token utility
  - Image processor: vision_tower.image_processor (CLIP)
  - Image token: IMAGE_TOKEN_INDEX=-200 placeholder in input_ids, expanded to 576

Output layout: <output_root>/<dataset>/<sample_id>.pt with the same schema
as collect_llava15.py (teacher_raw, teacher_norm, image_token_indices, etc.)
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qvik.teacher.answer_correctness import is_correct_prediction
# torch 2.5 + transformers 5.3 incompatibility: patch bin-load safety check.
try:
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# transformers 4.46+ GenerationConfig.from_model_config calls .to_dict() on
# nested config objects that may be plain dicts in older LLaVA checkpoints.
try:
    from transformers.generation import configuration_utils as _gen_cfg
    _orig_from_model_config = _gen_cfg.GenerationConfig.from_model_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_model_config(cls, model_config):
        for attr in ("decoder", "encoder", "text_config", "vision_config"):
            val = getattr(model_config, attr, None)
            if isinstance(val, dict):
                from types import SimpleNamespace
                ns = SimpleNamespace(**val)
                ns.to_dict = lambda _v=val: _v
                setattr(model_config, attr, ns)
        return _orig_from_model_config(cls, model_config)

    _gen_cfg.GenerationConfig.from_model_config = _patched_from_model_config
except Exception:
    pass

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
NUM_IMAGE_FEATURES = 576  # CLIP-ViT-L/14@336, 24×24 patches


def _patch_llava_arch_for_dynamic_cache():
    """Patch prepare_inputs_labels_for_multimodal to support new DynamicCache format."""
    import qvik.llava15.model.llava_arch as _arch
    import types
    original = _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

    def patched(self, input_ids, attention_mask, past_key_values, labels, images):
        vision_tower = self.get_vision_tower()
        if (vision_tower is None or images is None or input_ids.shape[1] == 1):
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                if hasattr(past_key_values, "get_seq_length"):
                    past_len = past_key_values.get_seq_length()
                else:
                    past_len = max([p[-1].shape[-2] for p in past_key_values])
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_len + 1),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        return original(self, input_ids, attention_mask, past_key_values, labels, images)

    _arch.LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = patched


def load_orig_llava_model(model_path: str, device: torch.device):
    """Load original LLaVA-1.5 model, tokenizer, and image processor."""
    from transformers import AutoTokenizer
    from qvik.llava15.model.language_model.llava_llama import LlavaLlamaForCausalLM
    _patch_llava_arch_for_dynamic_cache()

    print(f"[load] {model_path} dtype=bf16 device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # required for output_attentions=True in generate
    ).to(device).eval()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # Add special tokens if needed
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens(["<im_patch>"], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(["<im_start>", "<im_end>"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    n_layers = model.config.num_hidden_layers
    print(f"[info] num_hidden_layers={n_layers}", flush=True)
    return tokenizer, model, image_processor


def tokenizer_image_token(prompt: str, tokenizer, return_tensors: str | None = None):
    """Tokenize prompt, replacing <image> with IMAGE_TOKEN_INDEX placeholder."""
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_sep(X, sep):
        return [e for pair in zip(X, [sep] * len(X)) for e in pair][:-1]

    ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        ids.append(chunks[0][0])

    for x in insert_sep(chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
        ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(ids, dtype=torch.long)
    return ids


def infer_image_positions_orig(
    input_ids: torch.Tensor,
    n_images: int = 1,
    num_image_features: int = NUM_IMAGE_FEATURES,
) -> tuple[torch.Tensor, int]:
    """Return (image_positions [n_images*576], prompt_len_mm) for original LLaVA.

    input_ids: [T] or [1, T] text-space ids with IMAGE_TOKEN_INDEX=-200.
    """
    ids = input_ids.squeeze(0).cpu()
    placeholder_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
    if placeholder_positions.numel() != n_images:
        raise ValueError(
            f"Expected {n_images} image placeholders, found {placeholder_positions.numel()}"
        )

    all_positions: list[int] = []
    offset = 0  # extra tokens added by image expansion
    for ph_pos in placeholder_positions.tolist():
        start_mm = ph_pos + offset
        all_positions.extend(range(start_mm, start_mm + num_image_features))
        offset += num_image_features - 1  # placeholder becomes num_image_features tokens

    text_len = int(ids.shape[0])
    prompt_len_mm = text_len - n_images + n_images * num_image_features
    image_positions = torch.tensor(all_positions, dtype=torch.long)
    return image_positions, prompt_len_mm


def prepare_inputs(
    tokenizer,
    image_processor,
    prompt: str,
    image: Image.Image,
    device: torch.device,
) -> dict:
    """Tokenize prompt and preprocess image for original LLaVA generate call."""
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    return {"input_ids": input_ids, "images": pixel_values}


def infer_question_positions(
    prompt_len_mm: int, image_positions: torch.Tensor
) -> torch.Tensor:
    if image_positions.numel() == 0:
        return torch.arange(prompt_len_mm, dtype=torch.long)
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


def _generate_answer(
    model,
    inputs: dict,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Generate answer tokens without retaining generation-time attentions."""
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=0,
            num_beams=1,
            use_cache=True,
        )
    # gen_out: [1, prompt_len_text + T_generated]
    prompt_len_text = int(inputs["input_ids"].shape[1])
    answer_ids = gen_out[:, prompt_len_text:].detach()
    del gen_out
    return answer_ids


def _normalize_teacher(teacher: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize each layer's image-token scores into a distribution."""
    return teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)


def _mix_teacher_signals(
    question_teacher: torch.Tensor,
    answer_teacher: torch.Tensor,
    question_weight: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix separately normalized question and answer attention distributions."""
    if not 0.0 <= question_weight <= 1.0:
        raise ValueError(f"question_weight must be in [0, 1], got {question_weight}")
    question_norm = _normalize_teacher(question_teacher, eps)
    answer_norm = _normalize_teacher(answer_teacher, eps)
    mixed = (
        question_weight * question_norm
        + (1.0 - question_weight) * answer_norm
    )
    return _normalize_teacher(mixed, eps), question_norm, answer_norm


def _collect_attention_for_question_and_answer(
    model,
    inputs: dict,
    image_indices: torch.Tensor,
    question_indices: torch.Tensor,
    prompt_len_mm: int,
    answer_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Collect question→image and answer→image attention in one full pass.

    The model is causal, so question-token rows are identical to those from a
    standalone prefill pass even though answer tokens are appended here.
    """
    T = int(answer_ids.shape[1])
    full_input_ids = torch.cat([inputs["input_ids"], answer_ids], dim=1)  # [1, prompt+T]

    with torch.no_grad():
        full_out = model(
            input_ids=full_input_ids,
            images=inputs.get("images"),
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    # Attentions: tuple of [B, H, T_mm, T_mm] per layer, T_mm = multimodal length
    # Extract attention from answer positions to image positions
    L = len(full_out.attentions)
    n_img = int(image_indices.numel())
    question_teacher = torch.zeros(L, n_img, dtype=torch.float32)
    answer_teacher = torch.zeros(L, n_img, dtype=torch.float32)

    full_len_mm = prompt_len_mm + T
    answer_positions = list(range(prompt_len_mm, full_len_mm))
    question_positions = question_indices.to(full_input_ids.device)

    if not answer_positions:
        # Model generated nothing (immediate EOS) — fall back to last prompt token
        answer_positions = [prompt_len_mm - 1]
    if question_positions.numel() == 0:
        # Defensive fallback for prompts whose image tokens end the prefill.
        question_positions = torch.tensor(
            [prompt_len_mm - 1], dtype=torch.long, device=full_input_ids.device
        )

    for l, attn in enumerate(full_out.attentions):
        # attn: [B, H, T_mm, T_mm]
        image_indices_dev = image_indices.to(attn.device)
        question_to_img = attn[0, :, question_positions, :].index_select(
            dim=-1, index=image_indices_dev
        )
        answer_to_img = attn[0, :, answer_positions, :].index_select(
            dim=-1, index=image_indices_dev
        )
        # [H, T_query, n_img] → average over heads and query tokens.
        question_teacher[l] = question_to_img.float().mean(dim=(0, 1)).cpu()
        answer_teacher[l] = answer_to_img.float().mean(dim=(0, 1)).cpu()

    del full_out
    gc.collect()
    torch.cuda.empty_cache()
    return question_teacher, answer_teacher, T


@torch.no_grad()
def collect_one(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    trajectory_m: int = 1,
    trajectory_temperature: float = 0.7,
    trajectory_top_p: float = 0.9,
    dataset: str = "",
    answers: tuple[str, ...] = (),
    answer_index: int | None = None,
    choices: tuple[str, ...] = (),
    require_correct: bool = True,
    question_weight: float = 0.5,
    eps: float = 1e-8,
) -> tuple[dict | None, str]:
    if not 0.0 <= question_weight <= 1.0:
        raise ValueError(f"question_weight must be in [0, 1], got {question_weight}")

    inputs = prepare_inputs(tokenizer, image_processor, prompt, image, device)
    image_positions, prompt_len_mm = infer_image_positions_orig(inputs["input_ids"])
    image_indices = image_positions.to(device)
    n_img = int(image_indices.numel())
    question_positions = infer_question_positions(prompt_len_mm, image_positions)

    M = max(1, trajectory_m)
    use_sampling = M > 1
    max_attempts = 1 if M == 1 else max(20, M * 10)

    question_traj_scores: list[torch.Tensor] = []
    answer_traj_scores: list[torch.Tensor] = []
    t_lengths: list[int] = []
    predictions: list[str] = []
    prediction_correctness: list[bool] = []
    last_prediction = ""
    for _ in range(max_attempts):
        answer_ids = _generate_answer(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            temperature=trajectory_temperature,
            top_p=trajectory_top_p,
        )
        prediction = tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()
        last_prediction = prediction
        prediction_correct = is_correct_prediction(
            dataset,
            prediction,
            answers,
            answer_index=answer_index,
            choices=choices,
        )
        if require_correct and not prediction_correct:
            del answer_ids
            if M == 1:
                return None, prediction
            continue

        question_score, answer_score, T = _collect_attention_for_question_and_answer(
            model=model,
            inputs=inputs,
            image_indices=image_indices,
            question_indices=question_positions,
            prompt_len_mm=prompt_len_mm,
            answer_ids=answer_ids,
        )
        del answer_ids
        question_traj_scores.append(question_score)
        answer_traj_scores.append(answer_score)
        t_lengths.append(T)
        predictions.append(prediction)
        prediction_correctness.append(prediction_correct)
        if len(answer_traj_scores) == M:
            break

    if len(answer_traj_scores) != M:
        return None, last_prediction

    question_stacked = torch.stack(question_traj_scores, dim=0)
    answer_stacked = torch.stack(answer_traj_scores, dim=0)
    teacher_question_raw = question_stacked.mean(dim=0)
    teacher_answer_raw = answer_stacked.mean(dim=0)
    answer_weight = 1.0 - question_weight
    # Mix normalized distributions so question and answer make the requested
    # contribution even when their total attention mass differs.
    teacher_norm, teacher_question_norm, teacher_answer_norm = _mix_teacher_signals(
        teacher_question_raw,
        teacher_answer_raw,
        question_weight,
        eps,
    )

    return dict(
        teacher_raw=teacher_norm.to(torch.float16),
        teacher_norm=teacher_norm.to(torch.float16),
        teacher_question_raw=teacher_question_raw.to(torch.float16),
        teacher_question_norm=teacher_question_norm.to(torch.float16),
        teacher_answer_raw=teacher_answer_raw.to(torch.float16),
        teacher_answer_norm=teacher_answer_norm.to(torch.float16),
        teacher_question_weight=float(question_weight),
        teacher_answer_weight=float(answer_weight),
        teacher_signal="question_answer_normalized_mix",
        teacher_question_source="causal_prefill_question_tokens",
        teacher_answer_source="generated_answer_tokens",
        image_token_indices=image_positions.to(torch.long),
        question_token_indices=question_positions.to(torch.long),
        prompt_len_mm=int(prompt_len_mm),
        T=int(np.mean(t_lengths)),
        n_img=int(n_img),
        trajectory_m=M,
        predictions=predictions,
        predictions_correct=prediction_correctness,
        prediction=predictions[0],
        prediction_correct=prediction_correctness[0],
    ), predictions[0]


# ── dataset loaders ────────────────────────────────────────────────────────────

_PROMPT = "USER: <image>\n{question}\nASSISTANT:"


def _fmt(question: str) -> str:
    return _PROMPT.format(question=question)


def _resolve(p: str) -> str | None:
    path = Path(p)
    if path.exists():
        return str(path)
    alt = Path(str(p).replace("/workspace/zap/data/train/", "/workspace/zap/data/train/", 1))
    return str(alt) if alt.exists() else None


def load_samples_from_json(
    samples_json: Path, max_candidates: int, seed: int
) -> list[dict]:
    records = json.loads(samples_json.read_text())
    candidates: list[dict] = []
    for rec in records:
        resolved = _resolve(rec["image_path"])
        if resolved is None:
            continue
        answer = str(rec.get("answer", "")).strip()
        if not answer:
            continue
        candidates.append(dict(
            sample_id=str(rec["sample_id"]),
            prompt=_fmt(str(rec["question"]).strip()),
            image_path=resolved,
            answers=(answer,),
            answer_index=None,
            choices=(),
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_candidates] if max_candidates > 0 else candidates


def load_scienceqa_samples(
    problems_json: Path, images_root: Path, split: str, max_candidates: int, seed: int
) -> list[dict]:
    problems = json.loads(problems_json.read_text())
    candidates: list[dict] = []
    for qid, prob in problems.items():
        if prob.get("split") != split:
            continue
        img_path = images_root / split / qid / "image.png"
        if not img_path.exists():
            continue
        question = prob.get("question", "").strip()
        choices = prob.get("choices", [])
        if choices:
            question += "\n" + "\n".join(
                f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
            )
            question += "\nAnswer with the option letter only."
        answer_index = int(prob["answer"])
        candidates.append(dict(
            sample_id=str(qid),
            prompt=_fmt(question),
            image_path=str(img_path),
            answers=(str(choices[answer_index]),),
            answer_index=answer_index,
            choices=tuple(str(choice) for choice in choices),
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_candidates] if max_candidates > 0 else candidates


def load_gqa_samples(
    questions_json: Path, images_root: Path, max_candidates: int, seed: int
) -> list[dict]:
    questions = json.loads(questions_json.read_text())
    candidates: list[dict] = []
    for qid, rec in questions.items():
        image_id = rec.get("imageId") or rec.get("image_id")
        if not image_id:
            continue
        img_path = images_root / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        question = str(rec.get("question", "")).strip()
        answer = str(rec.get("answer", "")).strip()
        if not question or not answer:
            continue
        candidates.append(dict(
            sample_id=str(qid),
            prompt=_fmt(f"Question: {question}\nAnswer the question briefly."),
            image_path=str(img_path),
            answers=(answer,),
            answer_index=None,
            choices=(),
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_candidates] if max_candidates > 0 else candidates


def load_textvqa_samples(
    data_json: Path, data_root: Path, max_candidates: int, seed: int
) -> list[dict]:
    payload = json.loads(data_json.read_text())
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[dict] = []
    for rec in records:
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("image_path", "")
        img_path = data_root / rel if rel and not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            continue
        answers = tuple(
            str(answer).strip() for answer in rec.get("answers", []) if str(answer).strip()
        )
        if not answers:
            continue
        qid = str(rec.get("question_id", rec.get("id", f"tvqa_{len(candidates):06d}")))
        candidates.append(dict(
            sample_id=qid,
            prompt=_fmt(f"Question: {question}\nAnswer the question briefly."),
            image_path=str(img_path),
            answers=answers,
            answer_index=None,
            choices=(),
        ))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_candidates] if max_candidates > 0 else candidates


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(WORKSPACE_ROOT / "models/llava-v1.5-7b"))
    p.add_argument(
        "--dataset",
        required=True,
        choices=["scienceqa", "gqa", "textvqa", "llava_instruct"],
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Target number of teacher samples to save.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum shuffled candidates to inspect; 0 means all available candidates.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(WORKSPACE_ROOT / "data/train/teacher/llava15"))
    p.add_argument("--problems-json", default=str(WORKSPACE_ROOT / "data/train/scienceqa/problems.json"))
    p.add_argument("--images-root", default=str(WORKSPACE_ROOT / "data/train/scienceqa/images"))
    p.add_argument("--split", default="train")
    p.add_argument("--gqa-questions-json", default=str(WORKSPACE_ROOT / "data/train/gqa/train_balanced_questions.json"))
    p.add_argument("--gqa-images-root", default=str(WORKSPACE_ROOT / "data/train/gqa/images"))
    p.add_argument("--llava-instruct-samples-json", default=str(WORKSPACE_ROOT / "data/train/llava_instruct_sample/samples.json"))
    p.add_argument("--textvqa-json", default=str(WORKSPACE_ROOT / "data/train/textvqa/train/data.json"))
    p.add_argument("--textvqa-data-root", default=str(WORKSPACE_ROOT / "data/train"))
    p.add_argument("--trajectory-m", type=int, default=1)
    p.add_argument("--trajectory-temperature", type=float, default=0.7)
    p.add_argument("--trajectory-top-p", type=float, default=0.9)
    p.add_argument(
        "--require-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save teacher records only when the base model prediction is correct.",
    )
    p.add_argument(
        "--question-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the normalized question/prefill attention in the teacher; "
            "answer attention receives 1 - this value."
        ),
    )
    args = p.parse_args()
    if not 0.0 <= args.question_weight <= 1.0:
        p.error("--question-weight must be in [0, 1]")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer, model, image_processor = load_orig_llava_model(args.model, device)

    if args.dataset == "scienceqa":
        samples = load_scienceqa_samples(
            Path(args.problems_json), Path(args.images_root), args.split, args.max_candidates, args.seed
        )
    elif args.dataset == "gqa":
        samples = load_gqa_samples(
            Path(args.gqa_questions_json), Path(args.gqa_images_root), args.max_candidates, args.seed
        )
    elif args.dataset == "textvqa":
        samples = load_textvqa_samples(
            Path(args.textvqa_json), Path(args.textvqa_data_root), args.max_candidates, args.seed
        )
    else:  # llava_instruct
        samples = load_samples_from_json(
            Path(args.llava_instruct_samples_json), args.max_candidates, args.seed
        )
    print(
        f"[info] dataset={args.dataset} candidates={len(samples)} "
        f"target={args.n_samples} require_correct={args.require_correct} "
        f"question_weight={args.question_weight:.3f} "
        f"answer_weight={1.0 - args.question_weight:.3f}",
        flush=True,
    )

    existing_paths = list(out_dir.glob("*.pt"))
    existing = len(existing_paths)
    existing_correct = 0
    for existing_path in existing_paths:
        existing_rec = torch.load(existing_path, weights_only=False, map_location="cpu")
        existing_correct += int(bool(existing_rec.get("prediction_correct", False)))
    existing_incorrect = existing - existing_correct
    saved = existing
    saved_correct = existing_correct
    saved_incorrect = existing_incorrect
    newly_saved = 0
    evaluated = 0
    rejected_incorrect = 0
    incorrect_examples: list[dict] = []
    skipped: list[tuple[str, str]] = []
    t_list: list[int] = []
    t0 = time.time()

    for sample in samples:
        if saved >= args.n_samples:
            break
        sid = sample["sample_id"]
        prompt = sample["prompt"]
        img_path = sample["image_path"]
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid))[:128]
        out_path = out_dir / f"{safe_sid}.pt"
        if out_path.exists():
            continue
        evaluated += 1
        try:
            image = Image.open(img_path).convert("RGB")
            rec, prediction = collect_one(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
                dataset=args.dataset,
                answers=sample["answers"],
                answer_index=sample["answer_index"],
                choices=sample["choices"],
                require_correct=args.require_correct,
                question_weight=args.question_weight,
            )
            if rec is None:
                rejected_incorrect += 1
                if len(incorrect_examples) < 100:
                    incorrect_examples.append(dict(
                        sample_id=sid,
                        prediction=prediction,
                        answers=list(sample["answers"]),
                        answer_index=sample["answer_index"],
                    ))
                if evaluated % 25 == 0:
                    print(
                        f"[progress] evaluated={evaluated}/{len(samples)} "
                        f"saved={saved}/{args.n_samples} "
                        f"incorrect={rejected_incorrect} skipped={len(skipped)}",
                        flush=True,
                    )
                continue
            rec.update(
                sample_id=sid,
                dataset=args.dataset,
                model="llava-v1.5-7b-orig",
                prompt_text=prompt,
                image_path=img_path,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                ground_truth_answers=list(sample["answers"]),
                ground_truth_answer_index=sample["answer_index"],
                ground_truth_choices=list(sample["choices"]),
                require_correct=args.require_correct,
            )
            torch.save(rec, out_path)
            saved += 1
            if rec["prediction_correct"]:
                saved_correct += 1
            else:
                saved_incorrect += 1
            newly_saved += 1
            t_list.append(rec["T"])
            if newly_saved == 1:
                print(
                    f"[sanity] sid={sid} L={rec['teacher_raw'].shape[0]} "
                    f"N_I={rec['teacher_raw'].shape[1]} T={rec['T']} "
                    f"prompt_len_mm={rec['prompt_len_mm']} "
                    f"|img|={rec['image_token_indices'].numel()} "
                    f"|q|={rec['question_token_indices'].numel()}",
                    flush=True,
                )
        except (RuntimeError, ValueError, IOError) as e:
            skipped.append((sid, repr(e)))
            print(f"[skip] {sid}: {e}", flush=True)
            continue

        if evaluated % 25 == 0 or saved == args.n_samples:
            elapsed = time.time() - t0
            print(
                f"[progress] evaluated={evaluated}/{len(samples)} "
                f"| rate={evaluated/max(elapsed, 1e-6):.2f}/s "
                f"| elapsed={elapsed:.1f}s | T_mean={np.mean(t_list):.2f} "
                f"| saved={saved}/{args.n_samples} correct={saved_correct} "
                f"incorrect={saved_incorrect} rejected={rejected_incorrect} "
                f"skipped={len(skipped)}",
                flush=True,
            )
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"[done] dataset={args.dataset} saved={saved}/{args.n_samples} "
        f"correct={saved_correct} incorrect={saved_incorrect} "
        f"candidates={len(samples)} evaluated={evaluated} "
        f"rejected={rejected_incorrect} skipped={len(skipped)} elapsed={elapsed:.1f}s "
        f"T_mean={np.mean(t_list) if t_list else 0:.2f}",
        flush=True,
    )

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(dict(
        dataset=args.dataset,
        n_requested=args.n_samples,
        n_saved=saved,
        n_existing=existing,
        n_existing_correct=existing_correct,
        n_existing_incorrect=existing_incorrect,
        n_newly_saved=newly_saved,
        n_saved_correct=saved_correct,
        n_saved_incorrect=saved_incorrect,
        n_candidates=len(samples),
        n_evaluated=evaluated,
        n_rejected_incorrect=rejected_incorrect,
        incorrect_examples=incorrect_examples,
        n_skipped=len(skipped),
        skipped=skipped,
        t_mean=float(np.mean(t_list)) if t_list else 0.0,
        seed=args.seed,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        require_correct=args.require_correct,
        question_weight=args.question_weight,
        answer_weight=1.0 - args.question_weight,
        elapsed_seconds=elapsed,
    ), indent=2))
    print(f"[save] {summary_path}", flush=True)
    if saved != args.n_samples:
        print(
            f"[error] Exhausted candidates before reaching target: "
            f"{saved}/{args.n_samples}",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
