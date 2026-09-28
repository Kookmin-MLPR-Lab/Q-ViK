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
sys.path.insert(0, str(PROJECT_ROOT))

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


def _collect_attention_two_pass(
    model,
    inputs: dict,
    image_indices: torch.Tensor,
    prompt_len_mm: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Two-pass teacher extraction: generate answer, then forward pass with attention.

    This avoids generate(output_attentions=True) which doesn't work in newer
    transformers versions for the original LLaVA model.
    """
    # Pass 1: generate answer tokens (no attention tracking)
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
    T = int(gen_out.shape[1]) - prompt_len_text

    # Pass 2: forward on full sequence (prompt + answer) with output_attentions=True
    # Build full input: original input_ids + generated answer tokens
    answer_ids = gen_out[0, prompt_len_text:].unsqueeze(0)  # [1, T]
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
    teacher = torch.zeros(L, n_img, dtype=torch.float32)

    full_len_mm = prompt_len_mm + T
    answer_positions = list(range(prompt_len_mm, full_len_mm))

    if not answer_positions:
        # Model generated nothing (immediate EOS) — fall back to last prompt token
        answer_positions = [prompt_len_mm - 1]

    for l, attn in enumerate(full_out.attentions):
        # attn: [B, H, T_mm, T_mm]
        # Average attention from answer tokens to image positions
        attn_to_img = attn[0, :, answer_positions, :].index_select(dim=-1, index=image_indices.to(attn.device))
        # [H, T_answer, n_img] → average over heads then answer tokens
        teacher[l] = attn_to_img.float().mean(dim=0).mean(dim=0).cpu()

    del gen_out, full_out
    gc.collect()
    torch.cuda.empty_cache()
    return teacher, T


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
    eps: float = 1e-8,
) -> dict:
    inputs = prepare_inputs(tokenizer, image_processor, prompt, image, device)
    image_positions, prompt_len_mm = infer_image_positions_orig(inputs["input_ids"])
    image_indices = image_positions.to(device)
    n_img = int(image_indices.numel())
    question_positions = infer_question_positions(prompt_len_mm, image_positions)

    M = max(1, trajectory_m)
    use_sampling = M > 1

    traj_scores: list[torch.Tensor] = []
    t_lengths: list[int] = []
    for _ in range(M):
        score, T = _collect_attention_two_pass(
            model=model,
            inputs=inputs,
            image_indices=image_indices,
            prompt_len_mm=prompt_len_mm,
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            temperature=trajectory_temperature,
            top_p=trajectory_top_p,
            eps=eps,
        )
        traj_scores.append(score)
        t_lengths.append(T)

    stacked = torch.stack(traj_scores, dim=0)
    teacher = stacked.mean(dim=0)
    teacher_norm = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)

    return dict(
        teacher_raw=teacher.to(torch.float16),
        teacher_norm=teacher_norm.to(torch.float16),
        image_token_indices=image_positions.to(torch.long),
        question_token_indices=question_positions.to(torch.long),
        prompt_len_mm=int(prompt_len_mm),
        T=int(np.mean(t_lengths)),
        n_img=int(n_img),
        trajectory_m=M,
    )


# ── dataset loaders ────────────────────────────────────────────────────────────

_PROMPT = "USER: <image>\n{question}\nASSISTANT:"


def _fmt(question: str) -> str:
    return _PROMPT.format(question=question)


def _resolve(p: str) -> str | None:
    path = Path(p)
    return str(path) if path.exists() else None


def load_samples_from_json(
    samples_json: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    records = json.loads(samples_json.read_text())
    candidates: list[tuple[str, str, str]] = []
    for rec in records:
        resolved = _resolve(rec["image_path"])
        if resolved is None:
            continue
        candidates.append((str(rec["sample_id"]), _fmt(str(rec["question"]).strip()), resolved))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_scienceqa_samples(
    problems_json: Path, images_root: Path, split: str, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    problems = json.loads(problems_json.read_text())
    candidates: list[tuple[str, str, str]] = []
    for qid, prob in problems.items():
        if not qid.startswith(f"{split}_"):
            continue
        img_path = images_root / split / qid / "image.png"
        if not img_path.exists():
            continue
        question = prob.get("question", "").strip()
        choices = prob.get("choices", [])
        if choices:
            question += "\n" + " ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        candidates.append((qid, _fmt(question), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_gqa_samples(
    questions_json: Path, images_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    questions = json.loads(questions_json.read_text())
    candidates: list[tuple[str, str, str]] = []
    for qid, rec in questions.items():
        image_id = rec.get("imageId") or rec.get("image_id")
        if not image_id:
            continue
        img_path = images_root / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        candidates.append((str(qid), _fmt(f"Question: {question}\nAnswer the question briefly."), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_textvqa_samples(
    data_json: Path, data_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, str]]:
    payload = json.loads(data_json.read_text())
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[tuple[str, str, str]] = []
    for rec in records:
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("image_path", "")
        img_path = data_root / rel if rel and not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            continue
        qid = str(rec.get("question_id", rec.get("id", f"tvqa_{len(candidates):06d}")))
        candidates.append((qid, _fmt(f"Question: {question}\nAnswer the question briefly."), str(img_path)))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(PROJECT_ROOT / "model/llava-v1.5-7b"))
    p.add_argument(
        "--dataset",
        required=True,
        choices=["scienceqa", "gqa", "textvqa", "llava_instruct"],
    )
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "data/train/teacher/llava15"))
    p.add_argument("--problems-json", default=str(PROJECT_ROOT / "data/train/scienceqa/problems.json"))
    p.add_argument("--images-root", default=str(PROJECT_ROOT / "data/train/scienceqa/images"))
    p.add_argument("--split", default="train")
    p.add_argument("--gqa-questions-json", default=str(PROJECT_ROOT / "data/train/gqa/train_balanced_questions_600.json"))
    p.add_argument("--gqa-images-root", default=str(PROJECT_ROOT / "data/train/gqa/images"))
    p.add_argument("--llava-instruct-samples-json", default=str(PROJECT_ROOT / "data/train/llava_instruct_sample/samples.json"))
    p.add_argument("--textvqa-json", default=str(PROJECT_ROOT / "data/train/textvqa/train/data.json"))
    p.add_argument("--textvqa-data-root", default=str(PROJECT_ROOT / "data/train"))
    p.add_argument("--trajectory-m", type=int, default=1)
    p.add_argument("--trajectory-temperature", type=float, default=0.7)
    p.add_argument("--trajectory-top-p", type=float, default=0.9)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer, model, image_processor = load_orig_llava_model(args.model, device)

    if args.dataset == "scienceqa":
        samples = load_scienceqa_samples(
            Path(args.problems_json), Path(args.images_root), args.split, args.n_samples, args.seed
        )
    elif args.dataset == "gqa":
        samples = load_gqa_samples(
            Path(args.gqa_questions_json), Path(args.gqa_images_root), args.n_samples, args.seed
        )
    elif args.dataset == "textvqa":
        samples = load_textvqa_samples(
            Path(args.textvqa_json), Path(args.textvqa_data_root), args.n_samples, args.seed
        )
    else:  # llava_instruct
        samples = load_samples_from_json(
            Path(args.llava_instruct_samples_json), args.n_samples, args.seed
        )
    print(f"[info] dataset={args.dataset} loaded {len(samples)} samples", flush=True)

    saved = 0
    skipped: list[tuple[str, str]] = []
    t_list: list[int] = []
    t0 = time.time()

    for idx, (sid, prompt, img_path) in enumerate(samples):
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid))[:128]
        out_path = out_dir / f"{safe_sid}.pt"
        if out_path.exists():
            saved += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            rec = collect_one(
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
            )
            rec.update(
                sample_id=sid,
                dataset=args.dataset,
                model="llava-v1.5-7b-orig",
                prompt_text=prompt,
                image_path=img_path,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            torch.save(rec, out_path)
            saved += 1
            t_list.append(rec["T"])
            if idx == 0:
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

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[progress] {idx+1}/{len(samples)} | rate={(idx+1)/max(elapsed, 1e-6):.2f}/s "
                f"| elapsed={elapsed:.1f}s | T_mean={np.mean(t_list):.2f} "
                f"| saved={saved} skipped={len(skipped)}",
                flush=True,
            )
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"[done] dataset={args.dataset} saved={saved}/{len(samples)} "
        f"skipped={len(skipped)} elapsed={elapsed:.1f}s "
        f"T_mean={np.mean(t_list) if t_list else 0:.2f}",
        flush=True,
    )

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(dict(
        dataset=args.dataset,
        n_requested=args.n_samples,
        n_saved=saved,
        n_skipped=len(skipped),
        skipped=skipped,
        t_mean=float(np.mean(t_list)) if t_list else 0.0,
        seed=args.seed,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        elapsed_seconds=elapsed,
    ), indent=2))
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
