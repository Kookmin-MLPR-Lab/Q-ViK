#!/usr/bin/env python3
"""Stage 1 teacher cache for LLaVA-OneVision (original repo format).

Uses llava-onevision-qwen2-7b-ov (original repo) with llava.model.builder.
Runs prefill once with `output_attentions=False`, then loops decode steps
manually with `output_attentions=True` to avoid OOM on the full attention tensor.

Per-sample output: ``<output_root>/<dataset>/<sample_id>.pt`` with schema:
teacher_raw, teacher_norm, image_token_indices, question_token_indices,
prompt_len_mm, T, n_img, trajectory_m.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importing the vendored package patches transformers.modeling_outputs with the
# custom siglip output classes (see qvik/llava_onevision/__init__.py).
import qvik.llava_onevision  # noqa: F401

CONV_TEMPLATE = "qwen_1_5"


# ── dataset loaders ────────────────────────────────────────────────────────
# Return (sample_id, question_text, image_paths_list) — the OneVision
# Qwen2 chat template is applied in `collect_one()`.

_OPTION_LETTERS = "ABCDEFGHIJ"


def build_scienceqa_prompt(problem: dict) -> str:
    question = str(problem["question"]).strip()
    choices = [str(c).strip() for c in problem["choices"]]
    hint = str(problem.get("hint", "") or "").strip()
    lines = ["USER: <image>"]
    if hint:
        lines.append(f"Context: {hint}")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    for i, c in enumerate(choices):
        lines.append(f"{_OPTION_LETTERS[i]}. {c}")
    lines.append("Select the best answer based on the image and text.")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


_LLAVA_INSTRUCT_SUBSETS = [
    "complex_reasoning_77k",
    "conversation_58k",
    "detail_23k",
]


def _load_llava_instruct_subset(
    subset: str, hf_cache_dir: str | None
) -> list[tuple[str, str, list]]:
    from datasets import load_dataset

    ds = load_dataset(
        "liuhaotian/LLaVA-Instruct-150K", name=subset, split="train",
        cache_dir=hf_cache_dir,
    )
    candidates: list[tuple[str, str, list]] = []
    for i, rec in enumerate(ds):
        sid = str(rec.get("id", f"{subset}_{i:06d}"))
        convs = rec.get("conversations", [])
        if not convs:
            continue
        human_turn = next((c for c in convs if c.get("from") == "human"), None)
        if human_turn is None:
            continue
        question = human_turn.get("value", "").strip()
        question = re.sub(r"<image>\n?", "", question).strip()
        if not question:
            continue
        img = rec.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img).convert("RGB")
            except Exception:
                continue
        else:
            img = img.convert("RGB")
        candidates.append((sid, question, [img]))
    return candidates


def load_llava_instruct_questions_hf(
    n_samples: int, seed: int, hf_cache_dir: str | None = None
) -> list[tuple[str, str, list]]:
    """Load LLaVA-Instruct-150K with stratified sampling across 3 sub-categories.

    Samples are drawn equally from complex_reasoning_77k, conversation_58k,
    and detail_23k to avoid the natural skew (51% / 39% / 15%).
    Falls back to loading the default split if named configs are unavailable.
    """
    rng = random.Random(seed)

    # Try stratified loading first
    try:
        per_subset = max(1, n_samples // len(_LLAVA_INSTRUCT_SUBSETS))
        all_candidates: list[tuple[str, str, list]] = []
        for subset in _LLAVA_INSTRUCT_SUBSETS:
            pool = _load_llava_instruct_subset(subset, hf_cache_dir)
            rng.shuffle(pool)
            all_candidates.extend(pool[:per_subset])
        # Fill remainder from any subset to hit n_samples exactly
        if len(all_candidates) < n_samples:
            extra_pool: list[tuple[str, str, list]] = []
            for subset in _LLAVA_INSTRUCT_SUBSETS:
                extra_pool.extend(_load_llava_instruct_subset(subset, hf_cache_dir))
            existing_ids = {s[0] for s in all_candidates}
            extra_pool = [s for s in extra_pool if s[0] not in existing_ids]
            rng.shuffle(extra_pool)
            all_candidates.extend(extra_pool[: n_samples - len(all_candidates)])
        rng.shuffle(all_candidates)
        return all_candidates[:n_samples]
    except Exception as e:
        print(f"[warn] stratified load failed ({e}), falling back to default split", flush=True)

    # Fallback: single split (original distribution)
    from datasets import load_dataset

    ds = load_dataset(
        "liuhaotian/LLaVA-Instruct-150K", split="train", cache_dir=hf_cache_dir
    )
    candidates: list[tuple[str, str, list]] = []
    for i, rec in enumerate(ds):
        sid = str(rec.get("id", f"llava_instruct_{i:06d}"))
        convs = rec.get("conversations", [])
        if not convs:
            continue
        human_turn = next((c for c in convs if c.get("from") == "human"), None)
        if human_turn is None:
            continue
        question = human_turn.get("value", "").strip()
        question = re.sub(r"<image>\n?", "", question).strip()
        if not question:
            continue
        img = rec.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img).convert("RGB")
            except Exception:
                continue
        else:
            img = img.convert("RGB")
        candidates.append((sid, question, [img]))
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_mmvet_questions_hf(
    n_samples: int, seed: int, hf_cache_dir: str | None = None
) -> list[tuple[str, str, list]]:
    """Load MMVet from HuggingFace."""
    from datasets import load_dataset

    # lmms-lab/MMVet has a default split; try 'test' or default
    try:
        ds = load_dataset("lmms-lab/MMVet", split="test", cache_dir=hf_cache_dir)
    except Exception:
        ds = load_dataset("lmms-lab/MMVet", cache_dir=hf_cache_dir)
        # pick the first available split
        if hasattr(ds, "keys"):
            ds = ds[next(iter(ds.keys()))]
    candidates: list[tuple[str, str, list]] = []
    for i, rec in enumerate(ds):
        sid = str(rec.get("imagename", rec.get("id", f"mmvet_{i:04d}")))
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        img = rec.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img).convert("RGB")
            except Exception:
                continue
        else:
            img = img.convert("RGB")
        candidates.append((sid, question, [img]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_scienceqa_questions(
    problems_json: Path, images_root: Path, split: str, n_samples: int, seed: int
) -> list[tuple[str, str, list[str]]]:
    with problems_json.open() as f:
        problems = json.load(f)
    candidates: list[tuple[str, str, list[str]]] = []
    prefix = f"{split}_"
    for qid, prob in problems.items():
        if not qid.startswith(prefix):
            continue
        img_path = images_root / split / qid / "image.png"
        if not img_path.exists():
            continue
        prompt_full = build_scienceqa_prompt(prob)
        # Strip the LLaVA-1.5 scaffolding to recover bare question text.
        question = (
            prompt_full.replace("USER: <image>\n", "").replace("\nASSISTANT:", "").strip()
        )
        candidates.append((qid, question, [str(img_path)]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_gqa_questions(
    questions_json: Path, images_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, list[str]]]:
    with questions_json.open() as f:
        questions = json.load(f)
    candidates: list[tuple[str, str, list[str]]] = []
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
        question = f"{question}\nAnswer the question using a single word or phrase."
        candidates.append((str(qid), question, [str(img_path)]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_st_vqa_questions(
    data_json: Path, images_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, list[str]]]:
    """ST-VQA train_task_3 loader.

    Schema (CVC UAB ST-VQA): a JSON file with key 'data' containing records
    that include 'question', 'file_path' (or 'file_name'), 'answers'.
    Image directory layout matches `file_path` relative to `images_root`.
    """
    with data_json.open() as f:
        payload = json.load(f)
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[tuple[str, str, list[str]]] = []
    for i, rec in enumerate(records):
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("file_path") or rec.get("file_name")
        if not rel:
            continue
        img_path = images_root / rel
        if not img_path.exists():
            # Try a flat layout (file_name only) as fallback.
            alt = images_root / Path(rel).name
            if alt.exists():
                img_path = alt
            else:
                continue
        qid = str(rec.get("question_id", rec.get("id", f"st_vqa_{i:06d}")))
        question = f"{question}\nAnswer the question using a single word or phrase."
        candidates.append((qid, question, [str(img_path)]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_textvqa_questions(
    data_json: Path, data_root: Path, n_samples: int, seed: int
) -> list[tuple[str, str, list[str]]]:
    with data_json.open() as f:
        payload = json.load(f)
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    candidates: list[tuple[str, str, list[str]]] = []
    for rec in records:
        question = str(rec.get("question", "")).strip()
        if not question:
            continue
        rel = rec.get("image_path", "")
        img_path = data_root / rel if rel and not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            continue
        qid = str(rec.get("question_id", rec.get("id", f"tvqa_{len(candidates):06d}")))
        candidates.append((qid, f"{question}\nAnswer the question using a single word or phrase.", [str(img_path)]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


def load_samples_from_json(
    samples_json: Path, n_samples: int, seed: int
) -> list[tuple[str, str, list[str]]]:
    """Generic loader for pre-sampled datasets saved as samples.json.

    JSON format: list of {sample_id, image_path, question, answer}.
    """
    with samples_json.open() as f:
        records = json.load(f)
    candidates: list[tuple[str, str, list[str]]] = []
    for rec in records:
        sid = str(rec["sample_id"])
        question = str(rec["question"]).strip()
        question = f"{question}\nAnswer the question using a single word or phrase."
        img_path = rec["image_path"]
        if not Path(img_path).exists():
            continue
        candidates.append((sid, question, [img_path]))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n_samples]


# ── prompt construction ────────────────────────────────────────────────────


def build_onevision_prompt(tokenizer, question: str, num_images: int) -> str:
    from qvik.llava_onevision.constants import DEFAULT_IMAGE_TOKEN
    from qvik.llava_onevision.conversation import conv_templates

    conv = conv_templates[CONV_TEMPLATE].copy()
    image_prefix = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
    conv.append_message(conv.roles[0], f"{image_prefix}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ── manual prefill + decode loop ───────────────────────────────────────────


@torch.no_grad()
def _generate_with_per_step_attentions(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_indices: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_ids: set[int],
) -> tuple[torch.Tensor, int]:
    """Prefill (no attentions) then hook-based decode loop.

    Hooks capture attention per-layer and immediately compress to [N_I] scalars,
    so the full [H, 1, T] attention tensor is never held in memory.
    Returns (teacher [L, N_I] fp32 cpu, T).
    """
    prefill_out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past_kv = prefill_out.past_key_values
    next_logits = prefill_out.logits[:, -1, :]

    if do_sample:
        probs = _top_p_sample_probs(next_logits / max(temperature, 1e-6), top_p)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = next_logits.argmax(dim=-1, keepdim=True)

    # Collect decoder self-attention modules
    attn_layers = []
    for layer in model.model.layers:
        attn_layers.append(layer.self_attn)

    n_layers = len(attn_layers)
    n_img = int(image_indices.numel())
    image_indices_dev = image_indices.to(model.device)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int, orig_fwd):
        def hooked(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = orig_fwd(*args, **kwargs)
            w = out[1]
            if w is not None:
                captured[li] = (
                    w[0, :, -1, :].index_select(-1, image_indices_dev)
                    .float().mean(0).detach().cpu()
                )
            return (out[0], None) + out[2:]
        return hooked

    orig_forwards = {}
    for li, attn in enumerate(attn_layers):
        orig_forwards[li] = attn.forward
        attn.forward = make_hook(li, orig_forwards[li])

    teacher = torch.zeros(n_layers, n_img, dtype=torch.float32)
    T = 0
    attn_mask = attention_mask

    try:
        for _ in range(max_new_tokens):
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=attn_mask.dtype, device=attn_mask.device)],
                dim=1,
            )
            captured.clear()
            step_out = model(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = step_out.past_key_values
            for li in range(n_layers):
                if li in captured:
                    teacher[li] += captured[li]
            T += 1

            next_logits = step_out.logits[:, -1, :]
            if do_sample:
                probs = _top_p_sample_probs(next_logits / max(temperature, 1e-6), top_p)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            if int(next_token.item()) in eos_token_ids:
                break
    finally:
        for li, attn in enumerate(attn_layers):
            attn.forward = orig_forwards[li]

    if T == 0:
        raise RuntimeError("Decode loop produced zero steps")
    teacher /= float(T)

    del past_kv, prefill_out
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return teacher, T


def _top_p_sample_probs(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum > top_p
    mask[..., 0] = False  # always keep top-1
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return out


def infer_question_positions(prompt_len_mm: int, image_positions: torch.Tensor) -> torch.Tensor:
    if image_positions.numel() == 0:
        return torch.arange(prompt_len_mm, dtype=torch.long)
    last_img = int(image_positions.max().item())
    if last_img + 1 >= prompt_len_mm:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(last_img + 1, prompt_len_mm, dtype=torch.long)


@torch.no_grad()
def collect_one(
    model,
    tokenizer,
    image_processor,
    image,
    question: str,
    max_new_tokens: int,
    device: torch.device,
    trajectory_m: int = 1,
    trajectory_temperature: float = 0.7,
    trajectory_top_p: float = 0.9,
    eps: float = 1e-8,
) -> dict:
    from qvik.llava_onevision.constants import IMAGE_TOKEN_INDEX
    from qvik.llava_onevision.mm_utils import process_images, tokenizer_image_token

    images = image if isinstance(image, (list, tuple)) else [image]
    n_images = len(images)
    prompt = build_onevision_prompt(tokenizer, question, num_images=n_images)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    attention_mask = input_ids.ne(
        tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    ).to(device)

    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [t.to(device, dtype=torch.bfloat16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

    _, _, attention_mask, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, None, None,
        image_tensor, ["image"], [img.size for img in images],
    )
    prompt_len_mm = int(inputs_embeds.shape[1])

    # Derive image positions: IMAGE_TOKEN_INDEX placeholder_pos → expanded range
    input_ids_1d = input_ids[0]
    placeholder_pos = int((input_ids_1d == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item())
    n_text_tokens = int(input_ids_1d.shape[0]) - n_images  # each placeholder expands
    n_img = prompt_len_mm - n_text_tokens
    image_positions = torch.arange(placeholder_pos, placeholder_pos + n_img, dtype=torch.long)

    n_img = int(image_positions.numel())
    question_positions = infer_question_positions(prompt_len_mm, image_positions)

    # Use ALL eos tokens from generation_config (Qwen2 uses both 151645 im_end and
    # 151643 endoftext; manual loop must catch either, like HF model.generate does).
    _eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if _eos is None:
        _eos = tokenizer.eos_token_id
    if isinstance(_eos, int):
        eos_token_ids = {_eos}
    elif isinstance(_eos, (list, tuple)):
        eos_token_ids = {int(t) for t in _eos}
    else:
        eos_token_ids = {151645, 151643}
    M = max(1, trajectory_m)
    use_sampling = M > 1

    traj_scores: list[torch.Tensor] = []
    t_lengths: list[int] = []
    for _ in range(M):
        score, T = _generate_with_per_step_attentions(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_indices=image_positions,
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            temperature=trajectory_temperature,
            top_p=trajectory_top_p,
            eos_token_ids=eos_token_ids,
        )
        traj_scores.append(score)
        t_lengths.append(T)

    stacked = torch.stack(traj_scores, dim=0)
    teacher = stacked.mean(dim=0)
    teacher_norm = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)

    result = dict(
        teacher_raw=teacher.to(torch.float16),
        teacher_norm=teacher_norm.to(torch.float16),
        image_token_indices=image_positions.to(torch.long),
        question_token_indices=question_positions.to(torch.long),
        prompt_len_mm=int(prompt_len_mm),
        T=int(np.mean(t_lengths)),
        n_img=int(n_img),
        trajectory_m=M,
    )
    if M > 1:
        result["teacher_var"] = stacked.var(dim=0).to(torch.float16)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/zap/model/llava-onevision-qwen2-7b-ov")
    p.add_argument("--dataset", required=True, choices=["scienceqa", "gqa", "textvqa", "st_vqa", "chartqa", "docvqa", "infovqa", "llava_instruct", "mmvet"])
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default="/workspace/zap/data/train/teacher/llava_onevision")
    p.add_argument("--problems-json", default="/workspace/zap/data/train/scienceqa/problems.json")
    p.add_argument("--images-root", default="/workspace/zap/data/train/scienceqa/images")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--gqa-questions-json",
        default="/workspace/zap/data/train/gqa/val_balanced_questions.json",
    )
    p.add_argument("--gqa-images-root", default="/workspace/zap/data/train/gqa/images")
    p.add_argument("--textvqa-json", default="/workspace/zap/data/train/textvqa/train/data.json")
    p.add_argument("--textvqa-data-root", default="/workspace/zap/data/train")
    p.add_argument(
        "--st-vqa-json",
        default="/workspace/zap/data/train/st_vqa/train_task_3.json",
    )
    p.add_argument(
        "--st-vqa-images-root",
        default="/workspace/zap/data/train/st_vqa",
    )
    p.add_argument(
        "--chartqa-samples-json",
        default="/workspace/zap/data/train/chartqa_train_sample/samples.json",
    )
    p.add_argument(
        "--docvqa-samples-json",
        default="/workspace/zap/data/train/docvqa_train_sample/samples.json",
    )
    p.add_argument(
        "--infovqa-samples-json",
        default="/workspace/zap/data/train/infovqa_train_sample/samples.json",
    )
    p.add_argument("--max-image-size", type=int, default=0,
                   help="If >0, resize images so the longest side <= this value before processing.")
    p.add_argument("--hf-cache-dir", default=None,
                   help="HuggingFace datasets cache directory for llava_instruct / mmvet.")
    p.add_argument("--llava-instruct-samples-json",
                   default="/workspace/zap/data/train/llava_instruct_sample/samples.json")
    p.add_argument("--mmvet-samples-json",
                   default="/workspace/zap/data/train/mmvet_sample/samples.json")
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

    print(f"[load] {args.model} dtype=fp16 attn=eager device={device}", flush=True)
    from qvik.llava_onevision.mm_utils import get_model_name_from_path
    from qvik.llava_onevision.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(args.model)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, model_name,
        device_map=device,
        attn_implementation="sdpa",
        multimodal=True,
    )
    model = model.to(torch.bfloat16).eval()
    print(
        f"[info] num_hidden_layers={model.config.num_hidden_layers} "
        f"hidden_size={model.config.hidden_size}",
        flush=True,
    )

    if args.dataset == "scienceqa":
        samples = load_scienceqa_questions(
            problems_json=Path(args.problems_json),
            images_root=Path(args.images_root),
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "textvqa":
        samples = load_textvqa_questions(
            data_json=Path(args.textvqa_json),
            data_root=Path(args.textvqa_data_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "gqa":
        samples = load_gqa_questions(
            questions_json=Path(args.gqa_questions_json),
            images_root=Path(args.gqa_images_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "st_vqa":
        samples = load_st_vqa_questions(
            data_json=Path(args.st_vqa_json),
            images_root=Path(args.st_vqa_images_root),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "chartqa":
        samples = load_samples_from_json(
            samples_json=Path(args.chartqa_samples_json),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "docvqa":
        samples = load_samples_from_json(
            samples_json=Path(args.docvqa_samples_json),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "infovqa":
        samples = load_samples_from_json(
            samples_json=Path(args.infovqa_samples_json),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    elif args.dataset == "llava_instruct":
        samples = load_samples_from_json(
            samples_json=Path(args.llava_instruct_samples_json),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    else:  # mmvet
        samples = load_samples_from_json(
            samples_json=Path(args.mmvet_samples_json),
            n_samples=args.n_samples,
            seed=args.seed,
        )
    print(f"[info] dataset={args.dataset} loaded {len(samples)} samples", flush=True)

    saved = 0
    skipped: list[tuple[str, str]] = []
    t_list: list[int] = []
    t0 = time.time()

    for idx, (sid, question, img_paths) in enumerate(samples):
        safe_sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sid))[:128]
        out_path = out_dir / f"{safe_sid}.pt"
        if out_path.exists():
            saved += 1
            continue
        try:
            def _resolve_image(src) -> Image.Image:
                if isinstance(src, str):
                    img = Image.open(src).convert("RGB")
                else:
                    img = src.convert("RGB")
                if args.max_image_size > 0:
                    w, h = img.size
                    long_side = max(w, h)
                    if long_side > args.max_image_size:
                        scale = args.max_image_size / long_side
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                return img

            def _stored(src) -> str:
                return src if isinstance(src, str) else f"hf:{args.dataset}:{sid}"

            if isinstance(img_paths, (list, tuple)) and len(img_paths) == 1:
                image = _resolve_image(img_paths[0])
                stored_path = _stored(img_paths[0])
            else:
                image = [_resolve_image(p) for p in img_paths]
                stored_path = [_stored(p) for p in img_paths]
            rec = collect_one(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=image,
                question=question,
                max_new_tokens=args.max_new_tokens,
                device=device,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
            )
            rec.update(
                sample_id=sid,
                dataset=args.dataset,
                model="llava-onevision-qwen2-7b-ov",
                question_text=question,
                image_path=stored_path,
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
                f"[progress] {idx+1}/{len(samples)} | rate={(idx+1)/max(elapsed,1e-6):.2f}/s "
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
    summary_path.write_text(
        json.dumps(
            dict(
                dataset=args.dataset,
                n_requested=args.n_samples,
                n_saved=saved,
                n_skipped=len(skipped),
                skipped=skipped,
                t_mean=float(np.mean(t_list)) if t_list else 0.0,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                elapsed_seconds=elapsed,
                trajectory_m=args.trajectory_m,
                trajectory_temperature=args.trajectory_temperature,
                trajectory_top_p=args.trajectory_top_p,
            ),
            indent=2,
        )
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
