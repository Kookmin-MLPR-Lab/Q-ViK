
"""Train the LLaVA-1.5 visual-utility student with original LLaVA hidden states."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _patch_transformers_tokenizers_check() -> None:
    import importlib.metadata as importlib_metadata

    original_version = importlib_metadata.version

    def version(package_name: str) -> str:
        if package_name == "tokenizers":
            return "0.20.3"
        if package_name == "huggingface-hub":
            return "0.26.5"
        return original_version(package_name)

    importlib_metadata.version = version


def _patch_torch_load_legacy_bin_mmap() -> None:
    original_torch_load = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        retry_kwargs = dict(kwargs)
        while True:
            try:
                return original_torch_load(*args, **retry_kwargs)
            except RuntimeError as exc:
                if retry_kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs.pop("mmap", None)
                    continue
                raise
            except pickle.UnpicklingError:
                if retry_kwargs.get("weights_only") is True:
                    retry_kwargs["weights_only"] = False
                    retry_kwargs.pop("mmap", None)
                    continue
                raise

    torch.load = load


_patch_transformers_tokenizers_check()
_patch_torch_load_legacy_bin_mmap()

from qvik.llava15.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from qvik.llava15.mm_utils import tokenizer_image_token  # noqa: E402
from qvik.llava15.model.builder import load_pretrained_model  # noqa: E402

from qvik.student.visual_utility_student_llava15 import (  # noqa: E402
    VisualUtilityStudent,
    pairwise_ranking_loss,
)


def list_teacher_files(
    teacher_root: Path,
    datasets: list[str],
    n_per_dataset: int | None,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    files: list[Path] = []
    for dataset in datasets:
        ds_files = sorted((teacher_root / dataset).glob("*.pt"))
        if n_per_dataset is not None and len(ds_files) > n_per_dataset:
            ds_files = sorted(rng.sample(ds_files, n_per_dataset))
        files.extend(ds_files)
    return files


class TeacherDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = torch.load(self.files[idx], weights_only=False, map_location="cpu")
        rec["_record_path"] = str(self.files[idx])
        return rec


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("This trainer expects batch size 1.")
    return batch[0]


def resolve_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    candidates = [path]
    path_str = str(path)
    if path_str.startswith("/workspace/zap/"):
        rel_path = Path(path_str.removeprefix("/workspace/zap/"))
        candidates.append(REPO_ROOT / rel_path)
        rel_parts = rel_path.parts
        if len(rel_parts) >= 2 and rel_parts[0] == "data" and rel_parts[1] in {
            "gqa",
            "scienceqa",
            "textvqa",
        }:
            candidates.append(REPO_ROOT / "data/train" / Path(*rel_parts[1:]))
        if len(rel_parts) >= 5 and rel_parts[:4] == ("data", "gqa", "train", "images"):
            candidates.append(REPO_ROOT / "data/gqa/images" / rel_path.name)
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve image path {image_path}; tried: {tried}")


def load_original_llava(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    overwrite_config = None
    if args.vision_tower_path:
        overwrite_config = {"mm_vision_tower": args.vision_tower_path}
    print(
        f"[load] model={args.llava_path} model_name={args.model_name} "
        f"device_map={args.device_map} attn=sdpa vision={args.vision_tower_path}",
        flush=True,
    )
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=args.llava_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device_map,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"[load-ok] class={model.__class__.__name__} layers={model.config.num_hidden_layers} "
        f"dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    return tokenizer, model, image_processor


def build_inputs(
    rec: dict[str, Any],
    *,
    tokenizer: Any,
    image_processor: Any,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    image_path = resolve_image_path(rec["image_path"])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        # v1 LLaVA-1.5 student training used the raw CLIP image processor.
        # `process_images` honors image_aspect_ratio="pad" and changes the
        # hidden-state distribution relative to the reference checkpoint.
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=next(model.parameters()).dtype)
    input_ids = tokenizer_image_token(
        rec["prompt_text"],
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    return {
        "input_ids": input_ids,
        "images": image_tensor,
        "image_sizes": [image_size],
        "modalities": ["image"],
    }


@torch.no_grad()
def compute_hidden_states(
    rec: dict[str, Any],
    *,
    tokenizer: Any,
    image_processor: Any,
    model: Any,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = build_inputs(
        rec,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        device=device,
    )
    out = model(
        inputs["input_ids"],
        images=inputs["images"],
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    image_idx = rec["image_token_indices"].to(device=device, dtype=torch.long)
    question_idx = rec["question_token_indices"].to(device=device, dtype=torch.long)
    teacher_norm = rec["teacher_norm"].to(device=device, dtype=torch.float32)
    return out.hidden_states, image_idx, question_idx, teacher_norm


def train_one_sample(
    rec: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudent,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    hidden_states, image_idx, question_idx, teacher_norm = compute_hidden_states(
        rec,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    loss_sum = mse_sum = rank_sum = 0.0
    for layer_idx in student.layer_indices:
        # Same index in training, validation and inference: the student's
        # `hidden_state_offset` (1 = output of layer l, as in the zap v1 trainer).
        h_l = student.layer_input(hidden_states, layer_idx).to(torch.float32)
        scores = student.forward_layer(layer_idx, h_l, image_idx, question_idx)
        pred_norm = F.softmax(scores, dim=-1)
        target = teacher_norm[layer_idx].unsqueeze(0)
        loss_mse = F.mse_loss(pred_norm, target)
        loss_rank = pairwise_ranking_loss(
            pred_norm,
            target,
            margin=args.rank_margin,
            top_ratio=args.rank_top_ratio,
            bottom_ratio=args.rank_bottom_ratio,
        )
        # The v1 LLaVA-1.5 trainer backpropagated the sum over layer losses.
        # Dividing by num_layers here shrinks the effective LR by 32x and
        # reproduces the regressed train_mse curve.
        layer_loss = loss_mse + args.lambda_rank * loss_rank
        layer_loss.backward()
        loss_sum += float((loss_mse + args.lambda_rank * loss_rank).detach().item())
        mse_sum += float(loss_mse.detach().item())
        rank_sum += float(loss_rank.detach().item())

    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.max_grad_norm)
    optimizer.step()

    n_layers = len(student.layer_indices)
    del hidden_states
    torch.cuda.empty_cache()
    return loss_sum, mse_sum / n_layers, rank_sum / n_layers


@torch.no_grad()
def eval_one_sample(
    rec: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    student: VisualUtilityStudent,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float]:
    hidden_states, image_idx, question_idx, teacher_norm = compute_hidden_states(
        rec,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        device=device,
    )
    loss_sum = mse_sum = rank_sum = 0.0
    for layer_idx in student.layer_indices:
        h_l = student.layer_input(hidden_states, layer_idx).to(torch.float32)
        scores = student.forward_layer(layer_idx, h_l, image_idx, question_idx)
        pred_norm = F.softmax(scores, dim=-1)
        target = teacher_norm[layer_idx].unsqueeze(0)
        loss_mse = F.mse_loss(pred_norm, target)
        loss_rank = pairwise_ranking_loss(
            pred_norm,
            target,
            margin=args.rank_margin,
            top_ratio=args.rank_top_ratio,
            bottom_ratio=args.rank_bottom_ratio,
        )
        loss_sum += float((loss_mse + args.lambda_rank * loss_rank).item())
        mse_sum += float(loss_mse.item())
        rank_sum += float(loss_rank.item())
    n_layers = len(student.layer_indices)
    del hidden_states
    torch.cuda.empty_cache()
    return loss_sum, mse_sum / n_layers, rank_sum / n_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", default=str(REPO_ROOT / "data/train/teacher/llava15"))
    parser.add_argument("--datasets", nargs="+", default=["textvqa", "scienceqa", "gqa"])
    parser.add_argument("--llava-path", default=str(REPO_ROOT / "model/llava-v1.5-7b"))
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--vision-tower-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-top-ratio", type=float, default=0.2)
    parser.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-dataset", type=int, default=600)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing checkpoint in --output-dir.")
    parser.add_argument("--conv-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--num-conv-blocks", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--grid-h", type=int, default=24)
    parser.add_argument("--grid-w", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    if (out_dir / "pytorch_model.bin").exists() and not args.overwrite:
        raise SystemExit(
            f"{out_dir} already holds a checkpoint; pass --overwrite to replace it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "train_log.jsonl").open("w")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer, model, image_processor = load_original_llava(args)

    all_files = list_teacher_files(
        Path(args.teacher_root),
        args.datasets,
        n_per_dataset=args.n_per_dataset,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    if n_total == 0:
        raise RuntimeError(f"No teacher files found under {args.teacher_root}")
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    print(f"[data] total={n_total} train={len(train_files)} val={len(val_files)}", flush=True)

    train_loader = DataLoader(
        TeacherDataset(train_files),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TeacherDataset(val_files),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
    )

    student = VisualUtilityStudent(
        conv_dim=args.conv_dim,
        proj_dim=args.proj_dim,
        mlp_dim=args.mlp_dim,
        num_conv_blocks=args.num_conv_blocks,
        kernel_size=args.kernel_size,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(
        f"[student] layers={student.layer_indices} "
        f"params={n_params:,}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        student.train()
        train_loss = train_mse = train_rank = 0.0
        train_seen = 0
        for step, rec in enumerate(train_loader, start=1):
            try:
                loss, mse, rank = train_one_sample(
                    rec,
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    student=student,
                    optimizer=optimizer,
                    args=args,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[skip train] {rec.get('dataset')}:{rec.get('sample_id')} "
                    f"{exc}\n{traceback.format_exc()}",
                    flush=True,
                )
                continue
            train_seen += 1
            train_loss += loss
            train_mse += mse
            train_rank += rank
            if step % args.log_every == 0:
                print(
                    f"[epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)}] "
                    f"loss={train_loss/train_seen:.6f} mse={train_mse/train_seen:.6f} "
                    f"rank={train_rank/train_seen:.6f} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

        train_loss /= max(1, train_seen)
        train_mse /= max(1, train_seen)
        train_rank /= max(1, train_seen)

        student.eval()
        val_loss = val_mse = val_rank = 0.0
        val_seen = 0
        for rec in val_loader:
            try:
                loss, mse, rank = eval_one_sample(
                    rec,
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    student=student,
                    args=args,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[skip val] {rec.get('dataset')}:{rec.get('sample_id')} "
                    f"{exc}\n{traceback.format_exc()}",
                    flush=True,
                )
                continue
            val_seen += 1
            val_loss += loss
            val_mse += mse
            val_rank += rank
        val_loss /= max(1, val_seen)
        val_mse /= max(1, val_seen)
        val_rank /= max(1, val_seen)

        elapsed = time.time() - t0
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_rank": train_rank,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_rank": val_rank,
            "train_seen": train_seen,
            "val_seen": val_seen,
            "elapsed": elapsed,
        }
        print(
            f"[epoch {epoch + 1}] train_loss={train_loss:.6f} train_mse={train_mse:.6f} "
            f"train_rank={train_rank:.6f} | val_loss={val_loss:.6f} "
            f"val_mse={val_mse:.6f} val_rank={val_rank:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

        torch.save(
            {
                "epoch": epoch + 1,
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": record,
            },
            out_dir / "last_checkpoint.pt",
        )
        if val_loss < best_val:
            best_val = val_loss
            student.save_pretrained(out_dir)
            print(f"[ckpt] saved best val_loss={best_val:.6f} to {out_dir}", flush=True)

    log_f.close()
    print(f"[done] best_val={best_val:.6f} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
