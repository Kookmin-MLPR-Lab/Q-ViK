#!/usr/bin/env python3
"""Stage 2: Train OneVision visual-utility student (1D conv branch).

Uses LLaVA-OneVision original repo format (llava-onevision-qwen2-7b-ov).
Teacher shards are loaded from `teacher_root` and inputs are prepared via
the llava original tokenizer + image processor pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importing the vendored package patches transformers.modeling_outputs with the
# custom siglip output classes (see qvik/llava_onevision/__init__.py).
import qvik.llava_onevision  # noqa: F401

from kvpress.presses.visual_utility_student_onevision import (
    VisualUtilityStudentOneVision,
    pairwise_ranking_loss,
)


CONV_TEMPLATE = "qwen_1_5"


def _build_prompt(tokenizer, question: str, num_images: int = 1) -> str:
    from qvik.llava_onevision.constants import DEFAULT_IMAGE_TOKEN
    from qvik.llava_onevision.conversation import conv_templates

    conv = conv_templates[CONV_TEMPLATE].copy()
    image_prefix = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
    conv.append_message(conv.roles[0], f"{image_prefix}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def list_teacher_files(teacher_root: Path, datasets: list[str], per_ds_limit: int | None) -> list[Path]:
    files: list[Path] = []
    for ds in datasets:
        ds_dir = teacher_root / ds
        if not ds_dir.exists():
            continue
        ds_files = sorted(ds_dir.glob("*.pt"))
        if per_ds_limit is not None:
            ds_files = ds_files[:per_ds_limit]
        files.extend(ds_files)
    return files


class TeacherCacheDataset(Dataset):
    """Preload teacher .pt files and tokenize inputs for llava original format."""

    def __init__(
        self,
        files: list[Path],
        tokenizer=None,
        image_processor=None,
        model_config=None,
        verbose: bool = True,
    ) -> None:
        from qvik.llava_onevision.constants import IMAGE_TOKEN_INDEX
        from qvik.llava_onevision.mm_utils import process_images, tokenizer_image_token

        self.files = list(files)
        self.image_processor = image_processor
        self.model_config = model_config
        self.records: list[dict] = []
        for i, p in enumerate(self.files):
            rec = torch.load(p, weights_only=False, map_location="cpu")
            if tokenizer is not None and image_processor is not None:
                question = rec.get("question_text") or rec.get("prompt_text") or ""
                if isinstance(rec["image_path"], (list, tuple)):
                    images = [Image.open(pp).convert("RGB") for pp in rec["image_path"]]
                    n_images = len(images)
                else:
                    images = [Image.open(rec["image_path"]).convert("RGB")]
                    n_images = 1
                prompt = _build_prompt(tokenizer, question, num_images=n_images)
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                image_tensor = process_images(images, image_processor, model_config)
                rec["_input_ids"] = input_ids
                rec["_image_tensor"] = image_tensor
                rec["_image_sizes"] = [img.size for img in images]
            self.records.append(rec)
            if verbose and (i + 1) % 100 == 0:
                print(f"[preload] {i+1}/{len(self.files)}", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_single(batch: list[dict]) -> dict:
    assert len(batch) == 1, "this trainer uses batch size 1"
    return batch[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-root", default="/workspace/zap/data/train/teacher/llava_onevision")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["textvqa", "gqa", "scienceqa"],
    )
    p.add_argument(
        "--per-ds-limit",
        type=int,
        default=600,
        help="Cap samples per dataset (0 or negative to use all).",
    )
    p.add_argument(
        "--llava-path",
        default="/workspace/zap/model/llava-onevision-qwen2-7b-ov",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lambda-rank", type=float, default=0.1)
    p.add_argument("--rank-margin", type=float, default=0.05)
    p.add_argument("--rank-top-ratio", type=float, default=0.2)
    p.add_argument("--rank-bottom-ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--student-variant", choices=["full", "mlp_only", "cnn_only"], default="full")
    p.add_argument("--conv-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--num-conv-blocks", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=7)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w")
    (out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[load] OneVision={args.llava_path} device={device}", flush=True)
    from qvik.llava_onevision.mm_utils import get_model_name_from_path
    from qvik.llava_onevision.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(args.llava_path)
    tokenizer, lvlm, image_processor, _ = load_pretrained_model(
        args.llava_path, None, model_name,
        device_map=args.device,
        attn_implementation="sdpa",
        multimodal=True,
    )
    lvlm = lvlm.to(torch.bfloat16).eval()
    # builder.py hardcodes vision_tower to "cuda" (cuda:0); move everything to target device
    vision_tower = lvlm.get_model().get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(device=device, dtype=torch.bfloat16)
    for p_ in lvlm.parameters():
        p_.requires_grad_(False)

    per_ds_limit = args.per_ds_limit if args.per_ds_limit and args.per_ds_limit > 0 else None
    all_files = list_teacher_files(Path(args.teacher_root), args.datasets, per_ds_limit)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_total = len(all_files)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    print(
        f"[data] total={n_total} train={n_train} val={n_val} "
        f"per_ds_limit={per_ds_limit} datasets={args.datasets} (preloading...)",
        flush=True,
    )
    train_ds = TeacherCacheDataset(all_files[:n_train], tokenizer=tokenizer, image_processor=image_processor, model_config=lvlm.config)
    val_ds = TeacherCacheDataset(all_files[n_train:], tokenizer=tokenizer, image_processor=image_processor, model_config=lvlm.config)
    print(f"[data] preloaded train={len(train_ds)} val={len(val_ds)}", flush=True)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_single, generator=loader_gen
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    student = VisualUtilityStudentOneVision(
        variant=args.student_variant,
        conv_dim=args.conv_dim,
        proj_dim=args.proj_dim,
        mlp_dim=args.mlp_dim,
        num_conv_blocks=args.num_conv_blocks,
        kernel_size=args.kernel_size,
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(
        f"[student] variant={student.variant} layers={student.layer_indices} "
        f"params={n_params:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_forward(rec: dict, train: bool) -> tuple[float, float, float]:
        input_ids = rec["_input_ids"].unsqueeze(0).to(device)
        image_tensor = rec["_image_tensor"]
        if isinstance(image_tensor, list):
            image_tensor = [t.to(device, dtype=torch.bfloat16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.bfloat16)
        image_sizes = rec["_image_sizes"]
        attention_mask = input_ids.ne(
            tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        ).to(device)

        with torch.no_grad():
            _, _, attention_mask, _, inputs_embeds, _ = lvlm.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None,
                image_tensor, ["image"], image_sizes,
            )
            out = lvlm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # index 0 = embeddings, index l+1 = output of layer l
        H_all = out.hidden_states

        image_idx = rec["image_token_indices"].to(device, dtype=torch.long)
        q_idx = rec["question_token_indices"].to(device, dtype=torch.long)
        teacher_norm = rec["teacher_norm"].to(device, dtype=torch.float32)  # [L, N_I]

        # Sanity: prompt expansion at training time must match the cached image-token count.
        N_I_cached = int(image_idx.numel())
        N_prefill = int(H_all[0].shape[1])
        if int(image_idx.max().item()) >= N_prefill:
            raise ValueError(
                f"image_idx max {int(image_idx.max().item())} exceeds prefill length {N_prefill}; "
                "OneVision processor expansion drifted from collection time."
            )

        total_mse = 0.0
        total_rank = 0.0
        total_loss_val = 0.0
        n_layers = len(student.layer_indices)

        if train:
            optimizer.zero_grad(set_to_none=True)

        n_layers_used = 0
        for li in student.layer_indices:
            t = teacher_norm[li].unsqueeze(0)  # [1, N_I]
            if t.shape[-1] != N_I_cached:
                raise ValueError(
                    f"teacher row width {t.shape[-1]} != cached N_I {N_I_cached}"
                )
            # OneVision fp16 attention overflows in some deep layers (mostly L=27);
            # skip those layer/sample combos so a corrupt teacher row does not
            # contaminate the optimizer step.
            if not torch.isfinite(t).all():
                continue
            # Cast each layer's hidden state separately so the bf16→fp32 buffer
            # for the previous layer can be freed once its backward completes.
            H_l = H_all[li + 1].to(torch.float32)
            s_pred = student.forward_layer(li, H_l, image_idx, q_idx)  # [B, N_I]
            pred_norm = F.softmax(s_pred, dim=-1)
            loss_mse = F.mse_loss(pred_norm, t)
            loss_rank = pairwise_ranking_loss(
                pred_norm,
                t,
                margin=args.rank_margin,
                top_ratio=args.rank_top_ratio,
                bottom_ratio=args.rank_bottom_ratio,
            )
            loss_l = loss_mse + args.lambda_rank * loss_rank

            if torch.isnan(loss_l) or torch.isinf(loss_l):
                raise RuntimeError(
                    f"non-finite loss at layer {li}: mse={loss_mse.item()} rank={loss_rank.item()} N_I={N_I_cached}"
                )

            if train:
                # Per-layer backward keeps only one layer's activations live at a time.
                (loss_l / n_layers).backward()

            total_mse += float(loss_mse.detach().item())
            total_rank += float(loss_rank.detach().item())
            total_loss_val += float(loss_l.detach().item())
            n_layers_used += 1

            del H_l, s_pred, pred_norm, loss_mse, loss_rank, loss_l

        if train:
            # Skip the optimizer step entirely if any param grad is non-finite —
            # otherwise a single bad sample can poison the whole student.
            bad_grad = False
            for p_ in student.parameters():
                if p_.grad is not None and not torch.isfinite(p_.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"non-finite gradient detected; skipping step (N_I={N_I_cached})"
                )
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        denom = max(1, n_layers_used)
        return total_loss_val, total_mse / denom, total_rank / denom

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        student.train()
        train_loss = train_mse = train_rank = 0.0
        n_seen = 0
        for step, rec in enumerate(train_loader):
            try:
                loss, mse, rank = run_forward(rec, train=True)
            except (RuntimeError, ValueError, IOError) as e:
                print(
                    f"[skip] {rec.get('sample_id', '?')}: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
                continue
            train_loss += loss
            train_mse += mse
            train_rank += rank
            n_seen += 1
            if (step + 1) % args.log_every == 0:
                print(
                    f"[epoch {epoch} step {step+1}/{len(train_loader)}] "
                    f"loss={train_loss/n_seen:.5f} mse={train_mse/n_seen:.5f} "
                    f"rank={train_rank/n_seen:.5f} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
        train_loss /= max(1, n_seen)
        train_mse /= max(1, n_seen)
        train_rank /= max(1, n_seen)

        student.eval()
        val_loss = val_mse = val_rank = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for rec in val_loader:
                try:
                    loss, mse, rank = run_forward(rec, train=False)
                except (RuntimeError, ValueError, IOError) as e:
                    print(
                        f"[skip val] {rec.get('sample_id', '?')}: {e}\n{traceback.format_exc()}",
                        flush=True,
                    )
                    continue
                val_loss += loss
                val_mse += mse
                val_rank += rank
                n_val_seen += 1
        if n_val_seen == 0:
            n_val_seen = 1
        val_loss /= n_val_seen
        val_mse /= n_val_seen
        val_rank /= n_val_seen

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={train_loss:.5f} train_mse={train_mse:.5f} "
            f"train_rank={train_rank:.5f} | val_loss={val_loss:.5f} val_mse={val_mse:.5f} "
            f"val_rank={val_rank:.5f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        log_f.write(
            json.dumps(
                dict(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_mse=train_mse,
                    train_rank=train_rank,
                    val_loss=val_loss,
                    val_mse=val_mse,
                    val_rank=val_rank,
                    elapsed=elapsed,
                )
            )
            + "\n"
        )
        log_f.flush()

        if val_loss < best_val:
            best_val = val_loss
            student.save_pretrained(out_dir)
            print(f"[ckpt] saved best (val_loss={val_loss:.5f}) to {out_dir}", flush=True)

    log_f.close()
    print(f"[done] best_val={best_val:.5f} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
