"""Visual-utility student variants for LLaVA-OneVision (anyres image tokens).

OneVision uses AnyRes image tokenization, so the per-sample image-token count
N_I varies and is NOT a multiple of any fixed (grid_h, grid_w). The 2D CNN
branch from `visual_utility_student.py` cannot be reused; here we replace it
with a 1D ConvNeXt-style stack along the image-token sequence.

Per-layer student (`VisualUtilityStudentLayerOneVision`):
- 1D conv branch over the variable-length image-token sequence
- Question pooled from H_q, projected, broadcast across N_I
- Raw H_img projection
The default ``full`` variant fuses all three branches. Ablations can select
``mlp_only`` (raw image-token + question) or ``cnn_only`` (1D CNN context +
question) while keeping the same teacher labels and loss code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


# Qwen2-7B language model has 28 decoder layers in OneVision-7B.
_ALL_LAYERS_ONEVISION: tuple[int, ...] = tuple(range(28))
StudentVariant = Literal["full", "mlp_only", "cnn_only"]


def _fusion_input_dim(variant: StudentVariant, proj_dim: int) -> int:
    if variant == "full":
        return proj_dim * 5
    if variant in ("mlp_only", "cnn_only"):
        return proj_dim * 3
    raise ValueError(f"Unsupported student variant: {variant}")


class ConvNeXt1DBlock(nn.Module):
    """[B, C, L] → [B, C, L] residual block.

    DWConv1d(k) → GroupNorm(1, C) → 1x1 expand → GELU → 1x1 contract → +residual.
    """

    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class VisualUtilityStudentLayerOneVision(nn.Module):
    """Per-layer student. Takes H_l + index sets → score [B, N_I]."""

    def __init__(
        self,
        hidden_dim: int = 3584,
        conv_dim: int = 256,
        proj_dim: int = 256,
        mlp_dim: int = 512,
        num_conv_blocks: int = 2,
        kernel_size: int = 7,
        variant: StudentVariant = "full",
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.proj_dim = proj_dim
        self.variant = variant

        if variant in ("full", "cnn_only"):
            self.conv_1x1_proj = nn.Conv1d(hidden_dim, conv_dim, kernel_size=1)
            self.conv_blocks = nn.Sequential(
                *[ConvNeXt1DBlock(conv_dim, kernel_size=kernel_size) for _ in range(num_conv_blocks)]
            )
            if conv_dim != proj_dim:
                self.W_c = nn.Linear(conv_dim, proj_dim)
            else:
                self.W_c = nn.Identity()
        if variant in ("full", "mlp_only"):
            self.W_h = nn.Linear(hidden_dim, proj_dim)
        self.W_q = nn.Linear(hidden_dim, proj_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(_fusion_input_dim(variant, proj_dim), mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        H_l: torch.Tensor,
        image_indices: torch.Tensor,
        question_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        H_l: [B, N_prefill, D]
        image_indices: [N_I]   long
        question_indices: [N_Q] long (>= 0)
        return: [B, N_I]   per-image-token logit (not softmaxed)
        """
        B = H_l.shape[0]
        D = H_l.shape[-1]
        N_I = int(image_indices.numel())

        H_img = H_l.index_select(dim=1, index=image_indices)  # [B, N_I, D]
        if question_indices.numel() == 0:
            q = H_l.new_zeros((B, D))
        else:
            q_sum = H_l.new_zeros((B, D))
            q_count = int(question_indices.numel())
            for start in range(0, q_count, 256):
                idx = question_indices[start : start + 256]
                q_sum = q_sum + H_l.index_select(dim=1, index=idx).sum(dim=1)
            q = q_sum / max(1, q_count)

        if self.variant in ("full", "cnn_only"):
            # --- 1D conv branch over image-token sequence
            F_img = H_img.permute(0, 2, 1).contiguous()  # [B, D, N_I]
            X_img = self.conv_1x1_proj(F_img)            # [B, C, N_I]
            C_img = self.conv_blocks(X_img)              # [B, C, N_I]
            C_flat = C_img.permute(0, 2, 1).contiguous() # [B, N_I, C]
            C_proj = self.W_c(C_flat)                    # [B, N_I, d]
        else:
            C_proj = None

        # --- Question branch (pooled, broadcast)
        q_proj = self.W_q(q)                         # [B, d]
        Q = q_proj.unsqueeze(1).expand(-1, N_I, -1)  # [B, N_I, d]

        if self.variant in ("full", "mlp_only"):
            # --- Raw image-token projection
            H_proj = self.W_h(H_img)                 # [B, N_I, d]
        else:
            H_proj = None

        # --- Fusion
        if self.variant == "full":
            assert H_proj is not None and C_proj is not None
            Z = torch.cat([H_proj, C_proj, Q, H_proj * Q, C_proj * Q], dim=-1)  # [B, N_I, 5d]
        elif self.variant == "mlp_only":
            assert H_proj is not None
            Z = torch.cat([H_proj, Q, H_proj * Q], dim=-1)  # [B, N_I, 3d]
        elif self.variant == "cnn_only":
            assert C_proj is not None
            Z = torch.cat([C_proj, Q, C_proj * Q], dim=-1)  # [B, N_I, 3d]
        else:
            raise ValueError(f"Unsupported student variant: {self.variant}")
        score = self.mlp_head(Z).squeeze(-1)         # [B, N_I]
        return score


class VisualUtilityStudentOneVision(nn.Module):
    """One `VisualUtilityStudentLayerOneVision` per layer in the chosen scope."""

    def __init__(
        self,
        hidden_dim: int = 3584,
        conv_dim: int = 256,
        proj_dim: int = 256,
        mlp_dim: int = 512,
        num_conv_blocks: int = 2,
        kernel_size: int = 7,
        variant: StudentVariant = "full",
    ) -> None:
        super().__init__()
        self.layer_indices = _ALL_LAYERS_ONEVISION
        self.variant = variant
        self.config = dict(
            layer_indices=list(self.layer_indices),
            hidden_dim=hidden_dim,
            conv_dim=conv_dim,
            proj_dim=proj_dim,
            mlp_dim=mlp_dim,
            num_conv_blocks=num_conv_blocks,
            kernel_size=kernel_size,
            variant=variant,
        )
        self.layers = nn.ModuleDict(
            {
                str(li): VisualUtilityStudentLayerOneVision(
                    hidden_dim=hidden_dim,
                    conv_dim=conv_dim,
                    proj_dim=proj_dim,
                    mlp_dim=mlp_dim,
                    num_conv_blocks=num_conv_blocks,
                    kernel_size=kernel_size,
                    variant=variant,
                )
                for li in self.layer_indices
            }
        )

    def forward_layer(
        self,
        layer_idx: int,
        H_l: torch.Tensor,
        image_indices: torch.Tensor,
        question_indices: torch.Tensor,
    ) -> torch.Tensor:
        if str(layer_idx) not in self.layers:
            raise KeyError(f"layer {layer_idx} not in layer_indices={self.layer_indices}")
        return self.layers[str(layer_idx)](H_l, image_indices, question_indices)

    def save_pretrained(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(self.config, indent=2))
        torch.save(self.state_dict(), out / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, map_location: str = "cpu"
    ) -> "VisualUtilityStudentOneVision":
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        cfg.pop("layer_indices", None)
        cfg.pop("scope", None)  # backwards compat: old checkpoints saved scope="A"
        cfg.setdefault("variant", "full")
        model = cls(**cfg)
        state = torch.load(d / "pytorch_model.bin", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.05,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.4,
) -> torch.Tensor:
    """Margin-ranking loss between pred-top and pred-bottom selected by target.

    Mirrors `kvpress.presses.visual_utility_student.pairwise_ranking_loss`.
    """
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must match and be 2-D"
        )
    B, N = pred.shape
    k_top = max(1, int(round(N * top_ratio)))
    k_bot = max(1, int(round(N * bottom_ratio)))
    losses = []
    for b in range(B):
        t = target[b]
        p = pred[b]
        top_idx = torch.topk(t, k_top, largest=True).indices
        bot_idx = torch.topk(t, k_bot, largest=False).indices
        p_top = p.index_select(0, top_idx).unsqueeze(1)  # [k_top, 1]
        p_bot = p.index_select(0, bot_idx).unsqueeze(0)  # [1, k_bot]
        diff = p_bot - p_top + margin
        losses.append(torch.clamp(diff, min=0.0).mean())
    return torch.stack(losses).mean()
