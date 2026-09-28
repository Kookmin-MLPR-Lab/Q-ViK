"""Visual-utility student variants for image-token eviction.

Per-layer student takes prefill hidden state H_l (B, N, D) plus image and
question position indices, returns predicted score (B, N_I) per image token.
Branches:
- Image CNN over reshaped H_img grid (1x1 + ConvNeXt blocks)
- Question pooled from H_q, projected and broadcast
- Raw H_img projection
The default ``full`` variant fuses all three branches. Ablations can select
``mlp_only`` (raw image-token + question) or ``cnn_only`` (CNN context +
question) while keeping the same teacher labels and loss code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn


_ALL_LAYERS: tuple[int, ...] = tuple(range(32))
StudentVariant = Literal["full", "mlp_only", "cnn_only"]


def _fusion_input_dim(variant: StudentVariant, proj_dim: int) -> int:
    if variant == "full":
        return proj_dim * 5
    if variant in ("mlp_only", "cnn_only"):
        return proj_dim * 3
    raise ValueError(f"Unsupported student variant: {variant}")


class ConvNeXtStyleBlock(nn.Module):
    """[B, C, H, W] → [B, C, H, W] residual block.

    DW{k}x{k} → GroupNorm(1, C) → 1x1 expand to 4C → GELU → 1x1 to C → +residual
    """

    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class VisualUtilityStudentLayer(nn.Module):
    """Per-layer student. Takes H_l, image_idx, question_idx → score [B, N_I]."""

    def __init__(
        self,
        hidden_dim: int = 4096,
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
            self.conv_1x1_proj = nn.Conv2d(hidden_dim, conv_dim, kernel_size=1)
            self.conv_blocks = nn.Sequential(
                *[ConvNeXtStyleBlock(conv_dim, kernel_size=kernel_size) for _ in range(num_conv_blocks)]
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
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        H_l: [B, N_prefill, D]
        image_indices: [N_I]   long. N_I must be a multiple of grid_h*grid_w —
                                each block of grid_h*grid_w tokens is one image.
        question_indices: [N_Q] long (>=0)
        return: [B, N_I]  scalar score per image token (logits, not softmaxed)
        """
        B = H_l.shape[0]
        D = H_l.shape[-1]
        N_I = image_indices.numel()
        n_per_img = grid_h * grid_w
        if self.variant in ("full", "cnn_only") and N_I % n_per_img != 0:
            raise ValueError(
                f"N_I={N_I} is not a multiple of grid_h*grid_w={n_per_img}; "
                f"non-uniform images are not supported."
            )
        n_images = N_I // n_per_img if n_per_img > 0 else 0

        H_img = H_l.index_select(dim=1, index=image_indices)  # [B, N_I, D]
        if question_indices.numel() == 0:
            H_q = H_l.new_zeros((B, 1, D))
        else:
            H_q = H_l.index_select(dim=1, index=question_indices)  # [B, N_Q, D]

        if self.variant in ("full", "cnn_only"):
            # --- Image CNN branch (per-image grid; multi-image batched along B*k)
            F_img = (
                H_img.reshape(B, n_images, grid_h, grid_w, D)
                .reshape(B * n_images, grid_h, grid_w, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # [B*k, D, H, W]
            X_img = self.conv_1x1_proj(F_img)  # [B*k, C, H, W]
            C_img = self.conv_blocks(X_img)
            C_flat = C_img.flatten(2).transpose(1, 2)  # [B*k, n_per_img, C]
            C_flat = C_flat.reshape(B, N_I, -1)  # [B, N_I, C]
            C_proj = self.W_c(C_flat)  # [B, N_I, d]
        else:
            C_proj = None

        # --- Question branch (shared across all images in the sample)
        q = H_q.mean(dim=1)  # [B, D]
        q_proj = self.W_q(q)  # [B, d]
        Q = q_proj.unsqueeze(1).expand(-1, N_I, -1)  # [B, N_I, d]

        if self.variant in ("full", "mlp_only"):
            # --- Raw image-token projection
            H_proj = self.W_h(H_img)  # [B, N_I, d]
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
        score = self.mlp_head(Z).squeeze(-1)  # [B, N_I]
        return score


class VisualUtilityStudent(nn.Module):
    """Holds one VisualUtilityStudentLayer per layer in the chosen scope."""

    def __init__(
        self,
        hidden_dim: int = 4096,
        conv_dim: int = 256,
        proj_dim: int = 256,
        mlp_dim: int = 512,
        num_conv_blocks: int = 2,
        kernel_size: int = 7,
        grid_h: int = 24,
        grid_w: int = 24,
        variant: StudentVariant = "full",
    ) -> None:
        super().__init__()
        self.layer_indices = _ALL_LAYERS
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.variant = variant
        self.config = dict(
            layer_indices=list(self.layer_indices),
            hidden_dim=hidden_dim,
            conv_dim=conv_dim,
            proj_dim=proj_dim,
            mlp_dim=mlp_dim,
            num_conv_blocks=num_conv_blocks,
            kernel_size=kernel_size,
            grid_h=grid_h,
            grid_w=grid_w,
            variant=variant,
        )
        self.layers = nn.ModuleDict(
            {
                str(li): VisualUtilityStudentLayer(
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
        return self.layers[str(layer_idx)](
            H_l, image_indices, question_indices, self.grid_h, self.grid_w
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(self.config, indent=2))
        torch.save(self.state_dict(), out / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, map_location: str = "cpu") -> "VisualUtilityStudent":
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        cfg.pop("layer_indices", None)
        cfg.pop("scope", None)  # backwards compat: old checkpoints saved scope="A"
        cfg.setdefault("variant", "full")
        model = cls(**cfg)
        sd = torch.load(d / "pytorch_model.bin", map_location=map_location, weights_only=True)
        model.load_state_dict(sd)
        return model


def pairwise_ranking_loss(
    pred: torch.Tensor,  # [B, N]
    target: torch.Tensor,  # [B, N]
    margin: float = 0.05,
    top_ratio: float = 0.2,
    bottom_ratio: float = 0.4,
) -> torch.Tensor:
    """Hinge ranking loss between top-p and bottom-q teacher positions.

    For each sample, pick top_ratio of positions (by target) as 'good' and
    bottom_ratio as 'bad'; penalize when pred[good] - pred[bad] < margin.
    """
    B, N = pred.shape
    k_top = max(1, int(N * top_ratio))
    k_bot = max(1, int(N * bottom_ratio))
    losses = []
    for b in range(B):
        t = target[b]
        p = pred[b]
        top_idx = torch.topk(t, k=k_top, largest=True).indices
        bot_idx = torch.topk(t, k=k_bot, largest=False).indices
        diff = p[top_idx][:, None] - p[bot_idx][None, :]  # [k_top, k_bot]
        losses.append(torch.clamp(margin - diff, min=0.0).mean())
    return torch.stack(losses).mean()
