# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Literal, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

try:
    from transformers import QuantizedCache
except ImportError:
    class QuantizedCache:  # type: ignore[override]
        pass

from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

HeadReduce = Literal["amax", "mean"]


# ── BasePress ────────────────────────────────────────────────────────────────

def _resolve_decoder_model(model: PreTrainedModel):
    """Return the decoder module that owns `.layers` for LM or VLM wrappers."""
    candidates = []
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            candidates.append(inner.language_model)
        candidates.append(inner)
    candidates.append(model)

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return candidate
        nested = getattr(candidate, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            return nested

    raise AttributeError(f"Could not resolve decoder layers for model {type(model)}")


@dataclass
class BasePress:
    """Base class for KV cache compression methods."""

    def post_init_from_model(self, model: PreTrainedModel):
        pass

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = kwargs.get("past_key_value")
        if cache is None:
            return output
        cache_layer = cache.layers[module.layer_idx] if hasattr(cache, "layers") else None
        q_len = hidden_states.shape[1]

        if kwargs["cache_position"][-1] > q_len:
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            if cache_layer is None:
                raise AttributeError(f"Quantized cache for {type(cache)} does not expose layers")
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        elif cache_layer is not None:
            cache_layer.keys = keys
            cache_layer.values = values
        elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values
        else:
            raise AttributeError(f"Unsupported cache layout for {type(cache)}")

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """Context manager to apply compression to all attention layers during prefill."""
        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = _resolve_decoder_model(model)
            for layer in language_model.layers:
                if hasattr(language_model, "rotary_emb"):
                    layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()


# ── VizCapture ───────────────────────────────────────────────────────────────

@dataclass
class VizCapture:
    """Lightweight per-layer score/mask logger for image token presses.

    Attach to an ImageTokenTopKPress via ``attach_viz_capture()`` before calling
    ``model.generate()``.  After generation, read ``.layer_scores`` and
    ``.keep_masks``.  Detach with ``detach_viz_capture()`` when done, or reset
    between samples with ``.reset()``.

    All tensors are stored on CPU (float32 / bool) to avoid GPU memory pressure.
    Visualization semantics are zero-overhead when not attached — the press checks
    ``self._viz_capture is not None`` before any recording.
    """

    layer_scores: List[torch.Tensor] = field(default_factory=list)
    keep_masks: List[torch.Tensor] = field(default_factory=list)
    n_image_keep: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self.layer_scores.clear()
        self.keep_masks.clear()
        self.n_image_keep.clear()

    def record(self, *, agg_score: torch.Tensor, keep_mask: torch.Tensor, n_keep: int) -> None:
        self.layer_scores.append(agg_score)
        self.keep_masks.append(keep_mask)
        self.n_image_keep.append(n_keep)

    def to_arrays(self) -> Dict[str, Any]:
        import numpy as np
        if not self.layer_scores:
            return {
                "layer_scores": np.zeros((0, 0), dtype=np.float32),
                "keep_masks": np.zeros((0, 0), dtype=bool),
                "n_image_keep": np.zeros(0, dtype=np.int32),
            }
        scores = torch.stack(self.layer_scores).numpy().astype(np.float32)
        masks = torch.stack(self.keep_masks).numpy()
        return {
            "layer_scores": scores,
            "keep_masks": masks,
            "n_image_keep": np.array(self.n_image_keep, dtype=np.int32),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_num_heads(module: nn.Module) -> int:
    return int(getattr(module, "num_heads", module.config.num_attention_heads))


def _get_num_kv_heads(module: nn.Module) -> int:
    return int(getattr(module, "num_key_value_heads", getattr(module.config, "num_key_value_heads", _get_num_heads(module))))


def _aggregate_scores_to_kv_heads(scores: torch.Tensor, module: nn.Module, reduce: HeadReduce = "amax") -> torch.Tensor:
    if scores.dim() == 2:
        scores = scores.unsqueeze(0)
    if scores.dim() != 3:
        raise ValueError(f"Expected [B, H, T] scores, got {tuple(scores.shape)}")

    num_score_heads = int(scores.shape[1])
    num_kv_heads = _get_num_kv_heads(module)
    if num_score_heads == num_kv_heads:
        return scores

    if num_score_heads == 1 and num_kv_heads > 1:
        return scores.expand(scores.shape[0], num_kv_heads, scores.shape[-1])

    if num_score_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot aggregate {num_score_heads} score heads into {num_kv_heads} kv heads for layer {module.layer_idx}"
        )

    n_groups = num_score_heads // num_kv_heads
    scores = scores.view(scores.shape[0], num_kv_heads, n_groups, scores.shape[-1])
    if reduce == "amax":
        return scores.amax(dim=2)
    if reduce == "mean":
        return scores.mean(dim=2)
    raise ValueError(f"Unsupported head reduction: {reduce}")


def _compute_n_image_keep(
    n_image: int,
    n_text: int,
    image_keep_ratio: Optional[float],
    total_keep_ratio: Optional[float],
) -> int:
    """Compute how many image tokens to keep.

    If ``total_keep_ratio`` is given, it is treated as the fraction of ALL tokens
    (text + image) to retain. Text tokens are always kept, so:
        n_image_keep = ceil(total_keep_ratio * (n_text + n_image)) - n_text

    If only ``image_keep_ratio`` is given (legacy), it is the fraction of image
    tokens to retain directly.
    """
    if total_keep_ratio is not None:
        total_tokens = n_text + n_image
        total_keep = int(math.ceil(total_keep_ratio * total_tokens))
        n_image_keep = total_keep - n_text
    else:
        assert image_keep_ratio is not None
        n_image_keep = int(math.ceil(n_image * image_keep_ratio))

    return min(n_image, max(0, n_image_keep))


def _iterative_topk(scores: torch.Tensor, final_k: int, n_rounds: int = 4) -> torch.Tensor:
    """Iterative top-k: narrow the candidate pool over n_rounds before final per-head top-k."""
    n_free = scores.shape[-1]
    k_clamped = min(final_k, n_free)

    if n_rounds <= 1 or k_clamped >= n_free:
        return torch.topk(scores, k=k_clamped, dim=-1).indices

    schedule = [
        max(k_clamped, int(math.ceil(n_free - (n_free - k_clamped) * (i + 1) / n_rounds)))
        for i in range(n_rounds)
    ]
    schedule[-1] = k_clamped

    pool = torch.arange(n_free, device=scores.device, dtype=torch.long)

    for i, round_k in enumerate(schedule):
        pool_scores = scores[:, pool]
        k = min(round_k, pool.numel())
        if i == len(schedule) - 1:
            final_local = torch.topk(pool_scores, k=k, dim=-1).indices
            return pool[final_local]
        agg = pool_scores.amax(dim=0)
        local_idx = torch.topk(agg, k=k).indices
        pool = pool[local_idx]

    return pool.unsqueeze(0).expand(scores.shape[0], -1)


def _find_image_blocks(image_positions: torch.Tensor) -> List[torch.Tensor]:
    """Split image token positions into per-image contiguous blocks."""
    if image_positions.numel() == 0:
        return []
    blocks: List[torch.Tensor] = []
    start = 0
    pos = image_positions
    for i in range(1, pos.numel()):
        if pos[i].item() != pos[i - 1].item() + 1:
            blocks.append(pos[start:i])
            start = i
    blocks.append(pos[start:])
    return blocks


# ── ImageTokenTopKPress ───────────────────────────────────────────────────────

@dataclass
class ImageTokenTopKPress(BasePress):
    """Keep all non-image tokens and retain only the top-k image tokens per layer/head.

    Compression budget can be specified in two ways (mutually exclusive):
    - ``image_keep_ratio``: fraction of image tokens to keep (legacy, image-only basis).
    - ``total_keep_ratio``: fraction of ALL tokens (text + image) to keep. Text tokens
      are always preserved; image tokens fill the remaining budget. This puts all methods
      on the same r_eff_prompt basis for fair comparison.

    Positional forced-keep:
    - ``n_initial_keep``: always keep the first N image tokens (global or per-image).
    - ``n_recent_keep``: always keep the last N image tokens (global or per-image).
    - ``n_random_keep``: always keep N randomly chosen image tokens (control).
    - ``per_image_forced``: if True, apply initial/recent per image block instead of globally.
    """

    image_keep_ratio: Optional[float] = None
    total_keep_ratio: Optional[float] = None
    head_reduce: HeadReduce = "amax"
    n_iterative_rounds: int = 1
    n_initial_keep: int = 0
    n_recent_keep: int = 0
    n_random_keep: int = 0
    per_image_forced: bool = False
    current_image_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _forced_pos_indices: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _viz_capture: Optional[VizCapture] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.total_keep_ratio is not None and self.image_keep_ratio is not None:
            raise ValueError("Specify either image_keep_ratio or total_keep_ratio, not both")
        if self.total_keep_ratio is None and self.image_keep_ratio is None:
            self.image_keep_ratio = 1.0
        if self.image_keep_ratio is not None:
            assert 0.0 <= self.image_keep_ratio <= 1.0, "image_keep_ratio must be in [0, 1]"
        if self.total_keep_ratio is not None:
            assert 0.0 <= self.total_keep_ratio <= 1.0, "total_keep_ratio must be in [0, 1]"

    def _compute_forced_pos_indices(self, image_positions: torch.Tensor) -> torch.Tensor:
        n_image = image_positions.numel()
        forced: set = set()

        if self.per_image_forced:
            blocks = _find_image_blocks(image_positions)
            for block in blocks:
                block_start = (image_positions == block[0]).nonzero(as_tuple=True)[0][0].item()
                block_size = block.numel()
                n_init = min(self.n_initial_keep, block_size)
                n_rec = min(self.n_recent_keep, block_size)
                for i in range(n_init):
                    forced.add(int(block_start) + i)
                for i in range(n_rec):
                    forced.add(int(block_start) + block_size - 1 - i)
        else:
            for i in range(min(self.n_initial_keep, n_image)):
                forced.add(i)
            for i in range(min(self.n_recent_keep, n_image)):
                forced.add(n_image - 1 - i)

        if self.n_random_keep > 0:
            candidates = [i for i in range(n_image) if i not in forced]
            n_rand = min(self.n_random_keep, len(candidates))
            if n_rand > 0:
                forced.update(random.sample(candidates, n_rand))

        if not forced:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted(forced), dtype=torch.long)

    def set_image_positions(self, image_positions: torch.Tensor) -> None:
        self.current_image_positions = image_positions.detach().cpu().long().flatten()
        self._forced_pos_indices = self._compute_forced_pos_indices(self.current_image_positions)

    def clear_sample_context(self) -> None:
        self.current_image_positions = None
        self._forced_pos_indices = None

    def attach_viz_capture(self, capture: VizCapture) -> None:
        self._viz_capture = capture

    def detach_viz_capture(self) -> None:
        self._viz_capture = None

    def _should_skip(self) -> bool:
        if self.total_keep_ratio is not None:
            return self.total_keep_ratio >= 1.0
        return (self.image_keep_ratio or 1.0) >= 1.0

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._should_skip():
            return keys, values
        if self.current_image_positions is None:
            raise RuntimeError("Image positions must be set before entering the press context")
        if keys.shape[0] != 1:
            raise ValueError("ImageTokenTopKPress currently only supports batch size 1")

        image_positions = self.current_image_positions.to(keys.device)
        image_positions = image_positions[(image_positions >= 0) & (image_positions < keys.shape[2])]
        if image_positions.numel() == 0:
            return keys, values

        seq_len = keys.shape[2]
        n_image = image_positions.numel()
        n_text = seq_len - n_image
        n_image_keep = _compute_n_image_keep(n_image, n_text, self.image_keep_ratio, self.total_keep_ratio)

        score_tensor = self.score_image_tokens(module, hidden_states, keys, values, attentions, kwargs, image_positions)
        score_tensor = _aggregate_scores_to_kv_heads(score_tensor, module, reduce=self.head_reduce)

        if score_tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 in score tensor, got {tuple(score_tensor.shape)}")
        if score_tensor.shape[-1] != image_positions.numel():
            raise ValueError(
                f"Score/image length mismatch: scores {tuple(score_tensor.shape)} vs image positions {tuple(image_positions.shape)}"
            )
        if score_tensor.shape[1] != keys.shape[1]:
            raise ValueError(
                f"KV-head mismatch: scores have {score_tensor.shape[1]} heads but cache has {keys.shape[1]} heads"
            )

        all_positions = torch.arange(seq_len, device=keys.device, dtype=torch.long)
        non_image_mask = torch.ones(seq_len, dtype=torch.bool, device=keys.device)
        non_image_mask[image_positions] = False
        non_image_positions = all_positions[non_image_mask]

        forced_idx = self._forced_pos_indices
        has_forced = forced_idx is not None and forced_idx.numel() > 0
        if has_forced:
            forced_idx_dev = forced_idx.to(keys.device)
            forced_image_positions = image_positions[forced_idx_dev]
            forced_mask = torch.zeros(n_image, dtype=torch.bool, device=keys.device)
            forced_mask[forced_idx_dev] = True
            n_forced = int(forced_idx_dev.numel())
        else:
            forced_image_positions = torch.empty(0, dtype=torch.long, device=keys.device)
            forced_mask = torch.zeros(n_image, dtype=torch.bool, device=keys.device)
            n_forced = 0

        remaining_budget = max(0, n_image_keep - n_forced)

        if remaining_budget > 0 and (~forced_mask).any():
            free_indices = (~forced_mask).nonzero(as_tuple=True)[0]
            free_scores = score_tensor[0][:, free_indices]
            k = min(remaining_budget, free_indices.numel())
            topk_within_free = _iterative_topk(free_scores, k, n_rounds=self.n_iterative_rounds)
            scored_image_positions = image_positions[free_indices[topk_within_free]]
        elif remaining_budget > 0:
            scored_image_positions = torch.empty((keys.shape[1], 0), dtype=torch.long, device=keys.device)
        else:
            scored_image_positions = torch.empty((keys.shape[1], 0), dtype=torch.long, device=keys.device)

        base_positions = non_image_positions.unsqueeze(0).expand(keys.shape[1], -1)
        forced_expanded = forced_image_positions.unsqueeze(0).expand(keys.shape[1], -1)
        keep_positions = torch.cat([base_positions, forced_expanded, scored_image_positions], dim=-1)
        keep_positions, _ = torch.sort(keep_positions, dim=-1)

        if self._viz_capture is not None:
            agg_score = score_tensor[0].amax(dim=0).detach().cpu().float()
            keep_mask = torch.zeros(n_image, dtype=torch.bool)
            if has_forced:
                keep_mask[forced_idx] = True
            if remaining_budget > 0 and (~forced_mask).any():
                scored_indices_union = free_indices[topk_within_free].cpu().unique()
                keep_mask[scored_indices_union] = True
            self._viz_capture.record(agg_score=agg_score, keep_mask=keep_mask, n_keep=n_image_keep)

        gather_idx = keep_positions.unsqueeze(0).unsqueeze(-1).expand(1, keys.shape[1], keep_positions.shape[-1], module.head_dim)
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


# ── Student Presses ───────────────────────────────────────────────────────────

@dataclass
class VisualUtilityStudentPress(ImageTokenTopKPress):
    """3-branch visual-utility student for image-only KV eviction (LLaVA-1.5).

    Loads a `VisualUtilityStudent` checkpoint trained against future-decode
    attention. At each layer in the scope, scores image tokens using the
    student's 3-branch (CNN + question + raw) head. Layers outside the scope
    are passed through unchanged.

    Per-sample, call `set_image_positions()` (parent) and `set_question_positions()`
    before `model.generate()`.
    """

    student_model_name: str = ""
    grid_h: int = 24
    grid_w: int = 24
    _student: Any = field(default=None, init=False, repr=False)
    _loaded_name: Optional[str] = field(default=None, init=False, repr=False)
    _scope_layers: Optional[set[int]] = field(default=None, init=False, repr=False)
    _question_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    student_score_total_ms: float = field(default=0.0, init=False, repr=False)

    def reset_student_timing(self) -> None:
        self.student_score_total_ms = 0.0

    def set_question_positions(self, q_positions: torch.Tensor) -> None:
        self._question_positions = q_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        super().clear_sample_context()
        self._question_positions = None

    def post_init_from_model(self, model) -> None:
        if not self.student_model_name:
            raise ValueError("student_model_name must be set for VisualUtilityStudentPress")
        if self.student_model_name != self._loaded_name:
            from kvpress.presses.visual_utility_student import VisualUtilityStudent

            self._loaded_name = self.student_model_name
            self._student = VisualUtilityStudent.from_pretrained(self.student_model_name)
            self._scope_layers = set(self._student.layer_indices)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self._scope_layers is not None and module.layer_idx not in self._scope_layers:
            return keys, values
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._student is None:
            raise RuntimeError("Student model not loaded; call post_init_from_model(model) first")
        if self._question_positions is None:
            raise RuntimeError("Call set_question_positions() before generate()")
        device = hidden_states.device
        dtype = hidden_states.dtype
        layer = self._student.layers[str(module.layer_idx)]
        layer = layer.to(device=device, dtype=dtype).eval()
        image_idx = image_positions.to(device=device, dtype=torch.long).flatten()
        q_idx = self._question_positions.to(device=device, dtype=torch.long).flatten()
        _t0 = time.perf_counter()
        with torch.no_grad():
            scores = layer(hidden_states, image_idx, q_idx, self.grid_h, self.grid_w)
        self.student_score_total_ms += (time.perf_counter() - _t0) * 1000.0
        return scores.unsqueeze(1)


@dataclass
class VisualUtilityStudentOneVisionPress(ImageTokenTopKPress):
    """OneVision-Qwen2 variant of `VisualUtilityStudentPress`.

    Wires a `VisualUtilityStudentOneVision` checkpoint (1D conv branch, no
    fixed grid). Identical scope/forced-keep semantics as the LLaVA-1.5 press.
    """

    student_model_name: str = ""
    _student: Any = field(default=None, init=False, repr=False)
    _loaded_name: Optional[str] = field(default=None, init=False, repr=False)
    _scope_layers: Optional[set[int]] = field(default=None, init=False, repr=False)
    _question_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    student_score_total_ms: float = field(default=0.0, init=False, repr=False)

    def reset_student_timing(self) -> None:
        self.student_score_total_ms = 0.0

    def set_question_positions(self, q_positions: torch.Tensor) -> None:
        self._question_positions = q_positions.detach().cpu().long().flatten()

    def clear_sample_context(self) -> None:
        super().clear_sample_context()
        self._question_positions = None

    def post_init_from_model(self, model) -> None:
        if not self.student_model_name:
            raise ValueError("student_model_name must be set for VisualUtilityStudentOneVisionPress")
        if self.student_model_name != self._loaded_name:
            from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision

            self._loaded_name = self.student_model_name
            self._student = VisualUtilityStudentOneVision.from_pretrained(self.student_model_name)
            self._scope_layers = set(self._student.layer_indices)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self._scope_layers is not None and module.layer_idx not in self._scope_layers:
            return keys, values
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    def score_image_tokens(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._student is None:
            raise RuntimeError("Student model not loaded; call post_init_from_model(model) first")
        if self._question_positions is None:
            raise RuntimeError("Call set_question_positions() before generate()")
        device = hidden_states.device
        dtype = hidden_states.dtype
        layer = self._student.layers[str(module.layer_idx)]
        layer = layer.to(device=device, dtype=dtype).eval()
        image_idx = image_positions.to(device=device, dtype=torch.long).flatten()
        q_idx = self._question_positions.to(device=device, dtype=torch.long).flatten()
        _t0 = time.perf_counter()
        with torch.no_grad():
            scores = layer(hidden_states, image_idx, q_idx)
        self.student_score_total_ms += (time.perf_counter() - _t0) * 1000.0
        return scores.unsqueeze(1)
