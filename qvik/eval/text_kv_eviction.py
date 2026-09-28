"""Text-only KV eviction applied after Q-ViK visual-token eviction.

The visual-token decisions are treated as immutable: every visual KV entry
that survives Q-ViK is protected here.  Only prompt/generated text entries are
eligible for the second-stage StreamingLLM or H2O-style policy.

H2O normally initializes its heavy-hitter scores from the full prefill
attention matrix.  Materializing that O(T^2) matrix defeats the purpose on the
long prompts targeted here, so this integration initializes and updates the
same cumulative per-head attention score from decode queries instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

PROMPT_TEXT = 0
VISUAL = 1
GENERATED_TEXT = 2


@dataclass(frozen=True, slots=True)
class TextKVConfig:
    mode: str = "none"
    keep_ratio: float = 0.2
    cache_size: int = 0
    h2o_recent_ratio: float = 0.5
    streaming_sink_size: int = 4

    def normalized(self) -> TextKVConfig:
        aliases = {
            "streaming": "streamingllm",
            "streaming_llm": "streamingllm",
            "local": "streamingllm",
        }
        mode = aliases.get(str(self.mode).strip().lower(), str(self.mode).strip().lower())
        if mode not in {"none", "streamingllm", "h2o"}:
            raise ValueError(
                f"Unsupported text_eviction_mode={self.mode!r}; "
                "expected one of: none, streamingllm, h2o."
            )
        if not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("text_keep_ratio must be in (0, 1].")
        if int(self.cache_size) < 0:
            raise ValueError("text_cache_size must be >= 0.")
        if not 0.0 <= float(self.h2o_recent_ratio) <= 1.0:
            raise ValueError("h2o_recent_ratio must be in [0, 1].")
        if int(self.streaming_sink_size) < 0:
            raise ValueError("streaming_sink_size must be >= 0.")
        return TextKVConfig(
            mode=mode,
            keep_ratio=float(self.keep_ratio),
            cache_size=int(self.cache_size),
            h2o_recent_ratio=float(self.h2o_recent_ratio),
            streaming_sink_size=int(self.streaming_sink_size),
        )

    def resolve_budget(self, prompt_text_tokens: int) -> int:
        if self.cache_size > 0:
            return int(self.cache_size)
        return max(1, round(int(prompt_text_tokens) * self.keep_ratio))


def _cache_layer_pairs(past_kv) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_kv, "key_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    if hasattr(past_kv, "layers"):
        return [(layer.keys, layer.values) for layer in past_kv.layers]
    return list(past_kv)


def _gather_cache_tensor(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices = indices.to(device=cache.device, dtype=torch.long)
    if indices.ndim == 1:
        return cache.index_select(2, indices).contiguous()
    if indices.ndim != 2:
        raise ValueError(f"KV indices must be 1-D or 2-D, got shape={tuple(indices.shape)}.")
    if indices.shape[0] != cache.shape[1]:
        raise ValueError(
            f"Per-head KV indices have {indices.shape[0]} heads, "
            f"but cache has {cache.shape[1]}."
        )
    gather = indices.unsqueeze(0).unsqueeze(-1).expand(
        cache.shape[0], -1, -1, cache.shape[-1]
    )
    return torch.gather(cache, dim=2, index=gather).contiguous()


def select_kv_cache_per_layer(past_kv, keep_indices: dict[int, torch.Tensor]):
    """Select common or per-head sequence indices from each KV-cache layer."""
    if not keep_indices:
        return past_kv

    if hasattr(past_kv, "key_cache"):
        for layer_idx, indices in keep_indices.items():
            past_kv.key_cache[layer_idx] = _gather_cache_tensor(
                past_kv.key_cache[layer_idx], indices
            )
            past_kv.value_cache[layer_idx] = _gather_cache_tensor(
                past_kv.value_cache[layer_idx], indices
            )
        return past_kv

    if hasattr(past_kv, "layers"):
        for layer_idx, indices in keep_indices.items():
            layer = past_kv.layers[layer_idx]
            layer.keys = _gather_cache_tensor(layer.keys, indices)
            layer.values = _gather_cache_tensor(layer.values, indices)
        return past_kv

    from transformers import DynamicCache

    new_cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_kv):
        if layer_idx in keep_indices:
            keys = _gather_cache_tensor(keys, keep_indices[layer_idx])
            values = _gather_cache_tensor(values, keep_indices[layer_idx])
        new_cache.key_cache.append(keys)
        new_cache.value_cache.append(values)
    return new_cache


def _post_visual_token_types(
    past_kv,
    prompt_len: int,
    image_positions: torch.Tensor,
    visual_keep_masks: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    base = torch.full((int(prompt_len),), PROMPT_TEXT, dtype=torch.int8)
    base[image_positions.detach().cpu().long()] = VISUAL
    result: dict[int, torch.Tensor] = {}
    for layer_idx, (keys, _) in enumerate(_cache_layer_pairs(past_kv)):
        layer_types = base
        if layer_idx in visual_keep_masks:
            layer_types = layer_types[visual_keep_masks[layer_idx].detach().cpu().bool()]
        if int(layer_types.numel()) != int(keys.shape[2]):
            raise ValueError(
                f"Layer {layer_idx} token map/cache mismatch after visual eviction: "
                f"map={layer_types.numel()} cache={keys.shape[2]}."
            )
        result[layer_idx] = layer_types.clone()
    return result


def _post_visual_cache_positions(
    past_kv,
    prompt_len: int,
    visual_keep_masks: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Map each post-Q-ViK cache entry to its RoPE position."""
    base = torch.arange(int(prompt_len), dtype=torch.long)
    result: dict[int, torch.Tensor] = {}
    for layer_idx, (keys, _) in enumerate(_cache_layer_pairs(past_kv)):
        positions = base
        if layer_idx in visual_keep_masks:
            positions = positions[visual_keep_masks[layer_idx].detach().cpu().bool()]
        if int(positions.numel()) != int(keys.shape[2]):
            raise ValueError(
                f"Layer {layer_idx} position map/cache mismatch after visual eviction: "
                f"map={positions.numel()} cache={keys.shape[2]}."
            )
        result[layer_idx] = positions.clone()
    return result


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    midpoint = hidden_states.shape[-1] // 2
    first = hidden_states[..., :midpoint]
    second = hidden_states[..., midpoint:]
    return torch.cat((-second, first), dim=-1)


def _rerotate_keys(
    keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    """Move already-RoPE-encoded Qwen2 keys from old to new positions.

    The official StreamingLLM attention keeps unrotated keys in the cache and
    applies RoPE according to their current cache slots. Qwen2 stores rotated
    keys, so applying the delta rotation here is the equivalent operation.
    """
    if torch.equal(old_positions, new_positions):
        return keys
    if keys.shape[2] != old_positions.numel() or old_positions.shape != new_positions.shape:
        raise ValueError(
            "RoPE position/key mismatch: "
            f"keys={keys.shape[2]} old={old_positions.numel()} "
            f"new={new_positions.numel()}."
        )

    positions = torch.cat([old_positions, new_positions]).to(
        device=keys.device, dtype=torch.long
    ).unsqueeze(0)
    cos, sin = rotary_emb(keys, positions)
    split = int(old_positions.numel())
    old_cos, new_cos = cos[:, :split], cos[:, split:]
    old_sin, new_sin = sin[:, :split], sin[:, split:]

    # Divide out any RoPE attention scaling so this is a pure delta rotation.
    denominator = (old_cos.square() + old_sin.square()).clamp_min(1e-12)
    delta_cos = (old_cos * new_cos + old_sin * new_sin) / denominator
    delta_sin = (old_cos * new_sin - old_sin * new_cos) / denominator
    delta_cos = delta_cos.unsqueeze(1)
    delta_sin = delta_sin.unsqueeze(1)
    return (keys * delta_cos + _rotate_half(keys) * delta_sin).contiguous()


def _mean_count(token_types: dict[int, torch.Tensor], token_type: int | None) -> float:
    counts: list[float] = []
    for types in token_types.values():
        if token_type is None:
            selected = types != VISUAL
        else:
            selected = types == token_type
        if selected.ndim == 1:
            counts.append(float(selected.sum().item()))
        else:
            counts.extend(float(value) for value in selected.sum(dim=-1).tolist())
    return sum(counts) / max(1, len(counts))


def _expand_types_for_heads(types: torch.Tensor, num_heads: int) -> torch.Tensor:
    if types.ndim == 1:
        return types.unsqueeze(0).expand(num_heads, -1).clone()
    if types.shape[0] != num_heads:
        raise ValueError(
            f"Token map has {types.shape[0]} heads, but cache has {num_heads}."
        )
    return types


def _aggregate_attention_for_kv_heads(
    attention: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Return original-H2O-style batch/query summed scores [KV heads, K]."""
    if attention.ndim != 4:
        raise ValueError(f"Expected attention [B,H,Q,K], got {tuple(attention.shape)}.")
    scores = attention.detach().float().sum(dim=(0, 2))
    num_query_heads = int(scores.shape[0])
    if num_query_heads == num_kv_heads:
        return scores
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot map {num_query_heads} query heads to {num_kv_heads} KV heads."
        )
    return scores.reshape(num_kv_heads, num_query_heads // num_kv_heads, -1).sum(dim=1)


class TextKVCacheManager:
    """Stateful second-stage text cache policy for one generated sample."""

    def __init__(
        self,
        past_kv,
        *,
        prompt_len: int,
        image_positions: torch.Tensor,
        visual_keep_masks: dict[int, torch.Tensor],
        config: TextKVConfig,
    ) -> None:
        self.config = config.normalized()
        self.prompt_text_tokens = int(prompt_len) - int(image_positions.numel())
        self.budget = self.config.resolve_budget(self.prompt_text_tokens)
        self.token_types = _post_visual_token_types(
            past_kv,
            prompt_len,
            image_positions,
            visual_keep_masks,
        )
        self.cache_positions = _post_visual_cache_positions(
            past_kv,
            prompt_len,
            visual_keep_masks,
        )
        self.h2o_scores: dict[int, torch.Tensor] = {}
        self.eviction_events = 0
        self._first_prompt_kept: float | None = None

    def append_generated_token(self, position: int | None = None) -> None:
        for layer_idx, types in self.token_types.items():
            shape = (*types.shape[:-1], 1)
            generated = torch.full(shape, GENERATED_TEXT, dtype=types.dtype)
            self.token_types[layer_idx] = torch.cat([types, generated], dim=-1)
            if self.config.mode == "streamingllm":
                generated_position = (
                    int(self.cache_positions[layer_idx][-1].item()) + 1
                    if position is None
                    else int(position)
                )
                self.cache_positions[layer_idx] = torch.cat(
                    [
                        self.cache_positions[layer_idx],
                        torch.tensor([generated_position], dtype=torch.long),
                    ]
                )

    def update_h2o_scores(self, attentions: Sequence[torch.Tensor | None], past_kv) -> None:
        pairs = _cache_layer_pairs(past_kv)
        if len(attentions) != len(pairs):
            raise RuntimeError(
                f"H2O needs one attention tensor per cache layer; "
                f"got attentions={len(attentions)} layers={len(pairs)}."
            )
        for layer_idx, ((keys, _), attention) in enumerate(zip(pairs, attentions)):
            if attention is None:
                raise RuntimeError(
                    "H2O text eviction requires decode attentions, but the model "
                    f"returned None for layer {layer_idx}. Use eager attention or "
                    "a Transformers backend that falls back to eager when "
                    "output_attentions=True."
                )
            step_scores = _aggregate_attention_for_kv_heads(attention, int(keys.shape[1]))
            previous = self.h2o_scores.get(layer_idx)
            if previous is not None:
                old_len = int(previous.shape[-1])
                if old_len > int(step_scores.shape[-1]):
                    raise RuntimeError(
                        f"Layer {layer_idx} H2O score shrank before eviction: "
                        f"old={old_len} new={step_scores.shape[-1]}."
                    )
                step_scores[:, :old_len] += previous.to(step_scores.device)
            self.h2o_scores[layer_idx] = step_scores

    def prune(self, past_kv, *, rotary_emb=None):
        if self.config.mode == "none":
            return past_kv
        if self.config.mode == "streamingllm":
            keep_indices = self._streaming_indices()
            if rotary_emb is None:
                raise RuntimeError(
                    "StreamingLLM requires the language model rotary embedding "
                    "to apply its position-shift attention."
                )
        else:
            keep_indices = self._h2o_indices(past_kv)

        removed = False
        shifted_keys: dict[int, torch.Tensor] = {}
        cache_pairs = _cache_layer_pairs(past_kv)
        for layer_idx, indices in keep_indices.items():
            old_len = int(self.token_types[layer_idx].shape[-1])
            removed = removed or int(indices.shape[-1]) < old_len
            self.token_types[layer_idx] = self._gather_types(
                self.token_types[layer_idx], indices
            )
            if self.config.mode == "streamingllm":
                selected_positions = self.cache_positions[layer_idx].index_select(
                    0, indices.cpu().long()
                )
                new_positions = torch.arange(
                    selected_positions.numel(), dtype=torch.long
                )
                keys = cache_pairs[layer_idx][0]
                selected_keys = _gather_cache_tensor(keys, indices)
                shifted_keys[layer_idx] = _rerotate_keys(
                    selected_keys,
                    selected_positions,
                    new_positions,
                    rotary_emb,
                )
                self.cache_positions[layer_idx] = new_positions
            if layer_idx in self.h2o_scores:
                self.h2o_scores[layer_idx] = self._gather_scores(
                    self.h2o_scores[layer_idx], indices
                )
        if removed:
            self.eviction_events += 1
        if self._first_prompt_kept is None:
            self._first_prompt_kept = _mean_count(self.token_types, PROMPT_TEXT)
        past_kv = select_kv_cache_per_layer(past_kv, keep_indices)
        if shifted_keys:
            if not hasattr(past_kv, "key_cache"):
                raise TypeError(
                    "StreamingLLM Qwen2 position shift requires a DynamicCache."
                )
            for layer_idx, keys in shifted_keys.items():
                past_kv.key_cache[layer_idx] = keys
        return past_kv

    def next_streaming_position(self) -> int:
        lengths = {int(positions.numel()) for positions in self.cache_positions.values()}
        if len(lengths) != 1:
            raise RuntimeError(
                "StreamingLLM needs equal post-eviction cache lengths across layers; "
                f"got {sorted(lengths)}."
            )
        return next(iter(lengths))

    def _streaming_indices(self) -> dict[int, torch.Tensor]:
        keep: dict[int, torch.Tensor] = {}
        for layer_idx, types in self.token_types.items():
            if types.ndim != 1:
                raise RuntimeError("StreamingLLM expects a shared token order across heads.")
            text_idx = (types != VISUAL).nonzero(as_tuple=False).flatten()
            if int(text_idx.numel()) <= self.budget:
                keep[layer_idx] = torch.arange(types.numel(), dtype=torch.long)
                continue
            sink_size = min(self.config.streaming_sink_size, self.budget)
            if self.budget > 1:
                sink_size = min(sink_size, self.budget - 1)
            recent_size = self.budget - sink_size
            selected = text_idx[:sink_size]
            if recent_size > 0:
                selected = torch.cat([selected, text_idx[-recent_size:]])
            visual_idx = (types == VISUAL).nonzero(as_tuple=False).flatten()
            keep[layer_idx] = torch.cat([visual_idx, selected]).sort().values
        return keep

    def _h2o_indices(self, past_kv) -> dict[int, torch.Tensor]:
        keep: dict[int, torch.Tensor] = {}
        for layer_idx, (keys, _) in enumerate(_cache_layer_pairs(past_kv)):
            if layer_idx not in self.h2o_scores:
                raise RuntimeError(
                    "H2O scores are not initialized. Run one decode step with "
                    "output_attentions=True before pruning."
                )
            num_heads = int(keys.shape[1])
            types = _expand_types_for_heads(self.token_types[layer_idx], num_heads)
            scores = self.h2o_scores[layer_idx]
            if tuple(scores.shape) != tuple(types.shape):
                raise RuntimeError(
                    f"Layer {layer_idx} H2O score/token-map mismatch: "
                    f"scores={tuple(scores.shape)} map={tuple(types.shape)}."
                )
            head_indices: list[torch.Tensor] = []
            for head_idx in range(num_heads):
                head_types = types[head_idx]
                text_idx = (head_types != VISUAL).nonzero(as_tuple=False).flatten()
                if int(text_idx.numel()) <= self.budget:
                    head_indices.append(torch.arange(head_types.numel(), dtype=torch.long))
                    continue
                recent_size = round(self.budget * self.config.h2o_recent_ratio)
                recent_size = min(max(0, recent_size), self.budget)
                if self.budget > 1 and self.config.h2o_recent_ratio > 0:
                    recent_size = max(1, min(recent_size, self.budget - 1))
                heavy_size = self.budget - recent_size
                recent = text_idx[-recent_size:] if recent_size > 0 else text_idx[:0]
                old_text = text_idx[:-recent_size] if recent_size > 0 else text_idx
                heavy_size = min(heavy_size, int(old_text.numel()))
                if heavy_size > 0:
                    old_scores = scores[head_idx].index_select(
                        0, old_text.to(scores.device)
                    )
                    top = torch.topk(old_scores, k=heavy_size, largest=True).indices.cpu()
                    heavy = old_text.index_select(0, top)
                else:
                    heavy = old_text[:0]
                visual = (head_types == VISUAL).nonzero(as_tuple=False).flatten()
                head_indices.append(
                    torch.cat([visual, heavy, recent]).sort().values
                )
            keep[layer_idx] = torch.stack(head_indices, dim=0)
        return keep

    @staticmethod
    def _gather_types(types: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.cpu().long()
        if indices.ndim == 1:
            return types.index_select(-1, indices)
        types = _expand_types_for_heads(types, int(indices.shape[0]))
        return torch.gather(types, dim=1, index=indices)

    @staticmethod
    def _gather_scores(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(scores.device).long()
        if indices.ndim == 1:
            return scores.index_select(-1, indices)
        return torch.gather(scores, dim=1, index=indices)

    def stats(self) -> dict[str, float | int | str]:
        prompt_kept = (
            _mean_count(self.token_types, PROMPT_TEXT)
            if self._first_prompt_kept is None
            else self._first_prompt_kept
        )
        return {
            "text_eviction_mode": self.config.mode,
            "streaming_position_shift": (
                "qwen2_compact_rope" if self.config.mode == "streamingllm" else "none"
            ),
            "text_cache_budget": self.budget,
            "n_text_prompt_original": self.prompt_text_tokens,
            "avg_n_text_prompt_kept_after_eviction": prompt_kept,
            "text_prompt_keep_ratio_after_eviction": (
                prompt_kept / max(1, self.prompt_text_tokens)
            ),
            "avg_n_text_cache_final": _mean_count(self.token_types, None),
            "avg_n_visual_cache_final": _mean_count(self.token_types, VISUAL),
            "text_eviction_events": self.eviction_events,
        }


@torch.no_grad()
def greedy_decode_with_text_eviction(
    model,
    past_kv,
    first_next_token: torch.Tensor,
    *,
    prompt_len: int,
    image_positions: torch.Tensor,
    visual_keep_masks: dict[int, torch.Tensor],
    eos_token_id: int,
    max_new_tokens: int,
    config: TextKVConfig,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Greedy decode with protected visual KVs and a bounded text cache."""
    config = config.normalized()
    manager = TextKVCacheManager(
        past_kv,
        prompt_len=prompt_len,
        image_positions=image_positions,
        visual_keep_masks=visual_keep_masks,
        config=config,
    )
    rotary_emb = None
    if config.mode == "streamingllm":
        language_model = getattr(model, "model", None)
        rotary_emb = getattr(language_model, "rotary_emb", None)
        if rotary_emb is None:
            raise RuntimeError(
                "StreamingLLM position shift is implemented for a Qwen2-style "
                "model.model.rotary_emb, but this model does not expose one."
            )
        past_kv = manager.prune(past_kv, rotary_emb=rotary_emb)

    out_tokens: list[int] = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id or max_new_tokens <= 1:
        return torch.tensor(out_tokens, dtype=torch.long), manager.stats()

    next_token = first_next_token
    pos = (
        manager.next_streaming_position()
        if config.mode == "streamingllm"
        else int(prompt_len)
    )
    device = next_token.device
    cache_pos = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        cache_pos[0] = pos
        out = model(
            input_ids=next_token,
            past_key_values=past_kv,
            cache_position=cache_pos,
            position_ids=cache_pos.unsqueeze(0),
            use_cache=True,
            output_attentions=config.mode == "h2o",
            return_dict=True,
        )
        past_kv = out.past_key_values
        manager.append_generated_token(pos)
        if config.mode == "h2o":
            manager.update_h2o_scores(out.attentions, past_kv)
        past_kv = manager.prune(past_kv, rotary_emb=rotary_emb)

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_token.item())
        out_tokens.append(token)
        pos = (
            manager.next_streaming_position()
            if config.mode == "streamingllm"
            else pos + 1
        )
        if token == eos_token_id:
            break
    return torch.tensor(out_tokens, dtype=torch.long), manager.stats()
