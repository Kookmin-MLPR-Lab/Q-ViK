from __future__ import annotations

import torch
from transformers import DynamicCache

from qvik.eval.kv_decode_utils import trim_kv_cache_per_layer
from qvik.eval.text_kv_eviction import (
    TextKVCacheManager,
    TextKVConfig,
    _rerotate_keys,
    _rotate_half,
)


class _RotaryEmbedding:
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.einsum(
            "bi,j->bij", position_ids.float(), self.inv_freq.to(position_ids.device)
        )
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _IdentityRotaryEmbedding:
    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor):
        shape = (*position_ids.shape, x.shape[-1])
        return (
            torch.ones(shape, dtype=x.dtype, device=x.device),
            torch.zeros(shape, dtype=x.dtype, device=x.device),
        )


def _apply_rope(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: _RotaryEmbedding,
) -> torch.Tensor:
    cos, sin = rotary_emb(hidden_states, positions.unsqueeze(0))
    return hidden_states * cos.unsqueeze(1) + _rotate_half(hidden_states) * sin.unsqueeze(1)


def _cache(num_layers: int, num_heads: int, seq_len: int) -> DynamicCache:
    cache = DynamicCache()
    positions = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
    heads = torch.arange(num_heads, dtype=torch.float32).view(1, num_heads, 1, 1) * 100
    for layer_idx in range(num_layers):
        keys = (positions + heads + layer_idx * 1000).clone()
        cache.key_cache.append(keys)
        cache.value_cache.append(keys.clone())
    return cache


def test_streamingllm_only_evicts_text_after_visual_mask() -> None:
    past = _cache(num_layers=2, num_heads=2, seq_len=10)
    image_positions = torch.tensor([2, 3, 4, 5])
    visual_mask = torch.ones(10, dtype=torch.bool)
    visual_mask[torch.tensor([3, 5])] = False
    visual_keep_masks = {0: visual_mask, 1: visual_mask}
    past = trim_kv_cache_per_layer(past, visual_keep_masks)

    manager = TextKVCacheManager(
        past,
        prompt_len=10,
        image_positions=image_positions,
        visual_keep_masks=visual_keep_masks,
        config=TextKVConfig(
            mode="streamingllm",
            cache_size=3,
            streaming_sink_size=1,
        ),
    )
    past = manager.prune(past, rotary_emb=_IdentityRotaryEmbedding())

    # Q-ViK visual survivors are original positions 2 and 4.  StreamingLLM
    # additionally retains the first text token and two most recent text tokens.
    expected = torch.tensor([0.0, 2.0, 4.0, 8.0, 9.0])
    assert torch.equal(past.key_cache[0][0, 0, :, 0], expected)
    assert manager.stats()["avg_n_visual_cache_final"] == 2.0
    assert manager.stats()["avg_n_text_cache_final"] == 3.0


def test_streamingllm_rerotates_qwen_keys_to_compact_cache_positions() -> None:
    rotary_emb = _RotaryEmbedding(dim=4)
    unrotated = torch.randn(1, 2, 3, 4)
    old_positions = torch.tensor([0, 5, 9])
    new_positions = torch.tensor([0, 1, 2])
    old_keys = _apply_rope(unrotated, old_positions, rotary_emb)

    shifted = _rerotate_keys(
        old_keys, old_positions, new_positions, rotary_emb
    )
    expected = _apply_rope(unrotated, new_positions, rotary_emb)

    torch.testing.assert_close(shifted, expected, atol=1e-5, rtol=1e-5)


def test_h2o_uses_per_head_heavy_hitters_and_protects_visual_tokens() -> None:
    past = _cache(num_layers=1, num_heads=2, seq_len=8)
    image_positions = torch.tensor([2, 3])
    manager = TextKVCacheManager(
        past,
        prompt_len=8,
        image_positions=image_positions,
        visual_keep_masks={},
        config=TextKVConfig(
            mode="h2o",
            cache_size=3,
            h2o_recent_ratio=1 / 3,
        ),
    )

    new_key = torch.tensor([[[[8.0]], [[108.0]]]])
    past.key_cache[0] = torch.cat([past.key_cache[0], new_key], dim=2)
    past.value_cache[0] = torch.cat([past.value_cache[0], new_key.clone()], dim=2)
    manager.append_generated_token()

    attention = torch.zeros(1, 2, 1, 9)
    attention[0, 0, 0, 1] = 10
    attention[0, 0, 0, 5] = 9
    attention[0, 1, 0, 0] = 10
    attention[0, 1, 0, 6] = 9
    manager.update_h2o_scores((attention,), past)
    past = manager.prune(past)

    # Each head gets its own two heavy text entries and the newest text entry.
    # Both heads retain visual positions 2 and 3 regardless of attention score.
    assert torch.equal(
        past.key_cache[0][0, 0, :, 0],
        torch.tensor([1.0, 2.0, 3.0, 5.0, 8.0]),
    )
    assert torch.equal(
        past.key_cache[0][0, 1, :, 0],
        torch.tensor([100.0, 102.0, 103.0, 106.0, 108.0]),
    )
    assert manager.stats()["avg_n_visual_cache_final"] == 2.0
    assert manager.stats()["avg_n_text_cache_final"] == 3.0


def test_h2o_accumulates_scores_for_surviving_entries() -> None:
    past = _cache(num_layers=1, num_heads=1, seq_len=4)
    manager = TextKVCacheManager(
        past,
        prompt_len=4,
        image_positions=torch.empty(0, dtype=torch.long),
        visual_keep_masks={},
        config=TextKVConfig(mode="h2o", cache_size=3, h2o_recent_ratio=1 / 3),
    )

    new_key = torch.tensor([[[[4.0]]]])
    past.key_cache[0] = torch.cat([past.key_cache[0], new_key], dim=2)
    past.value_cache[0] = torch.cat([past.value_cache[0], new_key], dim=2)
    manager.append_generated_token()
    first = torch.tensor([[[[0.0, 4.0, 3.0, 2.0, 1.0]]]])
    manager.update_h2o_scores((first,), past)
    past = manager.prune(past)

    second_key = torch.tensor([[[[5.0]]]])
    past.key_cache[0] = torch.cat([past.key_cache[0], second_key], dim=2)
    past.value_cache[0] = torch.cat([past.value_cache[0], second_key], dim=2)
    manager.append_generated_token()
    second = torch.tensor([[[[0.0, 0.0, 0.0, 1.0]]]])
    manager.update_h2o_scores((second,), past)

    assert torch.equal(
        manager.h2o_scores[0],
        torch.tensor([[4.0, 3.0, 1.0, 1.0]]),
    )
