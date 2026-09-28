
"""Shared KV-cache trimming and greedy decode helpers."""

from __future__ import annotations

import torch


def trim_kv_cache_per_layer(past_kv, keep_masks: dict[int, torch.Tensor]):
    if hasattr(past_kv, "key_cache"):
        for layer_idx in range(len(past_kv.key_cache)):
            if layer_idx not in keep_masks:
                continue
            mask = keep_masks[layer_idx].to(past_kv.key_cache[layer_idx].device)
            past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, mask, :].contiguous()
            past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, mask, :].contiguous()
        return past_kv

    if hasattr(past_kv, "layers"):
        for layer_idx, layer in enumerate(past_kv.layers):
            if layer_idx not in keep_masks:
                continue
            mask = keep_masks[layer_idx].to(layer.keys.device)
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    from transformers import DynamicCache

    new_cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_kv):
        if layer_idx in keep_masks:
            mask = keep_masks[layer_idx].to(keys.device)
            keys = keys[:, :, mask, :].contiguous()
            values = values[:, :, mask, :].contiguous()
        new_cache.key_cache.append(keys)
        new_cache.value_cache.append(values)
    return new_cache


@torch.no_grad()
def greedy_decode_with_kv(
    model,
    past_kv,
    first_next_token: torch.Tensor,
    prompt_len: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """Emit first_next_token, then continue decoding with explicit absolute positions."""
    out_tokens: list[int] = [int(first_next_token.item())]
    if out_tokens[0] == eos_token_id:
        return torch.tensor(out_tokens, dtype=torch.long)

    next_token = first_next_token
    pos = int(prompt_len)
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
            output_attentions=False,
            return_dict=True,
        )
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_token.item())
        out_tokens.append(tok)
        pos += 1
        if tok == eos_token_id:
            break
    return torch.tensor(out_tokens, dtype=torch.long)
