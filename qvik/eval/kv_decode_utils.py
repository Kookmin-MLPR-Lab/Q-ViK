
"""Shared KV-cache trimming and greedy decode helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SinkAbsorbPlan:
    sink_positions: dict[int, torch.Tensor]
    drop_weights: dict[int, torch.Tensor]
    value_scale: float = 1.0


def trim_kv_cache_per_layer(
    past_kv,
    keep_masks: dict[int, torch.Tensor],
    absorb_plan: SinkAbsorbPlan | None = None,
):
    if hasattr(past_kv, "key_cache"):
        for layer_idx in range(len(past_kv.key_cache)):
            if layer_idx not in keep_masks:
                continue
            mask = keep_masks[layer_idx].to(past_kv.key_cache[layer_idx].device)
            _absorb_layer_values(
                past_kv.value_cache[layer_idx],
                mask,
                _layer_sink_positions(absorb_plan, layer_idx, mask.device),
                _layer_drop_weights(absorb_plan, layer_idx, mask.device),
                _plan_scale(absorb_plan),
            )
            past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, mask, :].contiguous()
            past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, mask, :].contiguous()
        return past_kv

    if hasattr(past_kv, "layers"):
        for layer_idx, layer in enumerate(past_kv.layers):
            if layer_idx not in keep_masks:
                continue
            mask = keep_masks[layer_idx].to(layer.keys.device)
            _absorb_layer_values(
                layer.values,
                mask,
                _layer_sink_positions(absorb_plan, layer_idx, mask.device),
                _layer_drop_weights(absorb_plan, layer_idx, mask.device),
                _plan_scale(absorb_plan),
            )
            layer.keys = layer.keys[:, :, mask, :].contiguous()
            layer.values = layer.values[:, :, mask, :].contiguous()
        return past_kv

    from transformers import DynamicCache

    new_cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_kv):
        if layer_idx in keep_masks:
            mask = keep_masks[layer_idx].to(keys.device)
            _absorb_layer_values(
                values,
                mask,
                _layer_sink_positions(absorb_plan, layer_idx, keys.device),
                _layer_drop_weights(absorb_plan, layer_idx, keys.device),
                _plan_scale(absorb_plan),
            )
            keys = keys[:, :, mask, :].contiguous()
            values = values[:, :, mask, :].contiguous()
        new_cache.key_cache.append(keys)
        new_cache.value_cache.append(values)
    return new_cache


def _layer_sink_positions(plan: SinkAbsorbPlan | None, layer_idx: int, device: torch.device) -> torch.Tensor:
    if plan is None:
        return torch.empty(0, dtype=torch.long, device=device)
    return plan.sink_positions.get(layer_idx, torch.empty(0, dtype=torch.long)).to(device=device)


def _layer_drop_weights(plan: SinkAbsorbPlan | None, layer_idx: int, device: torch.device) -> torch.Tensor:
    if plan is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    return plan.drop_weights.get(layer_idx, torch.empty(0, dtype=torch.float32)).to(device=device)


def _plan_scale(plan: SinkAbsorbPlan | None) -> float:
    return 0.0 if plan is None else float(plan.value_scale)


def _absorb_layer_values(
    values: torch.Tensor,
    keep_mask: torch.Tensor,
    sink_positions: torch.Tensor,
    drop_weights: torch.Tensor,
    value_scale: float,
) -> None:
    if value_scale == 0.0 or sink_positions.numel() == 0:
        return
    drop_positions = (~keep_mask).nonzero(as_tuple=False).flatten()
    if drop_positions.numel() == 0:
        return
    valid_sink = sink_positions[sink_positions < values.shape[2]]
    if valid_sink.numel() == 0:
        return
    selected_weights = drop_weights.index_select(0, drop_positions).to(dtype=values.dtype)
    selected_weights = selected_weights / selected_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    dropped = values.index_select(2, drop_positions)
    payload = (dropped * selected_weights.view(1, 1, -1, 1)).sum(dim=2, keepdim=True)
    values[:, :, valid_sink, :] += payload.mul(value_scale / float(valid_sink.numel()))


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
