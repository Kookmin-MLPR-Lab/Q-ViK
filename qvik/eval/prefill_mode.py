"""How the first answer token is produced relative to Q-ViK visual KV eviction.

`prefill_mode="qvik"` (default, every task): the prompt is prefilled without its
final token, visual KVs are evicted, and the final prompt token is then fed
through the compressed cache. The first answer token -- often the whole answer
on short-answer / multiple-choice benchmarks -- therefore sees only the kept
visual KVs. The final token is always a text token (the assistant-turn suffix),
so image positions and eviction masks are unchanged.

`prefill_mode="origin"`: the full prompt is prefilled, the first answer token is
taken from that full-attention pass, and eviction happens afterwards.

Shared by the LLaVA-1.5 and LLaVA-OneVision student wrappers.
"""

from __future__ import annotations

from typing import Any

import torch

PREFILL_MODES = ("qvik", "origin")
DEFAULT_PREFILL_MODE = "qvik"


def normalize_prefill_mode(mode: str | None) -> str:
    mode = (mode or DEFAULT_PREFILL_MODE).strip().lower()
    if mode not in PREFILL_MODES:
        raise ValueError(f"prefill_mode must be one of {PREFILL_MODES}, got {mode!r}")
    return mode


def split_last_token(input_ids: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (prefill_ids, last_token_id); last_token_id is None in origin mode."""
    if mode == "origin" or input_ids.shape[1] < 2:
        return input_ids, None
    return input_ids[:, :-1], input_ids[:, -1:]


@torch.no_grad()
def feed_last_token(
    model: Any,
    past_kv: Any,
    last_token_id: torch.Tensor,
    prompt_len: int,
) -> tuple[Any, torch.Tensor, int]:
    """Feed the final prompt token through the (compressed) cache.

    `prompt_len` is the length of the prefilled prompt, i.e. the absolute
    position of the final token. Positions stay absolute even after the cache
    is trimmed, matching `kv_decode_utils.greedy_decode_with_kv`.
    Returns (past_kv, first_answer_token, new_prompt_len).
    """
    cache_pos = torch.tensor([int(prompt_len)], dtype=torch.long, device=last_token_id.device)
    out = model(
        input_ids=last_token_id,
        past_key_values=past_kv,
        cache_position=cache_pos,
        position_ids=cache_pos.unsqueeze(0),
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return out.past_key_values, next_token, int(prompt_len) + 1
