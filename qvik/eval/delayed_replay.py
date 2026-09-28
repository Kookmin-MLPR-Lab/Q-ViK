"""Delayed replay: evict visual KVs *before* the model produces its first answer token.

Ordinary Q-ViK inference prefills the full prompt, takes the first generated
token from that full-attention forward pass, and only then trims the cache.
For short-answer / multiple-choice benchmarks the whole answer is often that
single token, so accuracy becomes insensitive to keep_ratio.

Delayed replay holds out the final prompt token, prefills the rest, applies
eviction, then feeds the held-out token through the *trimmed* cache. The first
answer token therefore already sees only the kept visual KVs. The held-out token
is always a text token (the assistant-turn suffix), never an image token, so the
visual positions and eviction masks are unchanged.

Shared by the LLaVA-1.5 and LLaVA-OneVision student wrappers.
"""

from __future__ import annotations

import os
from typing import Any

import torch

# Short-answer / multiple-choice task families (lmms-eval task ids, with the
# local "_offline"/"_local"/"_lite" suffix stripped). A family also matches its
# sub-tasks, e.g. "vqav2" matches "vqav2_test_s3" and "mmbench" matches
# "mmbench_en_dev". "all" enables replay for every task.
DEFAULT_DELAYED_REPLAY_TASKS = (
    "mme|pope|mmstar|vizwiz_vqa|gqa|scienceqa_img|mmbench|vqav2|seedbench"
)
ENV_VAR = "QVIK_DELAYED_REPLAY_TASKS"
_TASK_SUFFIXES = ("_offline", "_local", "_lite")


def parse_replay_tasks(spec: str | None) -> frozenset[str]:
    """Parse the task list; `QVIK_DELAYED_REPLAY_TASKS` overrides the argument.

    lmms-eval splits --model_args on commas, so "|" is accepted as a separator.
    An empty string or "none" disables replay.
    """
    spec = os.environ.get(ENV_VAR, spec if spec is not None else DEFAULT_DELAYED_REPLAY_TASKS)
    names = {
        name.strip().lower()
        for name in spec.replace("|", ",").split(",")
        if name.strip()
    }
    names.discard("none")
    return frozenset(names)


def task_family(task: str) -> str:
    for suffix in _TASK_SUFFIXES:
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def should_replay(task: str, replay_tasks: frozenset[str]) -> bool:
    if "all" in replay_tasks:
        return True
    family = task_family(task).lower()
    return any(family == name or family.startswith(name + "_") for name in replay_tasks)


def split_held_out_token(input_ids: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (prefill_ids, held_out_id); held_out_id is None when replay is off."""
    if not enabled or input_ids.shape[1] < 2:
        return input_ids, None
    return input_ids[:, :-1], input_ids[:, -1:]


@torch.no_grad()
def replay_held_out_token(
    model: Any,
    past_kv: Any,
    held_out_id: torch.Tensor,
    prompt_len: int,
) -> tuple[Any, torch.Tensor, int]:
    """Feed the held-out prompt token through the (trimmed) cache.

    `prompt_len` is the length of the prefilled prompt, i.e. the absolute
    position of the held-out token. Positions stay absolute even after the
    cache is trimmed, matching `kv_decode_utils.greedy_decode_with_kv`.
    Returns (past_kv, first_answer_token, new_prompt_len).
    """
    cache_pos = torch.tensor([int(prompt_len)], dtype=torch.long, device=held_out_id.device)
    out = model(
        input_ids=held_out_id,
        past_key_values=past_kv,
        cache_position=cache_pos,
        position_ids=cache_pos.unsqueeze(0),
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return out.past_key_values, next_token, int(prompt_len) + 1


def extend_keep_masks_for_replay(keep_masks: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Append the replayed (always kept, text) token to each per-layer prompt mask.

    After replay the cache holds prompt_len + 1 absolute positions; code that
    rebuilds the cache layout from (prompt_len, keep_masks) -- e.g. the text
    KV eviction manager -- needs masks of that length.
    """
    return {
        layer_idx: torch.cat([mask, torch.ones(1, dtype=mask.dtype, device=mask.device)])
        for layer_idx, mask in keep_masks.items()
    }
