"""Total-token budget for in-LLM visual token pruning (FastV, SparseVLM).

Budget = keep * L tokens of KV per layer *averaged over all decoder layers*,
L = prompt length after image expansion. Text tokens are never pruned by
these methods, so only the number of image tokens per layer is solved for.
"""


def fastv_image_keep(keep, prompt_len, n_image, k, num_layers=32):
    """Layers < k see all tokens, layers >= k keep n_text + x tokens."""
    n_text = prompt_len - n_image
    x = (num_layers * keep * prompt_len - k * prompt_len) / (num_layers - k) - n_text
    return int(max(0, min(n_image, round(x))))


def fastv_kv_ratio(prompt_len, n_image, k, x, num_layers=32):
    n_text = prompt_len - n_image
    return (k * prompt_len + (num_layers - k) * (n_text + x)) / (num_layers * prompt_len)


# SparseVLM prunes after layers 2, 6, 15 (pruning_loc), so the KV of layers
# 0-2 is full, then 4 / 9 / 16 layers hold v1 / v2 / v3 image tokens.
SPARSEVLM_SPANS = (3, 4, 9, 16)
SPARSEVLM_SHAPE = (300, 200, 110)  # the repo's RETAIN_TOKN=192 schedule


def sparsevlm_stage_tokens(schedule, n_image=576):
    """Image tokens per stage, replaying score.py / modelling_sparse_llama.py:
    keep min(s, v-1) by attention, then merge the top 30% of the pruned tokens
    into int(n/10)+1 cluster tokens that stay in the sequence."""
    v, out = n_image, []
    for s in schedule:
        kept = min(s, v - 1)
        pruned = v - kept
        if pruned > 0:
            merge = int(pruned * 0.3) + 1
            kept += int(merge / 10) + 1
        v = kept
        out.append(v)
    return out


def sparsevlm_kv_ratio(prompt_len, schedule, n_image=576):
    n_text = prompt_len - n_image
    full, *spans = SPARSEVLM_SPANS
    vs = sparsevlm_stage_tokens(schedule, n_image)
    kv = full * prompt_len + sum(n * (n_text + v) for n, v in zip(spans, vs))
    return kv / (sum(SPARSEVLM_SPANS) * prompt_len)


def sparsevlm_schedule(keep, prompt_len, n_image=576):
    """Scale the repo's 300/200/110 shape so the 32-layer mean KV is keep * L."""
    lo, hi = 0.0, n_image / SPARSEVLM_SHAPE[-1]
    best = None
    for _ in range(60):
        a = (lo + hi) / 2
        sched = [max(1, round(a * s)) for s in SPARSEVLM_SHAPE]
        r = sparsevlm_kv_ratio(prompt_len, sched, n_image)
        if best is None or abs(r - keep) < abs(best[1] - keep):
            best = (sched, r)
        if r > keep:
            hi = a
        else:
            lo = a
    return best[0]
