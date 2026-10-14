"""lmms-eval wrapper: LLaVA-1.5-7B + SparseVLM as released (Gumpest/SparseVLMs,
main branch, default USE_VERSION=1_0), loaded through the repo's own builder
(LlavaLlamaDynamicForCausalLM) and generate(), eager attention, env `sparse-fastv`
(transformers 4.37.0).

The only change is the token schedule. The repo ships fixed schedules for
RETAIN_TOKN 192/128/96/64 ("for other numbers modify score.py"); here the
RETAIN_TOKN=192 shape (300/200/110 image tokens after layers 2/6/15) is scaled
per sample so the KV cache averaged over all 32 layers -- including the merged
cluster tokens SparseVLM keeps -- is keep * L (token_budget.sparsevlm_schedule).
The measured per-layer KV length of every prefill is logged next to the results.
Pruning happens inside the prefill forward, so no delayed replay is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

SPARSE_ROOT = Path("/workspace/nips/SparseVLMs")
sys.path.insert(0, str(SPARSE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llava.model.language_model.score as sparse_score  # noqa: E402
import llava.model.language_model.modelling_sparse_llama as sparse_llama  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.api.registry import register_model  # noqa: E402

from token_budget import sparsevlm_kv_ratio, sparsevlm_schedule  # noqa: E402

N_IMAGE = 576


def _flatten(items):
    return [child for item in items for child in item]


@register_model("sparsevlm_llava15_orig")
class SparseVLMLlava15Orig(lmms):
    def __init__(
        self,
        pretrained: str = "/workspace/nips/models/llava-v1.5-7b",
        keep_ratio: float = 0.2,
        device: str = "cuda:0",
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        max_new_tokens: int = 32,
        stats_output_dir: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[sparsevlm_llava15_orig] ignoring unexpected kwargs: {list(kwargs.keys())}")
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1
        model_name = get_model_name_from_path(pretrained)
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=str(self._device), device=str(self._device)
        )
        self._model.eval()
        self._config = self._model.config
        self.keep_ratio = float(keep_ratio)
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.stats_output_dir = stats_output_dir
        self._stats, self._last_kv = [], None
        self._rank, self._world_size = 0, 1

        # Record the KV length per layer of each prefill forward.
        wrapper = self
        inner = self._model.get_model()
        orig_forward = inner.forward

        def recording_forward(*args, **kw):
            out = orig_forward(*args, **kw)
            base = out[2] if isinstance(out, tuple) and len(out) == 3 else None
            pkv = getattr(base, "past_key_values", None)
            if pkv is not None and wrapper._last_kv is None:
                wrapper._last_kv = [layer[0].shape[2] for layer in pkv]
            return out

        inner.forward = recording_forward
        print(f"[sparsevlm_llava15_orig] keep_ratio={self.keep_ratio} RETAIN_TOKN={sparse_score.RETAIN_TOKN} "
              f"version={sparse_score.VERSION} pruning_loc={inner.pruning_loc}", flush=True)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        enc = self._tokenizer.encode(string, add_special_tokens=bool(add_special_tokens))
        return enc[-left_truncate_len:] if left_truncate_len else enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    @torch.inference_mode()
    def _generate_one(self, prompt: str, image, max_new_tokens: int) -> str:
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._device)
        image_tensor = process_images([image], self._image_processor, self._config).to(self._device, dtype=torch.float16)

        prompt_len = input_ids.shape[1] - 1 + N_IMAGE
        schedule = sparsevlm_schedule(self.keep_ratio, prompt_len, N_IMAGE)
        sparse_score.sparse_token_dict[sparse_score.RETAIN_TOKN] = schedule

        self._last_kv = None
        out = self._model.generate(input_ids, images=image_tensor, image_sizes=[image.size], do_sample=False,
                                   temperature=0, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True)
        kv = self._last_kv or []
        self._stats.append({"prompt_len": prompt_len, "schedule": schedule,
                            "planned_kv_ratio": sparsevlm_kv_ratio(prompt_len, schedule, N_IMAGE),
                            "measured_kv_ratio": (sum(kv) / len(kv) / prompt_len) if kv else None})

        text = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        conv = conv_templates[self.conv_template]
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop and stop in text:
            text = text.split(stop, 1)[0].strip()
        return text

    def _dump_stats(self, task):
        if not self._stats:
            return
        meas = [s["measured_kv_ratio"] for s in self._stats if s["measured_kv_ratio"] is not None]
        n = len(self._stats)
        summary = {"n": n, "keep_ratio": self.keep_ratio,
                   "avg_planned_kv_ratio": sum(s["planned_kv_ratio"] for s in self._stats) / n,
                   "avg_measured_kv_ratio": (sum(meas) / len(meas)) if meas else None,
                   "n_measured": len(meas),
                   "n_over_budget": sum((s["measured_kv_ratio"] or 0) > self.keep_ratio + 0.01 for s in self._stats),
                   "avg_prompt_len": sum(s["prompt_len"] for s in self._stats) / n}
        out = Path(self.stats_output_dir or ".") / f"{task}_keep_ratio_stats.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"[sparsevlm_llava15_orig] budget stats -> {out}: {summary}", flush=True)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res, task_name = [], None
        for context, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([r.args for r in requests], desc="SparseVLM LLaVA-1.5 Responding"):
            task_name = task
            visuals = _flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            question = context if DEFAULT_IMAGE_TOKEN in context else f"{DEFAULT_IMAGE_TOKEN}\n{context}"
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            gen_kwargs = dict(gen_kwargs)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", self.max_new_tokens))
            until = gen_kwargs.pop("until", None)
            text = self._generate_one(conv.get_prompt(), visuals[0], max_new_tokens)
            for s in ([until] if isinstance(until, str) else (until or [])):
                if s and s in text:
                    text = text.split(s, 1)[0]
            res.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
        self._dump_stats(task_name)
        return res
