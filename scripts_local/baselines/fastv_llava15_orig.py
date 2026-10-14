"""lmms-eval wrapper: LLaVA-1.5-7B + FastV as released (pkunlp-icler/FastV).

Follows FastV's own lmms-eval recipe (src/FastV/lmms-eval/README.md): latest
haotian-liu/LLaVA (/workspace/nips/LLaVA_fastv) with fastv_kvcache.py's
FastVLlamaModel as the LLaVA language model, and the env's transformers
modeling_llama.py RoPE patched to seq_len=position_ids.max()+1 (env `fastv`).
K = 3 as in fastv_kvcache.py. The kept image-token count (the README's "R") is
set per sample so the KV cache averaged over all 32 layers is keep * L
(token_budget.fastv_image_keep); the image span is located from the prompt
instead of the hard-coded 35:611 (identical for the vicuna_v1 template).
Pruning happens inside the prefill forward, so the first answer token is
already post-pruning (no delayed replay needed).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

LLAVA_ROOT = Path("/workspace/nips/LLaVA_fastv")
sys.path.insert(0, str(LLAVA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.api.registry import register_model  # noqa: E402

from token_budget import fastv_image_keep, fastv_kv_ratio  # noqa: E402

N_IMAGE = 576


def _flatten(items):
    return [child for item in items for child in item]


@register_model("fastv_llava15_orig")
class FastVLlava15Orig(lmms):
    def __init__(
        self,
        pretrained: str = "/workspace/nips/models/llava-v1.5-7b",
        keep_ratio: float = 0.2,
        fastv_k: int = 3,
        device: str = "cuda:0",
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        max_new_tokens: int = 32,
        stats_output_dir: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[fastv_llava15_orig] ignoring unexpected kwargs: {list(kwargs.keys())}")
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
        self.fastv_k = int(fastv_k)
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.stats_output_dir = stats_output_dir
        self._stats = []
        self._rank, self._world_size = 0, 1
        print(f"[fastv_llava15_orig] keep_ratio={self.keep_ratio} K={self.fastv_k}", flush=True)

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

        img_start = int(torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0][0])
        prompt_len = input_ids.shape[1] - 1 + N_IMAGE
        img_keep = fastv_image_keep(self.keep_ratio, prompt_len, N_IMAGE, self.fastv_k)
        lm = self._model.get_model()
        lm.fastv_k, lm.fastv_img_start, lm.fastv_img_len, lm.fastv_img_keep = self.fastv_k, img_start, N_IMAGE, img_keep
        self._stats.append({"prompt_len": prompt_len, "n_text": prompt_len - N_IMAGE, "image_kept": img_keep,
                            "kv_ratio": fastv_kv_ratio(prompt_len, N_IMAGE, self.fastv_k, img_keep)})

        conv = conv_templates[self.conv_template]
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        out = self._model.generate(input_ids, images=image_tensor, image_sizes=[image.size], do_sample=False,
                                   temperature=0, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True)
        text = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        if stop and stop in text:
            text = text.split(stop, 1)[0].strip()
        return text

    def _dump_stats(self, task):
        if not self._stats:
            return
        n = len(self._stats)
        summary = {"n": n, "keep_ratio": self.keep_ratio, "fastv_k": self.fastv_k,
                   "avg_kv_ratio": sum(s["kv_ratio"] for s in self._stats) / n,
                   "avg_image_kept": sum(s["image_kept"] for s in self._stats) / n,
                   "avg_prompt_len": sum(s["prompt_len"] for s in self._stats) / n,
                   "n_over_budget": sum(s["kv_ratio"] > self.keep_ratio + 0.01 for s in self._stats)}
        out = Path(self.stats_output_dir or ".") / f"{task}_keep_ratio_stats.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"[fastv_llava15_orig] budget stats -> {out}: {summary}", flush=True)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res, task_name = [], None
        for context, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm([r.args for r in requests], desc="FastV LLaVA-1.5 Responding"):
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
