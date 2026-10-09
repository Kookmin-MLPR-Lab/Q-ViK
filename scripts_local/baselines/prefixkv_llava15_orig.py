"""lmms-eval wrapper: LLaVA-1.5-7B + PrefixKV exactly as in the official repo
(THU-MIG/PrefixKV, pristine clone at /workspace/nips/PrefixKV).

Uses the official ``llava`` package, ``patch_llama_attention_forward`` and
``prefixkv.PrefixKV`` unmodified: per-layer forget ratios come from
``confs/prefixkv_llava-v1.5-7b_<ratio>.json`` and are applied over the WHOLE
prompt (text and image); the cache is re-trimmed after every decode step.
``ratio`` is the compression ratio, so keep_ratio=0.2 -> ratio=0.8.
Settings follow the official eval_rouge.py: start_size=1, recent_size=2047,
bfloat16, greedy decoding.

Positions (deviation from the official code, chosen 2026-10-08): the official
LlavaLlamaForCausalLM.forward takes no position_ids, so transformers numbers
every token after the prefill from the *compressed* cache length. The official
pipeline hides this because its first answer token comes from the uncompressed
prefill; with delayed replay the first token is already mis-positioned and the
model collapses (keep 0.5 MME: 16/16 empty answers, MME 441). Tokens after the
prefill therefore use logical (uncompressed) positions, as LOOK-M and Q-ViK do.
Token selection, per-layer budgets and per-step trimming are unchanged.

The decode loop mirrors cache_generate.greedy_search (forward with
output_attentions=True, then ``kv_cache(past, num_of_token, attentions)``).
Only change vs. the official code: with delayed_replay=True the last prompt
token is held out of the prefill and replayed through the compressed cache,
so the first answer token is produced post-eviction. delayed_replay=False is
the official behaviour (first token from the uncompressed prefill logits).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

PREFIXKV_ROOT = Path("/workspace/nips/PrefixKV")
sys.path.insert(0, str(PREFIXKV_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from patch_attention_forward import patch_llama_attention_forward  # noqa: E402
from prefixkv import PrefixKV  # noqa: E402

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.api.registry import register_model  # noqa: E402


def _flatten(items):
    return [child for item in items for child in item]


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no")
    return bool(value)


@register_model("prefixkv_llava15_orig")
class PrefixKVLlava15Orig(lmms):
    """Official PrefixKV over the whole prompt, optional delayed replay."""

    def __init__(
        self,
        pretrained: str = "/workspace/nips/models/llava-v1.5-7b",
        keep_ratio: float = 0.2,
        device: str = "cuda:0",
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        max_new_tokens: int = 32,
        stats_output_dir: str = "",
        delayed_replay: bool | str = True,
        start_size: int = 1,
        recent_size: int = 2047,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[prefixkv_llava15_orig] ignoring unexpected kwargs: {list(kwargs.keys())}")

        patch_llama_attention_forward()
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("PrefixKV keeps a per-sample, per-layer budget; batch_size must be 1.")

        model_name = get_model_name_from_path(pretrained)
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            self._max_length,
        ) = load_pretrained_model(pretrained, None, model_name, device=str(self._device))
        self._model.eval()
        self._config = self._model.config
        self.model_name = model_name
        # Compat shim: the official patched forward reads self.pretraining_tp
        # (transformers 4.31); 4.33 keeps it on the config only.
        for layer in self._model.model.layers:
            layer.self_attn.pretraining_tp = getattr(self._config, "pretraining_tp", 1)

        self.keep_ratio = float(keep_ratio)
        # Official ratio = fraction forgotten; round so it matches the conf file name.
        self.ratio = round(1.0 - self.keep_ratio, 4)
        conf = PREFIXKV_ROOT / "confs" / f"prefixkv_{model_name}_{self.ratio}.json"
        if not conf.is_file():
            raise FileNotFoundError(f"No official PrefixKV layer config for ratio={self.ratio}: {conf}")
        self.start_size = int(start_size)
        self.recent_size = int(recent_size)
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.stats_output_dir = stats_output_dir
        self.delayed_replay = _as_bool(delayed_replay)
        print(
            f"[prefixkv_llava15_orig] keep_ratio={self.keep_ratio} ratio={self.ratio} "
            f"delayed_replay={self.delayed_replay}",
            flush=True,
        )
        self._rank = 0
        self._world_size = 1

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
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for PrefixKVLlava15Orig.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation is not implemented.")

    def _step(self, cache, num_of_token, **inputs):
        """One forward + official PrefixKV trim, as in cache_generate.greedy_search."""
        if "past_key_values" in inputs:
            # Text-only step: go through the decoder directly so a logical
            # position can be passed (the official LLaVA forward has no
            # position_ids argument).
            out = self._model.model(
                input_ids=inputs["input_ids"],
                past_key_values=inputs["past_key_values"],
                position_ids=torch.full((1, 1), num_of_token, dtype=torch.long, device=self._device),
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            logits = self._model.lm_head(out.last_hidden_state)
            past, attentions = out.past_key_values, out.attentions
        else:
            out = self._model(**inputs, use_cache=True, output_attentions=True, return_dict=True)
            logits, past, attentions = out.logits, out.past_key_values, out.attentions
        num_of_token += logits.shape[1]
        past_key_values = cache(past, num_of_token, attentions)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return past_key_values, next_token, num_of_token

    @torch.inference_mode()
    def _generate_one(self, prompt: str, image, max_new_tokens: int) -> str:
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)
        image_tensor = process_images([image], self._image_processor, self._config)
        image_tensor = image_tensor.to(self._device, dtype=torch.bfloat16)

        cache = PrefixKV(
            model_name=self.model_name,
            start_size=self.start_size,
            recent_size=self.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            ratio=self.ratio,
            layer_num=self._config.num_hidden_layers,
            profile=False,
        )

        # PrefixKV reads confs/ relative to the cwd.
        cwd = os.getcwd()
        os.chdir(PREFIXKV_ROOT)
        try:
            prefill_ids = input_ids[:, :-1] if self.delayed_replay else input_ids
            past_key_values, next_token, num_of_token = self._step(
                cache, 0, input_ids=prefill_ids, images=image_tensor
            )
            if self.delayed_replay:
                past_key_values, next_token, num_of_token = self._step(
                    cache, num_of_token, input_ids=input_ids[:, -1:], past_key_values=past_key_values
                )
        finally:
            os.chdir(cwd)

        generated = []
        stop_token_ids = {self._tokenizer.eos_token_id}
        for generated_idx in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in stop_token_ids:
                break
            generated.append(token_id)
            if generated_idx + 1 == max_new_tokens:
                break
            past_key_values, next_token, num_of_token = self._step(
                cache, num_of_token, input_ids=next_token, past_key_values=past_key_values
            )

        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        conv = conv_templates[self.conv_template]
        stop_string = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_string and stop_string in text:
            text = text.split(stop_string, 1)[0].strip()
        return text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="PrefixKV(orig) LLaVA-1.5 Responding")
        for context, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = _flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            question = context
            if visuals and DEFAULT_IMAGE_TOKEN not in question:
                question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            gen_kwargs = dict(gen_kwargs)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", self.max_new_tokens))
            until = gen_kwargs.pop("until", None)

            image = visuals[0] if visuals else None
            text = self._generate_one(prompt, image, max_new_tokens)
            if until:
                stops = [until] if isinstance(until, str) else until
                for stop_string in stops:
                    if stop_string and stop_string in text:
                        text = text.split(stop_string, 1)[0]

            res.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            pbar.update(1)
        pbar.close()
        return res
