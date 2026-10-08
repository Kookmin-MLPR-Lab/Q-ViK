"""lmms-eval wrapper: LLaVA-1.5-7B + LOOK-M exactly as in the original repo
(look_rebuttal@ecf0f51, extracted to /workspace/nips/LOOK-M_orig).

kv_mode="text_prior_pivot_merge" (LOOK-M): at prefill each layer keeps
hh_ratio*L text-prior heavy hitters + recent_ratio*L recent tokens over the
WHOLE prompt (text and image), and pivot-merges every evicted KV into its most
similar kept KV. keep_ratio is split evenly: hh_ratio = recent_ratio = keep/2
(the repo's default 0.1/0.1 is keep=0.2).

Only change vs. the original: with delayed_replay=True the last prompt token is
held out, the rest is prefilled (eviction fires), and the held-out token is
replayed through the compressed cache, so the first answer token is produced
post-eviction. delayed_replay=False calls the original model.generate().
Based on look_rebuttal/lmms_models/lookm_llava15.py.
"""


from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

LLAVA_ROOT = Path("/workspace/nips/LOOK-M_orig/LLaVA-mix_merge_v1")
sys.path.insert(0, str(LLAVA_ROOT))

# LOOK-M_orig/.../llava_llama.py has one import-only fix applied: it imported
# MixMerLlamaAttention_drop (used only by an unused KV_DICT), which the same
# commit's modify_llama.py never defines. Nothing else in the tree is changed.

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402

from lmms_eval.api.instance import Instance  # noqa: E402
from lmms_eval.api.model import lmms  # noqa: E402
from lmms_eval.api.registry import register_model  # noqa: E402


def _flatten(items):
    return [child for item in items for child in item]


@register_model("lookm_llava15_orig")
class LookMLlava15Orig(lmms):
    """Original LOOK-M (``text_prior_pivot_merge``) over the whole prompt."""

    def __init__(
        self,
        pretrained: str = "/workspace/nips/models/llava-v1.5-7b",
        keep_ratio: float = 0.1,
        device: str = "cuda:0",
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        max_new_tokens: int = 32,
        stats_output_dir: str = "",
        delayed_replay: bool | str = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[lookm_llava15_orig] ignoring unexpected kwargs: {list(kwargs.keys())}")

        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError(
                "LOOK-M's kv_cache state lives on the attention module itself "
                "(not per-request); batch_size must be 1."
            )

        model_name = get_model_name_from_path(pretrained)
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            self._max_length,
        ) = load_pretrained_model(
            pretrained,
            None,
            model_name,
            device=str(self._device),
            device_map=None,
            kv_mode="text_prior_pivot_merge",
            hh_ratio=float(keep_ratio) / 2,
            recent_ratio=float(keep_ratio) / 2,
        )
        self._model.to(self._device)
        self._model.eval()
        self._config = self._model.config

        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.stats_output_dir = stats_output_dir
        if isinstance(delayed_replay, str):
            delayed_replay = delayed_replay.strip().lower() not in ("0", "false", "no")
        self.delayed_replay = bool(delayed_replay)
        print(f"[lookm_llava15_orig] keep_ratio={keep_ratio} hh=recent={float(keep_ratio)/2} delayed_replay={self.delayed_replay}", flush=True)
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
        raise NotImplementedError("loglikelihood is not implemented for LookMLlava15.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation is not implemented.")

    def _clean_kv_cache(self) -> None:
        for layer in self._model.model.layers:
            clean = getattr(layer.self_attn, "_clean_cache", None)
            if clean is not None:
                clean()

    def _set_image_token_spans(self, prompt_ids: torch.Tensor, expanded_seq_len: int) -> None:
        # Normally done inside LlavaLlamaForCausalLM.generate(); replicated
        # here since delayed replay bypasses generate() entirely.
        image_position = torch.where(prompt_ids[0] == IMAGE_TOKEN_INDEX)[0]
        for layer in self._model.model.layers:
            layer.self_attn.kv_cache.image_position = image_position

    @torch.inference_mode()
    def _generate_one_plain(self, prompt: str, image, max_new_tokens: int) -> str:
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)
        image_tensor = process_images([image], self._image_processor, self._config)
        image_tensor = image_tensor.to(self._device, dtype=torch.float16)

        conv = conv_templates[self.conv_template]
        stop_string = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = [KeywordsStoppingCriteria([stop_string], self._tokenizer, input_ids)]

        output_ids = self._model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        self._clean_kv_cache()

        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if stop_string and stop_string in text:
            text = text.split(stop_string, 1)[0].strip()
        return text

    @torch.inference_mode()
    def _generate_one_delayed_replay(self, prompt: str, image, max_new_tokens: int) -> str:
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)
        image_tensor = process_images([image], self._image_processor, self._config)
        image_tensor = image_tensor.to(self._device, dtype=torch.float16)

        if input_ids.shape[1] < 2:
            raise ValueError("Prompt must contain at least two tokens for delayed replay.")
        prefill_ids = input_ids[:, :-1]
        held_out_id = input_ids[:, -1:]

        _, position_ids, attention_mask, _, inputs_embeds, _ = self._model.prepare_inputs_labels_for_multimodal(
            prefill_ids, None, None, None, None, image_tensor, image_sizes=[image.size]
        )
        self._set_image_token_spans(prefill_ids, inputs_embeds.shape[1])

        outputs = self._model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        prefill_len = inputs_embeds.shape[1]
        past_key_values = outputs.past_key_values
        del outputs

        # Replay the held-out last prompt token through the now-compressed
        # cache so the first *generated* token is also produced under
        # eviction, not via the model's ordinary full-attention prefill.
        held_out_embed = self._model.get_model().embed_tokens(held_out_id)
        outputs = self._model(
            inputs_embeds=held_out_embed,
            past_key_values=past_key_values,
            position_ids=torch.full((1, 1), prefill_len, dtype=torch.long, device=self._device),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logical_prompt_length = prefill_len + 1
        del outputs

        generated = []
        stop_token_ids = {self._tokenizer.eos_token_id}
        if self._tokenizer.pad_token_id is not None:
            stop_token_ids.add(self._tokenizer.pad_token_id)

        for generated_idx in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id in stop_token_ids:
                break
            generated.append(token_id)
            if generated_idx + 1 == max_new_tokens:
                break

            next_embed = self._model.get_model().embed_tokens(next_token)
            outputs = self._model(
                inputs_embeds=next_embed,
                past_key_values=past_key_values,
                position_ids=torch.full(
                    (1, 1), logical_prompt_length + generated_idx, dtype=torch.long, device=self._device
                ),
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        self._clean_kv_cache()
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        conv = conv_templates[self.conv_template]
        stop_string = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_string and stop_string in text:
            text = text.split(stop_string, 1)[0].strip()
        return text

    def _generate_one(self, prompt: str, image, max_new_tokens: int) -> str:
        # The original worker (workers/model_workers.py) generates under
        # bfloat16 autocast; LOOK-M's -65516 score masks overflow without it.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.delayed_replay:
                return self._generate_one_delayed_replay(prompt, image, max_new_tokens)
            return self._generate_one_plain(prompt, image, max_new_tokens)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="LOOK-M LLaVA-1.5 Responding")
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
