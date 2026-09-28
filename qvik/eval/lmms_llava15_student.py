"""lmms-eval wrapper for original-repo LLaVA-1.5-7B + student KV pruning.

This intentionally does not use the Transformers LLaVA wrapper.
It loads the original LLaVA checkpoint layout through `llava.model.builder`,
matching the teacher extraction/training path used for the original labels.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

ZAP_ROOT = Path(os.environ.get("ZAP_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ZAP_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_ROOT))


def _patch_torch_load_legacy_bin_mmap() -> None:
    original_torch_load = torch.load

    def load(*args, **kwargs):
        retry_kwargs = dict(kwargs)
        while True:
            try:
                return original_torch_load(*args, **retry_kwargs)
            except RuntimeError as exc:
                if retry_kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs.pop("mmap", None)
                    continue
                raise
            except pickle.UnpicklingError:
                if retry_kwargs.get("weights_only") is True:
                    retry_kwargs["weights_only"] = False
                    retry_kwargs.pop("mmap", None)
                    continue
                raise

    torch.load = load


_patch_torch_load_legacy_bin_mmap()

from kvpress.presses.visual_utility_student import VisualUtilityStudent  # noqa: E402

from .kv_decode_utils import (  # noqa: E402
    SinkAbsorbPlan,
    greedy_decode_with_kv,
    trim_kv_cache_per_layer,
)
from .prefill_mode import (  # noqa: E402
    DEFAULT_PREFILL_MODE,
    extend_keep_masks,
    feed_last_token,
    normalize_prefill_mode,
    split_last_token,
)
from .text_kv_eviction import (  # noqa: E402
    TextKVConfig,
    greedy_decode_with_text_eviction,
)
from .visual_sink_utils import (  # noqa: E402
    VisualSinkDetection,
    detect_visual_sinks,
    topk_mask_with_optional_sink_filter,
)

from qvik.llava15.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from qvik.llava15.conversation import conv_templates
from qvik.llava15.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from qvik.llava15.model.builder import load_pretrained_model

try:
    from lmms_eval import utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - import error is environment-specific.
    raise ImportError("lmms-eval not installed. `pip install lmms-eval==0.2.4`") from exc


def _flatten(items):
    return [child for item in items for child in item]


def _resolve_eos_token_id(tokenizer, model_config) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        cfg = getattr(model_config, "eos_token_id", None)
        eos = cfg[0] if isinstance(cfg, (list, tuple)) and cfg else cfg
    return int(2 if eos is None else eos)


def _infer_original_image_positions(
    input_ids: torch.Tensor,
    image_feature_len: int,
) -> tuple[torch.Tensor, int]:
    """Map raw IMAGE_TOKEN_INDEX placeholders to multimodal prompt positions."""
    raw_ids = input_ids[0].detach().cpu().tolist()
    image_positions: list[int] = []
    cursor = 0
    for token_id in raw_ids:
        if int(token_id) == IMAGE_TOKEN_INDEX:
            image_positions.extend(range(cursor, cursor + image_feature_len))
            cursor += image_feature_len
        else:
            cursor += 1
    if not image_positions:
        raise ValueError("No image placeholders found in input_ids.")
    return torch.tensor(image_positions, dtype=torch.long), cursor


def _infer_raw_token_positions(
    input_ids: torch.Tensor,
    token_ids: list[int],
    image_feature_len: int,
) -> torch.Tensor:
    if not token_ids:
        return torch.empty(0, dtype=torch.long)
    raw_ids = input_ids[0].detach().cpu().tolist()
    targets = set(token_ids)
    positions: list[int] = []
    cursor = 0
    for token_id in raw_ids:
        if int(token_id) == IMAGE_TOKEN_INDEX:
            cursor += image_feature_len
            continue
        if int(token_id) in targets:
            positions.append(cursor)
        cursor += 1
    return torch.tensor(positions, dtype=torch.long)


def _decode_generated(
    tokenizer,
    sequences: torch.Tensor,
    input_len: int,
    max_new_tokens: int,
    prompt_ids: Optional[torch.Tensor] = None,
) -> str:
    ids = sequences[0].detach().cpu()
    # Original LLaVA generate() with inputs_embeds commonly returns only new
    # tokens; HF-style generate() may return prompt+new. Handle both.
    #
    # The old heuristic (`numel > max_new_tokens + 1`) misfires whenever
    # max_new_tokens exceeds the prompt length -- e.g. MMBench asks for 1024 new
    # tokens on a ~130-token prompt, so prompt+new never clears the threshold,
    # the prompt is left in, and decoding IMAGE_TOKEN_INDEX (-200) raises
    # "piece id is out of range". Prefer an exact prefix match on the prompt.
    if prompt_ids is not None:
        prompt = prompt_ids[0].detach().cpu()
        if ids.numel() > prompt.numel() and torch.equal(ids[: prompt.numel()], prompt):
            ids = ids[prompt.numel() :]
    elif ids.numel() > max_new_tokens + 1 and ids.numel() > input_len:
        ids = ids[input_len:]
    # Defence in depth: never hand the tokenizer an id it cannot map back.
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size:
        ids = ids[(ids >= 0) & (ids < vocab_size)]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()


@register_model("lmms_llava15_student")
class LmmsLlava15Student(lmms):
    """LLaVA-1.5-7B with student-scored image-token KV pruning (lmms-eval wrapper)."""

    def __init__(
        self,
        pretrained: str = str(ZAP_ROOT / "model/llava-v1.5-7b"),
        student_path: str = str(ZAP_ROOT / "ckpts/v1/student_llava15_orig_vflow_1800_lr1e4_e15"),
        vision_tower_path: str = "",
        keep_ratio: float = 0.5,
        keep_ratio_basis: str = "total",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        model_name: Optional[str] = None,
        conv_template: str = "vicuna_v1",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 32,
        image_feature_len: int = 576,
        grid_h: int = 24,
        grid_w: int = 24,
        stats_output_dir: str = "",
        sink_count: int = 0,
        sink_init: str = "bos",
        eviction_mode: str = "drop",
        sink_absorb_scale: float = 1.0,
        visual_sink_layer: int = 0,
        visual_sink_threshold: float = 8.0,
        visual_sink_max_ratio: float = 0.10,
        visual_sink_measure_mass: bool | str = False,
        text_eviction_mode: str = "none",
        text_keep_ratio: float = 0.2,
        text_cache_size: int = 0,
        h2o_recent_ratio: float = 0.5,
        streaming_sink_size: int = 4,
        prefill_mode: str = DEFAULT_PREFILL_MODE,
        hidden_state_offset: int | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[lmms-llava15-student] ignoring unexpected kwargs: {list(kwargs.keys())}", file=sys.stderr)

        self._device = torch.device(device)
        self.device_map = device_map
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            raise ValueError("Original LLaVA student wrapper only supports batch_size=1.")

        resolved_model_name = model_name or get_model_name_from_path(pretrained)
        (
            self._tokenizer,
            self._model,
            self._image_processor,
            self._max_length,
        ) = load_pretrained_model(
            pretrained,
            None,
            resolved_model_name,
            device_map=device_map,
        )

        self._model.eval()
        if device_map != "auto":
            self._model.to(self._device)
        try:
            self._model.tie_weights()
        except Exception:
            pass
        self._config = self._model.config

        self._model_dtype = next(self._model.parameters()).dtype
        if student_path:
            self.student = VisualUtilityStudent.from_pretrained(student_path)
            self.student = self.student.to(device=self._device, dtype=self._model_dtype).eval()
            if hidden_state_offset not in (None, ""):
                # Override for checkpoints whose training index is not recorded.
                self.student.hidden_state_offset = int(hidden_state_offset)
            print(
                f"[lmms-llava15-student] student={student_path} "
                f"hidden_state_offset={self.student.hidden_state_offset}",
                file=sys.stderr,
                flush=True,
            )
        else:
            self.student = None
        self.student_path = student_path
        self.keep_ratio = float(keep_ratio)
        self.keep_ratio_basis = str(keep_ratio_basis).strip().lower()
        if self.keep_ratio_basis not in {"total", "image"}:
            raise ValueError("keep_ratio_basis must be either 'total' or 'image'.")
        self.conv_template = conv_template
        self.max_new_tokens = int(max_new_tokens)
        self.image_feature_len = int(image_feature_len)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.stats_output_dir = stats_output_dir
        self.sink_count = int(sink_count)
        self.sink_init = sink_init
        self.sink_token_ids: list[int] = []
        self.eviction_mode = eviction_mode
        self.sink_absorb_scale = float(sink_absorb_scale)
        self.visual_sink_layer = int(visual_sink_layer)
        self.visual_sink_threshold = float(visual_sink_threshold)
        self.visual_sink_max_ratio = float(visual_sink_max_ratio)
        self.visual_sink_measure_mass = str(visual_sink_measure_mass).lower() in {"1", "true", "yes", "on"}
        self.text_kv_config = TextKVConfig(
            mode=text_eviction_mode,
            keep_ratio=float(text_keep_ratio),
            cache_size=int(text_cache_size),
            h2o_recent_ratio=float(h2o_recent_ratio),
            streaming_sink_size=int(streaming_sink_size),
        ).normalized()
        if self.sink_count > 0:
            self._attach_sink_tokens()
        # "qvik" (default): first answer token from the compressed cache;
        # "origin": first answer token from the full prefill (see prefill_mode.py).
        self.prefill_mode = normalize_prefill_mode(prefill_mode)
        print(f"[lmms-llava15-student] prefill_mode={self.prefill_mode}", file=sys.stderr, flush=True)
        self._img_keep_sum = 0
        self._img_total_sum = 0
        self._img_sample_count = 0
        self._keep_stats: list[dict] = []

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self._tokenizer.decode(tokens)
        except TypeError:
            return self._tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for Llava15OriginalStudent.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation is not implemented.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        task_name = None

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        self._keep_stats.clear()
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            if task_name is None:
                task_name = task

            batched_visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            flattened_visuals = _flatten(batched_visuals)

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs.pop("until", None)
            max_new_tokens = int(gen_kwargs.pop("max_new_tokens", self.max_new_tokens))

            prompt = self._build_prompt(contexts[0], batched_visuals[0])
            image_tensor, image_sizes = self._prepare_images(flattened_visuals)
            input_ids = tokenizer_image_token(
                prompt,
                self._tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self._device)
            input_ids = self._insert_sink_token_ids(input_ids)

            output = self._generate_with_student(
                input_ids=input_ids,
                image_tensor=image_tensor,
                image_sizes=image_sizes,
                num_images=len(flattened_visuals),
                max_new_tokens=max_new_tokens,
            )
            res.append(output)
            self.cache_hook.add_partial("generate_until", (contexts[0], gen_kwargs), output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        self._save_keep_stats(task_name)
        return res

    def _attach_sink_tokens(self) -> None:
        if self.sink_init != "bos":
            raise ValueError(f"Unsupported sink_init={self.sink_init!r}; supported: 'bos'.")
        tokens = [f"<sink_{idx}>" for idx in range(self.sink_count)]
        self._tokenizer.add_tokens(tokens, special_tokens=True)
        self._model.resize_token_embeddings(len(self._tokenizer))
        sink_ids = [int(self._tokenizer.convert_tokens_to_ids(token)) for token in tokens]
        bos_id = int(self._tokenizer.bos_token_id)
        input_embeddings = self._model.get_input_embeddings().weight
        output_embeddings = self._model.get_output_embeddings().weight
        with torch.no_grad():
            for token_id in sink_ids:
                input_embeddings[token_id].copy_(input_embeddings[bos_id])
                output_embeddings[token_id].copy_(output_embeddings[bos_id])
        self.sink_token_ids = sink_ids
        print(
            f"[lmms-llava15-student] attached {self.sink_count} BOS-init sink token(s): {tokens}",
            file=sys.stderr,
            flush=True,
        )

    def _insert_sink_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.sink_token_ids:
            return input_ids
        sink = torch.tensor(self.sink_token_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
        if input_ids.shape[1] > 0 and int(input_ids[0, 0].item()) == int(self._tokenizer.bos_token_id):
            return torch.cat([input_ids[:, :1], sink, input_ids[:, 1:]], dim=1)
        return torch.cat([sink, input_ids], dim=1)

    def _build_prompt(self, context: str, visuals: list) -> str:
        question = context
        if visuals and DEFAULT_IMAGE_TOKEN not in question:
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
            question = f"{image_tokens}\n{question}"

        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _prepare_images(self, visuals: list):
        if not visuals:
            return None, []
        image_sizes = [image.size for image in visuals]
        image_tensor = process_images(visuals, self._image_processor, self._config)
        if isinstance(image_tensor, list):
            image_tensor = [
                tensor.to(dtype=self._model_dtype, device=self._device) for tensor in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(dtype=self._model_dtype, device=self._device)
        return image_tensor, image_sizes

    def _save_keep_stats(self, task_name: Optional[str]) -> None:
        if not self._keep_stats:
            return
        n = len(self._keep_stats)
        summary = {
            "task": task_name,
            "keep_ratio": self.keep_ratio,
            "keep_ratio_basis": self.keep_ratio_basis,
            "eviction_mode": self.eviction_mode,
            "sink_count": self.sink_count,
            "visual_sink_layer": self.visual_sink_layer,
            "visual_sink_threshold": self.visual_sink_threshold,
            "visual_sink_measure_mass": self.visual_sink_measure_mass,
            "text_eviction_mode": self.text_kv_config.mode,
            "text_keep_ratio": self.text_kv_config.keep_ratio,
            "text_cache_size": self.text_kv_config.cache_size,
            "h2o_recent_ratio": self.text_kv_config.h2o_recent_ratio,
            "streaming_sink_size": self.text_kv_config.streaming_sink_size,
            "n_samples": n,
            "prefill_mode": self.prefill_mode,
            "avg_image_token_ratio": sum(s["image_token_ratio"] for s in self._keep_stats) / n,
            "avg_text_token_ratio": sum(s["text_token_ratio"] for s in self._keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in self._keep_stats) / n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in self._keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in self._keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in self._keep_stats) / n,
            "samples": self._keep_stats,
        }
        average_fields = {
            "n_visual_sink": "avg_n_visual_sink",
            "visual_sink_ratio": "avg_visual_sink_ratio",
            "visual_sink_score_mean": "avg_visual_sink_score_mean",
            "visual_sink_score_max": "avg_visual_sink_score_max",
            "visual_sink_keep_pollution": "avg_visual_sink_keep_pollution",
            "visual_sink_kept_ratio": "avg_visual_sink_kept_ratio",
            "visual_sink_mass_share": "avg_visual_sink_mass_share",
            "visual_total_mass_share": "avg_visual_total_mass_share",
            "visual_sink_mass_within_image": "avg_visual_sink_mass_within_image",
            "visual_non_sink_mass_share": "avg_visual_non_sink_mass_share",
            "text_cache_budget": "avg_text_cache_budget",
            "n_text_prompt_original": "avg_n_text_prompt_original",
            "avg_n_text_prompt_kept_after_eviction": "avg_n_text_prompt_kept_after_eviction",
            "text_prompt_keep_ratio_after_eviction": "avg_text_prompt_keep_ratio_after_eviction",
            "avg_n_text_cache_final": "avg_n_text_cache_final",
            "avg_n_visual_cache_final": "avg_n_visual_cache_final",
            "text_eviction_events": "avg_text_eviction_events",
        }
        for sample_key, summary_key in average_fields.items():
            summary[summary_key] = sum(
                float(sample.get(sample_key, 0.0)) for sample in self._keep_stats
            ) / n
        out_dir = self.stats_output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{task_name or 'unknown'}_keep_ratio_stats.json"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[lmms-llava15-student] stats saved -> {fpath} "
            f"(avg_total={summary['avg_total_keep_ratio']:.4f} "
            f"avg_image={summary['avg_image_keep_ratio']:.4f})",
            file=sys.stderr,
            flush=True,
        )

    @torch.no_grad()
    # The prefill below is a direct `self._model(...)` call, not HF `generate()`
    # (which carries its own @torch.no_grad()). Without this decorator autograd
    # retains every layer activation of the *uncompressed* prompt: measured 41.2
    # GiB vs 16.2 GiB for full-cache generate at 4608 image tokens, i.e. the
    # pruning path needed 2.5x the memory of the baseline it is meant to shrink,
    # and OOM'd on a 48 GiB A6000.
    @torch.no_grad()
    def _generate_with_student(
        self,
        *,
        input_ids: torch.Tensor,
        image_tensor,
        image_sizes: list,
        num_images: int,
        max_new_tokens: int,
    ) -> str:
        eos_token_id = _resolve_eos_token_id(self._tokenizer, self._model.config)
        pad_token_id = self._tokenizer.pad_token_id or eos_token_id
        modalities = ["image"] * max(1, num_images)

        def _safe_generate() -> str:
            out = self._model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long),
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            sequences = out.sequences if hasattr(out, "sequences") else out
            return _decode_generated(
                self._tokenizer,
                sequences,
                input_len=int(input_ids.shape[1]),
                max_new_tokens=max_new_tokens,
                prompt_ids=input_ids,
            )

        visual_sink_modes = {"visual_sink_stats", "visual_sink_drop", "visual_sink_aware_topk"}
        needs_prefill = image_tensor is not None and num_images > 0 and (
            self.eviction_mode in visual_sink_modes
            or self.keep_ratio < 1.0
            or self.text_kv_config.mode != "none"
        )
        needs_student = self.eviction_mode in {"drop", "sink_absorb", "visual_sink_aware_topk"}
        if not needs_prefill or (needs_student and self.student is None):
            return _safe_generate()

        # prefill_mode="qvik": prefill without the final prompt token, evict,
        # then feed that token through the compressed cache so the first
        # answer token sees only the kept visual KVs. "origin" prefills the
        # full prompt and takes the first token before eviction.
        prefill_ids, last_token_id = split_last_token(input_ids, self.prefill_mode)

        try:
            prefill = self._model(
                input_ids=prefill_ids,
                images=image_tensor,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=self.visual_sink_measure_mass,
                return_dict=True,
            )
        except Exception as exc:
            print(
                f"[lmms-llava15-student] WARNING: prefill failed ({exc}); falling back.",
                file=sys.stderr,
                flush=True,
            )
            return _safe_generate()

        H_all = prefill.hidden_states
        past_kv = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        try:
            image_positions, prompt_len = _infer_original_image_positions(
                prefill_ids,
                self.image_feature_len,
            )
        except ValueError:
            fallback_prompt_len = int(H_all[-1].shape[1])
            if last_token_id is not None:
                past_kv, next_token, fallback_prompt_len = feed_last_token(
                    self._model, past_kv, last_token_id, fallback_prompt_len
                )
            answer_ids = greedy_decode_with_kv(
                self._model,
                past_kv,
                next_token,
                prompt_len=fallback_prompt_len,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )
            return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

        actual_prompt_len = int(H_all[-1].shape[1])
        if actual_prompt_len != int(prompt_len):
            print(
                f"[lmms-llava15-student] WARNING: prompt_len mismatch "
                f"inferred={prompt_len} actual={actual_prompt_len}; using actual for decode.",
                file=sys.stderr,
                flush=True,
            )
            prompt_len = actual_prompt_len

        n_img = int(image_positions.numel())
        n_text = int(prompt_len) - n_img
        if self.keep_ratio_basis == "image":
            n_keep = max(1, math.ceil(n_img * self.keep_ratio))
        else:
            n_keep = max(1, round(n_img - (1.0 - self.keep_ratio) * int(prompt_len)))
        n_keep = min(n_keep, n_img)

        last_img = int(image_positions.max().item())
        q_positions = (
            torch.arange(last_img + 1, int(prompt_len), dtype=torch.long)
            if last_img + 1 < int(prompt_len)
            else torch.empty(0, dtype=torch.long)
        )
        image_idx_dev = image_positions.to(self._device)
        q_idx_dev = q_positions.to(self._device)
        visual_detection = detect_visual_sinks(
            H_all,
            image_positions,
            self.visual_sink_layer,
            self.visual_sink_threshold,
            self.visual_sink_max_ratio,
        )
        visual_sink_mass_stats = self._visual_sink_attention_stats(
            prefill.attentions,
            visual_detection,
            image_positions,
            q_positions,
        )
        # Last use of `prefill`. Drop it now: `past_kv` and `H_all` are only
        # aliases into it, so while it lives the *uncompressed* KV cache, all 33
        # hidden-state tensors and the full [1, L, vocab] logits stay reachable
        # -- through the trim and the entire decode loop. Measured ~3.8 GiB
        # retained at 4608 image tokens, which is why decode-phase memory was
        # almost independent of keep_ratio. Freeing here lets `del H_all` and the
        # `past_kv` rebind below actually release their tensors.
        del prefill

        keep_masks: dict[int, torch.Tensor] = {}
        drop_weights: dict[int, torch.Tensor] = {}
        visual_sink_pollution: list[float] = []
        visual_sink_kept_ratios: list[float] = []
        layer_indices = self._eviction_layer_indices(past_kv)
        if self.student is not None and self.eviction_mode != "visual_sink_drop":
            layer_indices = list(self.student.layer_indices)
        for layer_idx in layer_indices:
            if self.eviction_mode == "visual_sink_stats":
                continue
            if self.eviction_mode == "visual_sink_drop":
                image_keep = ~visual_detection.mask
                mask = torch.ones(int(prompt_len), dtype=torch.bool)
                mask[image_positions] = image_keep
                keep_masks[layer_idx] = mask
                weights = torch.zeros(int(prompt_len), dtype=torch.float32)
                dropped = image_positions[~image_keep]
                if dropped.numel() > 0:
                    weights[dropped] = 1.0 / float(dropped.numel())
                drop_weights[layer_idx] = weights
                continue
            # hidden_states[l + offset]; the offset is part of the checkpoint
            # config (default 1 = output of layer l, as in the zap v1 trainer)
            # so it always matches what the student was trained on.
            H_l = self.student.layer_input(H_all, layer_idx)
            scores = self.student.forward_layer(layer_idx, H_l, image_idx_dev, q_idx_dev).squeeze(0)
            if n_keep >= n_img:
                continue
            mask = torch.ones(int(prompt_len), dtype=torch.bool)
            image_keep = topk_mask_with_optional_sink_filter(
                scores,
                n_keep,
                visual_detection.mask,
                self.eviction_mode == "visual_sink_aware_topk",
            )
            if visual_detection.mask.numel() == image_keep.numel():
                visual_kept = int((image_keep & visual_detection.mask).sum().item())
                visual_total = max(1, visual_detection.count)
                visual_sink_pollution.append(visual_kept / max(1, int(image_keep.sum().item())))
                visual_sink_kept_ratios.append(visual_kept / visual_total)
            mask[image_positions] = image_keep
            keep_masks[layer_idx] = mask
            weights = torch.zeros(int(prompt_len), dtype=torch.float32)
            drop_image_positions = image_positions[~image_keep]
            if drop_image_positions.numel() > 0:
                weights[drop_image_positions] = torch.softmax(scores.detach().cpu()[~image_keep].float(), dim=0)
            drop_weights[layer_idx] = weights

        actual_n_keep = self._actual_image_keep_count(n_img, keep_masks, image_positions)
        self._record_keep_stats(
            n_img=n_img,
            n_text=n_text,
            prompt_len=int(prompt_len),
            n_keep=actual_n_keep,
            visual_detection=visual_detection,
            visual_sink_pollution=visual_sink_pollution,
            visual_sink_kept_ratios=visual_sink_kept_ratios,
            visual_sink_mass_stats=visual_sink_mass_stats,
        )
        del H_all
        absorb_plan = self._sink_absorb_plan(prefill_ids, keep_masks, drop_weights)
        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks, absorb_plan)
        if last_token_id is not None:
            past_kv, next_token, prompt_len = feed_last_token(
                self._model, past_kv, last_token_id, prompt_len
            )
            keep_masks = extend_keep_masks(keep_masks)
        if self.text_kv_config.mode == "none":
            answer_ids = greedy_decode_with_kv(
                self._model,
                past_kv,
                next_token,
                prompt_len=int(prompt_len),
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        else:
            answer_ids, text_stats = greedy_decode_with_text_eviction(
                self._model,
                past_kv,
                next_token,
                prompt_len=int(prompt_len),
                image_positions=image_positions,
                visual_keep_masks=keep_masks,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                config=self.text_kv_config,
            )
            self._keep_stats[-1].update(text_stats)
        torch.cuda.empty_cache()
        return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

    def _sink_absorb_plan(
        self,
        input_ids: torch.Tensor,
        keep_masks: dict[int, torch.Tensor],
        drop_weights: dict[int, torch.Tensor],
    ) -> SinkAbsorbPlan | None:
        if self.eviction_mode != "sink_absorb" or not self.sink_token_ids:
            return None
        sink_positions = _infer_raw_token_positions(
            input_ids,
            self.sink_token_ids,
            self.image_feature_len,
        )
        if sink_positions.numel() == 0:
            return None
        return SinkAbsorbPlan(
            sink_positions={layer_idx: sink_positions for layer_idx in keep_masks},
            drop_weights=drop_weights,
            value_scale=self.sink_absorb_scale,
        )

    def _eviction_layer_indices(self, past_kv) -> list[int]:
        if hasattr(past_kv, "key_cache"):
            return list(range(len(past_kv.key_cache)))
        if hasattr(past_kv, "layers"):
            return list(range(len(past_kv.layers)))
        return list(range(len(past_kv)))

    def _actual_image_keep_count(
        self,
        n_img: int,
        keep_masks: dict[int, torch.Tensor],
        image_positions: torch.Tensor,
    ) -> int:
        if not keep_masks:
            return n_img
        first_mask = next(iter(keep_masks.values()))
        return int(first_mask[image_positions].sum().item())

    def _visual_sink_attention_stats(
        self,
        attentions,
        visual_detection: VisualSinkDetection,
        image_positions: torch.Tensor,
        q_positions: torch.Tensor,
    ) -> dict[str, float]:
        empty = {
            "visual_sink_mass_share": 0.0,
            "visual_total_mass_share": 0.0,
            "visual_sink_mass_within_image": 0.0,
            "visual_non_sink_mass_share": 0.0,
        }
        if not self.visual_sink_measure_mass or attentions is None or len(attentions) == 0:
            return empty
        if image_positions.numel() == 0 or q_positions.numel() == 0:
            return empty
        attention_idx = min(max(int(visual_detection.layer_index) - 1, 0), len(attentions) - 1)
        attn = attentions[attention_idx]
        if attn is None:
            return empty
        attn = attn.detach().float()[0]
        q_idx = q_positions.to(attn.device)
        image_idx = image_positions.to(attn.device)
        query_attn = attn.index_select(dim=1, index=q_idx)
        visual_mass = query_attn.index_select(dim=2, index=image_idx).sum(dim=-1).mean().item()
        sink_mass = 0.0
        if visual_detection.count > 0:
            sink_positions = image_positions[visual_detection.mask].to(attn.device)
            sink_mass = query_attn.index_select(dim=2, index=sink_positions).sum(dim=-1).mean().item()
        return {
            "visual_sink_mass_share": float(sink_mass),
            "visual_total_mass_share": float(visual_mass),
            "visual_sink_mass_within_image": float(sink_mass / visual_mass) if visual_mass > 0 else 0.0,
            "visual_non_sink_mass_share": float(max(0.0, visual_mass - sink_mass)),
        }

    def _record_keep_stats(
        self,
        *,
        n_img: int,
        n_text: int,
        prompt_len: int,
        n_keep: int,
        visual_detection: VisualSinkDetection,
        visual_sink_pollution: list[float],
        visual_sink_kept_ratios: list[float],
        visual_sink_mass_stats: dict[str, float],
    ) -> None:
        self._img_keep_sum += n_keep
        self._img_total_sum += n_img
        self._img_sample_count += 1
        visual_scores = visual_detection.scores
        visual_mask = visual_detection.mask
        selected_scores = visual_scores[visual_mask] if visual_detection.count > 0 else visual_scores[:0]
        self._keep_stats.append(
            {
                "n_image_original": n_img,
                "n_image_kept": n_keep,
                "n_text": n_text,
                "prompt_len": prompt_len,
                "image_token_ratio": n_img / max(1, prompt_len),
                "text_token_ratio": n_text / max(1, prompt_len),
                "total_keep_ratio": (n_text + n_keep) / max(1, prompt_len),
                "image_keep_ratio": n_keep / max(1, n_img),
                "n_visual_sink": visual_detection.count,
                "visual_sink_ratio": visual_detection.ratio,
                "visual_sink_layer_resolved": visual_detection.layer_index,
                "visual_sink_score_mean": float(selected_scores.mean().item()) if selected_scores.numel() else 0.0,
                "visual_sink_score_max": float(visual_scores.max().item()) if visual_scores.numel() else 0.0,
                "visual_sink_keep_pollution": (
                    sum(visual_sink_pollution) / len(visual_sink_pollution) if visual_sink_pollution else 0.0
                ),
                "visual_sink_kept_ratio": (
                    sum(visual_sink_kept_ratios) / len(visual_sink_kept_ratios) if visual_sink_kept_ratios else 0.0
                ),
                **visual_sink_mass_stats,
            }
        )
        if not self._reported_keep_budget:
            print(
                f"[lmms-llava15-student] mode={self.eviction_mode} "
                f"keep_ratio_basis={self.keep_ratio_basis} "
                f"keep_ratio={self.keep_ratio} prompt_len={prompt_len} image_tokens={n_img} "
                f"text_tokens={n_text} image_tokens_kept={n_keep} "
                f"visual_sinks={visual_detection.count}",
                file=sys.stderr,
                flush=True,
            )
            self._reported_keep_budget = True
        if self._img_sample_count % 200 == 0:
            avg_img_ratio = self._img_keep_sum / max(1, self._img_total_sum)
            print(
                f"[lmms-llava15-student] img_keep_ratio_avg={avg_img_ratio:.4f} "
                f"({self._img_keep_sum}/{self._img_total_sum}) "
                f"over {self._img_sample_count} samples",
                file=sys.stderr,
                flush=True,
            )
