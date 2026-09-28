

"""lmms-eval model wrapper: LLaVA-OneVision + student-driven KV pruning.

Registered as `lmms_onevision_student`. The launcher (run_lmms_eval.py)
injects this module into the lmms-eval models registry before running.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kvpress.presses.visual_utility_student_onevision import VisualUtilityStudentOneVision
from .kv_decode_utils import greedy_decode_with_kv, trim_kv_cache_per_layer
from .text_kv_eviction import TextKVConfig, greedy_decode_with_text_eviction

try:
    from lmms_eval import utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
except ImportError as e:
    raise ImportError("lmms-eval not installed. `pip install lmms-eval==0.2.4`") from e

DEFAULT_IMAGE_TOKEN = "<image>"
LLAVA_IMAGE_TOKEN_INDEX = -200  # qvik.llava_onevision.constants.IMAGE_TOKEN_INDEX


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _kv_cache_nbytes(past_kv) -> int:
    """Return the bytes held by key/value tensors in a Transformers cache."""
    tensors: list[torch.Tensor] = []
    if hasattr(past_kv, "key_cache"):
        tensors.extend(past_kv.key_cache)
        tensors.extend(past_kv.value_cache)
    elif hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                tensors.append(keys)
            if isinstance(values, torch.Tensor):
                tensors.append(values)
    else:
        for keys, values in past_kv:
            tensors.extend((keys, values))

    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return int(total)


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


@register_model("lmms_onevision_student")
class LmmsOnevisionStudent(lmms):
    """LLaVA-OneVision with student-scored image-token KV pruning (lmms-eval wrapper)."""

    def __init__(
        self,
        pretrained: str = "/workspace/zap/model/llava-onevision-qwen2-7b-ov",
        student_path: str = "/workspace/zap/ckpts/student_onevision_A_ep20",
        keep_ratio: float = 0.5,
        device: str = "cuda:0",
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        stats_output_dir: str = "",
        conv_template: str = "qwen_1_5",
        text_eviction_mode: str = "none",
        text_keep_ratio: float = 0.2,
        text_cache_size: int = 0,
        h2o_recent_ratio: float = 0.5,
        streaming_sink_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        assert self.batch_size_per_gpu == 1, "Only batch_size=1 is supported."

        self._init_llava(pretrained, device, attn_implementation, conv_template)

        self._config = self._model.config

        self.student = VisualUtilityStudentOneVision.from_pretrained(student_path)
        self.student = self.student.to(device=self._device, dtype=torch.float16).eval()
        self.keep_ratio = float(keep_ratio)
        self.stats_output_dir = stats_output_dir
        self.text_kv_config = TextKVConfig(
            mode=text_eviction_mode,
            keep_ratio=float(text_keep_ratio),
            cache_size=int(text_cache_size),
            h2o_recent_ratio=float(h2o_recent_ratio),
            streaming_sink_size=int(streaming_sink_size),
        ).normalized()
        self._reported_keep_budget = False
        self._img_keep_sum = 0
        self._img_total_sum = 0
        self._img_sample_count = 0
        # per-sample stats: list of {total_keep_ratio, image_keep_ratio, n_image_original, n_image_kept, prompt_len}
        self._keep_stats: list[dict] = []
        self._rank = 0
        self._world_size = 1

    def _init_llava(self, pretrained: str, device: str, attn_implementation: str, conv_template: str) -> None:
        """Load LLaVA-format checkpoint (llava-onevision-qwen2-7b-ov)."""
        from qvik.llava_onevision.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from qvik.llava_onevision.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained, None, model_name,
            device_map=device,
            attn_implementation=attn_implementation,
            multimodal=True,
        )
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._processor = None
        self._conv_template = conv_template
        self._llava_process_images = process_images
        self._llava_tokenizer_image_token = tokenizer_image_token

    # --- lmms interface properties ---

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
        return getattr(self._config, "max_position_embeddings", 32768)

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
        return self._tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood not implemented for LmmsOnevisionStudent")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation not implemented for LmmsOnevisionStudent")

    # --- main generation ---

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
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = [img for sublist in visuals for img in sublist]  # flatten

            gen_kwargs = all_gen_kwargs[0]
            gen_kwargs.pop("until", None)
            max_new_tokens = gen_kwargs.get("max_new_tokens", 32)

            context = contexts[0]

            output = self._generate_llava(context, visuals, max_new_tokens)
            res.append(output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        self._save_keep_stats(task_name)
        return res

    def _save_keep_stats(self, task_name: Optional[str]) -> None:
        if not self._keep_stats:
            return
        n = len(self._keep_stats)
        summary = {
            "task": task_name,
            "keep_ratio": self.keep_ratio,
            "text_eviction_mode": self.text_kv_config.mode,
            "text_keep_ratio": self.text_kv_config.keep_ratio,
            "text_cache_size": self.text_kv_config.cache_size,
            "h2o_recent_ratio": self.text_kv_config.h2o_recent_ratio,
            "streaming_sink_size": self.text_kv_config.streaming_sink_size,
            "n_samples": n,
            "avg_image_token_ratio": sum(s["image_token_ratio"] for s in self._keep_stats) / n,
            "avg_text_token_ratio": sum(s["text_token_ratio"] for s in self._keep_stats) / n,
            "avg_total_keep_ratio": sum(s["total_keep_ratio"] for s in self._keep_stats) / n,
            "avg_image_keep_ratio": sum(s["image_keep_ratio"] for s in self._keep_stats) / n,
            "avg_n_image_original": sum(s["n_image_original"] for s in self._keep_stats) / n,
            "avg_n_image_kept": sum(s["n_image_kept"] for s in self._keep_stats) / n,
            "samples": self._keep_stats,
        }
        average_fields = {
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
        measured = [
            sample for sample in self._keep_stats
            if "decode_seconds" in sample
        ]
        if measured:
            runtime_average_fields = (
                "prefill_seconds",
                "decode_seconds",
                "decode_steps",
                "num_generated_tokens",
                "decode_ms_per_step",
                "end_to_end_seconds",
                "prefill_peak_allocated_gib",
                "decode_start_allocated_gib",
                "decode_peak_allocated_gib",
                "decode_peak_increment_gib",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "kv_cache_full_prompt_gib",
                "kv_cache_prompt_gib",
                "kv_cache_final_gib",
                "kv_cache_full_final_estimated_gib",
                "kv_cache_prompt_keep_ratio",
                "kv_cache_final_keep_ratio_estimated",
            )
            for key in runtime_average_fields:
                summary[f"avg_{key}"] = sum(
                    float(sample.get(key, 0.0)) for sample in measured
                ) / len(measured)

            total_decode_seconds = sum(
                float(sample["decode_seconds"]) for sample in measured
            )
            total_decode_steps = sum(
                int(sample["decode_steps"]) for sample in measured
            )
            full_prompt_bytes = sum(
                int(sample["kv_cache_full_prompt_bytes"]) for sample in measured
            )
            kept_prompt_bytes = sum(
                int(sample["kv_cache_prompt_bytes"]) for sample in measured
            )
            full_final_bytes = sum(
                int(sample["kv_cache_full_final_estimated_bytes"])
                for sample in measured
            )
            kept_final_bytes = sum(
                int(sample["kv_cache_final_bytes"]) for sample in measured
            )
            summary.update({
                "n_measured_samples": len(measured),
                "total_decode_seconds": total_decode_seconds,
                "total_decode_steps": total_decode_steps,
                "decode_ms_per_step_weighted": (
                    total_decode_seconds * 1000.0 / max(1, total_decode_steps)
                ),
                "kv_cache_prompt_keep_ratio_weighted": (
                    kept_prompt_bytes / max(1, full_prompt_bytes)
                ),
                "kv_cache_prompt_reduction_ratio_weighted": (
                    1.0 - kept_prompt_bytes / max(1, full_prompt_bytes)
                ),
                "kv_cache_final_keep_ratio_weighted_estimated": (
                    kept_final_bytes / max(1, full_final_bytes)
                ),
                "kv_cache_final_reduction_ratio_weighted_estimated": (
                    1.0 - kept_final_bytes / max(1, full_final_bytes)
                ),
            })
        out_dir = self.stats_output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{task_name or 'unknown'}_keep_ratio_stats.json"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[keep-ratio] stats saved → {fpath} "
            f"(avg_total={summary['avg_total_keep_ratio']:.4f} avg_image={summary['avg_image_keep_ratio']:.4f})",
            file=sys.stderr, flush=True,
        )

    @torch.no_grad()
    def _generate_llava(self, context: str, visuals, max_new_tokens: int) -> str:
        """Generation path for LLaVA-format checkpoint (llava-onevision-qwen2-7b-ov)."""
        from qvik.llava_onevision.conversation import conv_templates
        from qvik.llava_onevision.constants import (
            DEFAULT_IMAGE_TOKEN as LLAVA_DEFAULT_IMAGE_TOKEN,
            IGNORE_INDEX,
            IMAGE_TOKEN_INDEX,
        )

        measure_cuda = self._device.type == "cuda" and torch.cuda.is_available()
        _cuda_sync(self._device)
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        sample_start = time.perf_counter()

        # Build prompt via conv template
        conv = conv_templates[self._conv_template].copy()
        question = context
        if LLAVA_DEFAULT_IMAGE_TOKEN not in question and visuals:
            image_tokens = " ".join([LLAVA_DEFAULT_IMAGE_TOKEN] * len(visuals))
            question = f"{image_tokens}\n{question}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = self._llava_tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)
        attention_mask = input_ids.ne(
            self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        ).to(self._device)

        if not visuals:
            # Fallback: standard generate
            image_tensor = self._llava_process_images(visuals, self._image_processor, self._config) if visuals else None
            if image_tensor is not None:
                if isinstance(image_tensor, list):
                    image_tensor = [t.to(self._device, dtype=torch.float16) for t in image_tensor]
                else:
                    image_tensor = image_tensor.to(self._device, dtype=torch.float16)
            image_sizes = [img.size for img in visuals] if visuals else None
            try:
                out = self._model.generate(
                    input_ids, attention_mask=attention_mask,
                    images=image_tensor, image_sizes=image_sizes,
                    max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                return self._tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"[lmms-onevision-student] LLAVA fallback failed ({e})", file=sys.stderr, flush=True)
                return ""

        # Process images
        image_tensor = self._llava_process_images(visuals, self._image_processor, self._config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self._device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self._device, dtype=torch.float16)
        image_sizes = [img.size for img in visuals]

        # Expand image tokens via prepare_inputs_labels_for_multimodal
        try:
            # Supplying labels gives us an exact expanded visual-token mask:
            # multimodal preparation replaces every image feature label with
            # IGNORE_INDEX while preserving labels for all text positions.
            _, _, new_attn_mask, _, inputs_embeds, expanded_labels = self._model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, input_ids.clone(),
                image_tensor, ["image"], image_sizes,
            )
        except Exception as e:
            print(f"[lmms-onevision-student] prepare_inputs_labels failed ({e}), skipping.", file=sys.stderr, flush=True)
            return ""

        # Compute the exact visual positions after variable-length, multi-image
        # feature expansion.
        input_ids_1d = input_ids[0][attention_mask[0].bool()]
        n_img_placeholders = int((input_ids_1d == IMAGE_TOKEN_INDEX).sum().item())
        prompt_len = int(inputs_embeds.shape[1])
        if expanded_labels is None:
            raise RuntimeError("Expanded labels are required to locate OneVision image KVs.")
        image_positions = (
            (expanded_labels[0] == IGNORE_INDEX)
            .nonzero(as_tuple=False)
            .flatten()
            .detach()
            .cpu()
        )
        n_img = int(image_positions.numel())
        n_text = prompt_len - n_img
        if n_img_placeholders != len(visuals) or n_img == 0:
            raise RuntimeError(
                "OneVision image expansion mismatch: "
                f"placeholders={n_img_placeholders} images={len(visuals)} "
                f"expanded_visual_tokens={n_img}."
            )
        n_keep = max(1, int(math.ceil(n_img * self.keep_ratio)))

        self._img_keep_sum += n_keep
        self._img_total_sum += n_img
        self._img_sample_count += 1
        self._keep_stats.append({
            "n_image_original": n_img,
            "n_image_kept": n_keep,
            "n_text": n_text,
            "n_text_prompt_original": n_text,
            "avg_n_text_prompt_kept_after_eviction": float(n_text),
            "text_prompt_keep_ratio_after_eviction": 1.0,
            "avg_n_text_cache_final": float(n_text),
            "avg_n_visual_cache_final": float(n_keep),
            "text_eviction_events": 0,
            "prompt_len": prompt_len,
            "image_token_ratio": n_img / max(1, prompt_len),
            "text_token_ratio": n_text / max(1, prompt_len),
            "total_keep_ratio": (n_text + n_keep) / max(1, prompt_len),
            "image_keep_ratio": n_keep / max(1, n_img),
        })
        if not self._reported_keep_budget:
            print(
                f"[lmms-onevision-student] LLAVA keep_ratio_basis=image keep_ratio={self.keep_ratio} "
                f"prompt_len={prompt_len} image_tokens={n_img} text_tokens={n_text} "
                f"image_tokens_kept={n_keep}",
                file=sys.stderr, flush=True,
            )
            self._reported_keep_budget = True

        last_img = int(image_positions.max().item())
        q_positions = (
            torch.arange(last_img + 1, prompt_len, dtype=torch.long)
            if last_img + 1 < prompt_len else torch.empty(0, dtype=torch.long)
        )
        image_idx_dev = image_positions.to(self._device)
        q_idx_dev = q_positions.to(self._device)

        # Score each layer as soon as its hidden state is produced. Asking
        # Qwen2 for output_hidden_states retains all 29 [T,H] tensors on GPU;
        # the longest MileBench prompts cannot fit those together with the KV
        # cache. Hooks retain only the small per-image-token student scores.
        layer_scores: dict[int, torch.Tensor] = {}
        hook_handles = []
        num_layers = len(self._model.model.layers)
        needs_visual_pruning = n_keep < n_img

        def _capture_layer(layer_idx: int):
            def _hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                layer_scores[layer_idx] = self.student.layers[str(layer_idx)](
                    hidden, image_idx_dev, q_idx_dev
                ).squeeze(0).detach()
            return _hook

        if needs_visual_pruning:
            for li in self.student.layer_indices:
                source = (
                    self._model.model.norm
                    if li == num_layers - 1
                    else self._model.model.layers[li]
                )
                hook_handles.append(source.register_forward_hook(_capture_layer(li)))

        # Prefill through the Qwen2 backbone. Calling the CausalLM wrapper here
        # materializes [prompt_len, vocab_size] logits even though only the last
        # token is used, which OOMs on MileBench's longest prompts.
        _cuda_sync(self._device)
        prefill_start = time.perf_counter()
        try:
            prefill = self._model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attn_mask,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        except Exception as e:
            print(f"[lmms-onevision-student] prefill failed ({e})", file=sys.stderr, flush=True)
            return ""
        finally:
            for handle in hook_handles:
                handle.remove()

        past_kv = prefill.past_key_values
        last_logits = self._model.lm_head(prefill.last_hidden_state[:, -1:, :])
        next_token = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        kv_cache_full_prompt_bytes = _kv_cache_nbytes(past_kv)
        eos_token_id = int(
            self._tokenizer.eos_token_id
            if self._tokenizer.eos_token_id is not None
            else 151645
        )

        keep_masks: dict[int, torch.Tensor] = {}
        if needs_visual_pruning:
            for li in self.student.layer_indices:
                if li not in layer_scores:
                    raise RuntimeError(f"Missing student score for language layer {li}.")
                scores = layer_scores[li]
                top = torch.topk(scores, k=n_keep, largest=True).indices
                mask = torch.ones(prompt_len, dtype=torch.bool)
                image_keep = torch.zeros(n_img, dtype=torch.bool)
                image_keep[top.cpu()] = True
                mask[image_positions.cpu()] = image_keep
                keep_masks[li] = mask

        del layer_scores, prefill, last_logits, inputs_embeds, image_tensor
        past_kv = trim_kv_cache_per_layer(past_kv, keep_masks)
        kv_cache_prompt_bytes = _kv_cache_nbytes(past_kv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cuda_sync(self._device)
        prefill_seconds = time.perf_counter() - prefill_start

        prefill_peak_allocated = (
            torch.cuda.max_memory_allocated(self._device) if measure_cuda else 0
        )
        prefill_peak_reserved = (
            torch.cuda.max_memory_reserved(self._device) if measure_cuda else 0
        )
        decode_start_allocated = (
            torch.cuda.memory_allocated(self._device) if measure_cuda else 0
        )
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
        _cuda_sync(self._device)
        decode_start = time.perf_counter()

        if self.text_kv_config.mode == "none":
            answer_ids = greedy_decode_with_kv(
                self._model, past_kv, next_token,
                prompt_len=prompt_len,
                eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
            )
        else:
            answer_ids, text_stats = greedy_decode_with_text_eviction(
                self._model,
                past_kv,
                next_token,
                prompt_len=prompt_len,
                image_positions=image_positions,
                visual_keep_masks=keep_masks,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                config=self.text_kv_config,
            )
            self._keep_stats[-1].update(text_stats)
        _cuda_sync(self._device)
        decode_seconds = time.perf_counter() - decode_start
        decode_peak_allocated = (
            torch.cuda.max_memory_allocated(self._device) if measure_cuda else 0
        )
        decode_peak_reserved = (
            torch.cuda.max_memory_reserved(self._device) if measure_cuda else 0
        )
        kv_cache_final_bytes = _kv_cache_nbytes(past_kv)
        num_generated_tokens = int(answer_ids.numel())
        decode_steps = max(0, num_generated_tokens - 1)
        kv_bytes_per_token = (
            kv_cache_full_prompt_bytes / max(1, prompt_len)
        )
        kv_cache_full_final_estimated_bytes = int(round(
            kv_cache_full_prompt_bytes + decode_steps * kv_bytes_per_token
        ))
        self._keep_stats[-1].update({
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_steps": decode_steps,
            "num_generated_tokens": num_generated_tokens,
            "decode_ms_per_step": (
                decode_seconds * 1000.0 / max(1, decode_steps)
            ),
            "end_to_end_seconds": time.perf_counter() - sample_start,
            "prefill_peak_allocated_gib": _bytes_to_gib(prefill_peak_allocated),
            "decode_start_allocated_gib": _bytes_to_gib(decode_start_allocated),
            "decode_peak_allocated_gib": _bytes_to_gib(decode_peak_allocated),
            "decode_peak_increment_gib": _bytes_to_gib(
                max(0, decode_peak_allocated - decode_start_allocated)
            ),
            "peak_allocated_gib": _bytes_to_gib(
                max(prefill_peak_allocated, decode_peak_allocated)
            ),
            "peak_reserved_gib": _bytes_to_gib(
                max(prefill_peak_reserved, decode_peak_reserved)
            ),
            "kv_cache_full_prompt_bytes": kv_cache_full_prompt_bytes,
            "kv_cache_prompt_bytes": kv_cache_prompt_bytes,
            "kv_cache_final_bytes": kv_cache_final_bytes,
            "kv_cache_full_final_estimated_bytes": (
                kv_cache_full_final_estimated_bytes
            ),
            "kv_cache_full_prompt_gib": _bytes_to_gib(
                kv_cache_full_prompt_bytes
            ),
            "kv_cache_prompt_gib": _bytes_to_gib(kv_cache_prompt_bytes),
            "kv_cache_final_gib": _bytes_to_gib(kv_cache_final_bytes),
            "kv_cache_full_final_estimated_gib": _bytes_to_gib(
                kv_cache_full_final_estimated_bytes
            ),
            "kv_cache_prompt_keep_ratio": (
                kv_cache_prompt_bytes / max(1, kv_cache_full_prompt_bytes)
            ),
            "kv_cache_final_keep_ratio_estimated": (
                kv_cache_final_bytes
                / max(1, kv_cache_full_final_estimated_bytes)
            ),
        })
        torch.cuda.empty_cache()
        return self._tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
