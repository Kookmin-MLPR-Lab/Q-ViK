"""LLaVA-1.5 Q-ViK *oracle*: the student's scores are replaced by the teacher
signal computed on the test sample itself (upper bound for the student).

Teacher = qvik/teacher/extract_llava15.py::_collect_attention_two_pass on the
eval sample: greedy-decode the answer with the full cache, run one forward over
prompt + answer with output_attentions, and score every image token per layer
by the attention it receives from the answer tokens (mean over heads, then over
answer tokens). Everything else -- total/image keep budget, top-k per layer,
prefill_mode (delayed replay by default), decoding -- is LmmsLlava15Student's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

QVIK_ROOT = Path("/workspace/nips/Q-ViK")
sys.path.insert(0, str(QVIK_ROOT))

from qvik.eval.lmms_llava15_student import (  # noqa: E402
    LmmsLlava15Student,
    _infer_original_image_positions,
)
from lmms_eval.api.registry import register_model  # noqa: E402


class _OracleScorer:
    """Duck-types the VisualUtilityStudent API used by _generate_with_student."""

    def __init__(self, num_layers: int):
        self.layer_indices = list(range(num_layers))
        self.hidden_state_offset = 1
        self.teacher = None  # [L, n_img] for the current sample

    def layer_input(self, hidden_states, layer_idx):
        return None

    def forward_layer(self, layer_idx, H_l, image_idx, q_idx):
        return self.teacher[layer_idx].unsqueeze(0)


@register_model("lmms_llava15_oracle")
class LmmsLlava15Oracle(LmmsLlava15Student):
    def __init__(self, *args, student_path: str = "", **kwargs):
        super().__init__(*args, student_path="", **kwargs)
        self.student = _OracleScorer(self._config.num_hidden_layers)
        self._teacher_T = []
        print("[lmms-llava15-oracle] scores = on-sample teacher (answer->image attention)", file=sys.stderr, flush=True)

    @torch.no_grad()
    def _teacher_scores(self, input_ids, image_tensor, max_new_tokens):
        image_positions, prompt_len = _infer_original_image_positions(input_ids, self.image_feature_len)
        eos = self._tokenizer.eos_token_id
        gen = self._model.generate(
            inputs=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.long),
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self._tokenizer.pad_token_id or eos,
            eos_token_id=eos,
        )
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        # LLaVA generate() returns only new tokens when called with images.
        answer = seq[:, input_ids.shape[1]:] if seq.shape[1] > input_ids.shape[1] and torch.equal(seq[:, : input_ids.shape[1]], input_ids) else seq
        answer = answer[:, : max_new_tokens]
        T = int(answer.shape[1])
        full_ids = torch.cat([input_ids, answer.to(input_ids.device)], dim=1)
        out = self._model(input_ids=full_ids, images=image_tensor, output_attentions=True,
                          output_hidden_states=False, use_cache=False, return_dict=True)
        answer_pos = list(range(int(prompt_len), int(prompt_len) + T)) or [int(prompt_len) - 1]
        img_idx = image_positions.to(self._device)
        teacher = torch.stack([
            a[0][:, answer_pos, :].index_select(-1, img_idx).float().mean(0).mean(0)
            for a in out.attentions
        ])
        del out, gen
        self._teacher_T.append(T)
        return teacher.to(self._device)

    def _generate_with_student(self, *, input_ids, image_tensor, image_sizes, num_images, max_new_tokens):
        if image_tensor is not None and num_images > 0 and self.keep_ratio < 1.0:
            self.student.teacher = self._teacher_scores(input_ids, image_tensor, max_new_tokens)
        out = super()._generate_with_student(
            input_ids=input_ids, image_tensor=image_tensor, image_sizes=image_sizes,
            num_images=num_images, max_new_tokens=max_new_tokens,
        )
        self.student.teacher = None
        torch.cuda.empty_cache()
        return out
