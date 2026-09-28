from __future__ import annotations

import json

import pytest
import torch
from transformers import DynamicCache

from kvpress.presses.visual_utility_student import VisualUtilityStudent
from qvik.eval import prefill_mode as pm


def test_prefill_mode_values() -> None:
    assert pm.normalize_prefill_mode(None) == "qvik"
    assert pm.normalize_prefill_mode("ORIGIN") == "origin"
    with pytest.raises(ValueError):
        pm.normalize_prefill_mode("other")


def test_split_last_token() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    prefill, last = pm.split_last_token(ids, "qvik")
    assert prefill.tolist() == [[1, 2, 3]] and last.tolist() == [[4]]
    prefill, last = pm.split_last_token(ids, "origin")
    assert prefill is ids and last is None
    prefill, last = pm.split_last_token(ids[:, :1], "qvik")
    assert last is None


class _FakeOut:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, *, input_ids, past_key_values, cache_position, position_ids, **_):
        self.calls.append((int(input_ids.item()), int(cache_position.item()), int(position_ids.item())))
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 7] = 1.0
        return _FakeOut(logits, past_key_values + 1)


def test_last_token_uses_its_absolute_position() -> None:
    model = _FakeModel()
    past, nxt, new_len = pm.feed_last_token(model, 0, torch.tensor([[5]]), prompt_len=42)
    assert model.calls == [(5, 42, 42)]
    assert past == 1 and int(nxt.item()) == 7 and new_len == 43


def test_student_hidden_state_offset_roundtrip(tmp_path) -> None:
    st = VisualUtilityStudent(hidden_dim=8, conv_dim=4, proj_dim=4, mlp_dim=4, hidden_state_offset=0)
    st.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["hidden_state_offset"] == 0
    assert VisualUtilityStudent.from_pretrained(tmp_path).hidden_state_offset == 0
    hs = tuple(torch.full((1,), float(i)) for i in range(33))
    assert float(VisualUtilityStudent.from_pretrained(tmp_path).layer_input(hs, 5)) == 5.0

    # Legacy checkpoints without the field default to the output of layer l.
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg.pop("hidden_state_offset")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    legacy = VisualUtilityStudent.from_pretrained(tmp_path)
    assert legacy.hidden_state_offset == 1
    assert float(legacy.layer_input(hs, 5)) == 6.0
