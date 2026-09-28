from __future__ import annotations

import json

import pytest
import torch
from transformers import DynamicCache

from kvpress.presses.visual_utility_student import VisualUtilityStudent
from qvik.eval import delayed_replay as dr
from qvik.eval.text_kv_eviction import TextKVCacheManager, TextKVConfig


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(dr.ENV_VAR, raising=False)


def test_default_tasks_cover_short_answer_benchmarks() -> None:
    tasks = dr.parse_replay_tasks(None)
    for task in (
        "pope_offline",
        "mme_offline",
        "gqa_offline",
        "scienceqa_img_offline",
        "mmbench_en_dev_offline",
        "vqav2_test_s3_offline",
        "vqav2_testdev_offline",
    ):
        assert dr.should_replay(task, tasks), task
    for task in ("textvqa_offline", "chartqa_offline", "coco_cap_offline", "mmerealworld"):
        assert not dr.should_replay(task, tasks), task


def test_pipe_separator_env_override_and_none(monkeypatch) -> None:
    assert dr.parse_replay_tasks("pope|gqa") == {"pope", "gqa"}
    assert dr.parse_replay_tasks("none") == frozenset()
    monkeypatch.setenv(dr.ENV_VAR, "all")
    tasks = dr.parse_replay_tasks("pope")
    assert dr.should_replay("chartqa_offline", tasks)


def test_split_held_out_token() -> None:
    ids = torch.tensor([[1, 2, 3, 4]])
    prefill, held = dr.split_held_out_token(ids, True)
    assert prefill.tolist() == [[1, 2, 3]] and held.tolist() == [[4]]
    prefill, held = dr.split_held_out_token(ids, False)
    assert prefill is ids and held is None
    prefill, held = dr.split_held_out_token(ids[:, :1], True)
    assert held is None


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


def test_replay_uses_absolute_position_of_held_out_token() -> None:
    model = _FakeModel()
    past, nxt, new_len = dr.replay_held_out_token(model, 0, torch.tensor([[5]]), prompt_len=42)
    assert model.calls == [(5, 42, 42)]
    assert past == 1 and int(nxt.item()) == 7 and new_len == 43


def test_extended_masks_match_text_manager_layout_after_replay() -> None:
    # 3 text, 4 image, 2 text = prompt of 9 (last text token held out -> prefill of 9).
    prompt_len = 9
    image_positions = torch.arange(3, 7)
    mask = torch.ones(prompt_len, dtype=torch.bool)
    mask[torch.tensor([4, 6])] = False
    masks = {0: mask, 1: mask.clone()}
    ext = dr.extend_keep_masks_for_replay(masks)
    assert all(m.shape[0] == prompt_len + 1 and bool(m[-1]) for m in ext.values())
    kept = int(ext[0].sum())  # trimmed prompt + replayed token
    cache = DynamicCache()
    for layer_idx in range(2):
        cache.update(torch.zeros(1, 2, kept, 4), torch.zeros(1, 2, kept, 4), layer_idx)
    config = TextKVConfig(mode="streamingllm", cache_size=4)
    manager = TextKVCacheManager(
        cache,
        prompt_len=prompt_len + 1,
        image_positions=image_positions,
        visual_keep_masks=ext,
        config=config,
    )
    for layer_types in manager.token_types.values():
        assert layer_types.shape[-1] == kept

    # Without the extension the layout disagrees with the post-replay cache.
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        TextKVCacheManager(
            cache,
            prompt_len=prompt_len + 1,
            image_positions=image_positions,
            visual_keep_masks=masks,
            config=config,
        )


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
