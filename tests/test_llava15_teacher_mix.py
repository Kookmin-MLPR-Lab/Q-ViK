import pytest
import torch

from qvik.teacher.extract_llava15 import _mix_teacher_signals


def test_mix_teacher_signals_uses_equal_normalized_contributions() -> None:
    question = torch.tensor([[90.0, 10.0]])
    answer = torch.tensor([[1.0, 9.0]])

    mixed, question_norm, answer_norm = _mix_teacher_signals(
        question,
        answer,
        question_weight=0.5,
        eps=1e-8,
    )

    torch.testing.assert_close(question_norm, torch.tensor([[0.9, 0.1]]))
    torch.testing.assert_close(answer_norm, torch.tensor([[0.1, 0.9]]))
    torch.testing.assert_close(mixed, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(mixed.sum(dim=-1), torch.ones(1))


def test_mix_teacher_signals_rejects_invalid_weight() -> None:
    teacher = torch.ones(1, 2)

    with pytest.raises(ValueError, match="question_weight"):
        _mix_teacher_signals(teacher, teacher, question_weight=1.1, eps=1e-8)
