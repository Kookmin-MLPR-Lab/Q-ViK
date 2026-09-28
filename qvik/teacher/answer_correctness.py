"""Dataset-aware answer normalization and correctness checks for teacher filtering."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence


_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*",
    flags=re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Normalize short VQA answers while preserving decimal points."""
    text = str(text).lower().strip().replace("\n", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    punctuation = string.punctuation.replace(".", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    words = []
    for word in text.split():
        word = _NUMBER_WORDS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def _prediction_variants(prediction: str) -> set[str]:
    first_line = next(
        (line.strip() for line in str(prediction).splitlines() if line.strip()),
        "",
    )
    stripped = _ANSWER_PREFIX.sub("", first_line).strip(" \t\"'`")
    variants = {
        normalize_answer(prediction),
        normalize_answer(first_line),
        normalize_answer(stripped),
    }
    return {variant for variant in variants if variant}


def _matches_short_answer(prediction: str, answers: Sequence[str]) -> bool:
    predicted = _prediction_variants(prediction)
    expected = {normalize_answer(answer) for answer in answers}
    expected.discard("")
    return bool(predicted & expected)


def _scienceqa_letter(prediction: str, choice_count: int) -> int | None:
    first_line = next(
        (line.strip() for line in str(prediction).splitlines() if line.strip()),
        "",
    )
    patterns = (
        r"^\s*\(?([a-z])\)?(?:[\s.:\-]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([a-z])\)?(?:[\s.:\-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            index = ord(match.group(1).lower()) - ord("a")
            return index if 0 <= index < choice_count else None
    return None


def is_correct_prediction(
    dataset: str,
    prediction: str,
    answers: Sequence[str],
    *,
    answer_index: int | None = None,
    choices: Sequence[str] = (),
) -> bool:
    """Return whether a generated answer is correct for a supported dataset."""
    dataset = dataset.lower()
    if dataset == "scienceqa":
        if answer_index is None or not 0 <= answer_index < len(choices):
            return False
        predicted_index = _scienceqa_letter(prediction, len(choices))
        if predicted_index is not None:
            return predicted_index == answer_index
        return _matches_short_answer(prediction, [choices[answer_index]])

    if dataset in {"gqa", "textvqa", "llava_instruct"}:
        return _matches_short_answer(prediction, answers)

    raise ValueError(f"Unsupported dataset for correctness filtering: {dataset}")
