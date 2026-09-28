"""MM-NIAH-compatible answer scoring without the benchmark runtime deps."""

from __future__ import annotations

import json
import re
from itertools import chain
from typing import Any

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
_PERIOD_RE = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_RE = re.compile(r"(\d)(,)(\d)")


def _normalize_vqa_text(value: Any) -> str:
    """Apply the normalization used by MM-NIAH's bundled VQA scorer."""

    original = str(value).replace("\n", " ").replace("\t", " ").strip()
    normalized = original
    for punctuation in _PUNCTUATION:
        if (
            f"{punctuation} " in original
            or f" {punctuation}" in original
            or _COMMA_RE.search(original)
        ):
            normalized = normalized.replace(punctuation, "")
        else:
            normalized = normalized.replace(punctuation, " ")
    normalized = _PERIOD_RE.sub("", normalized)

    words = []
    for word in normalized.lower().split():
        mapped = _NUMBER_WORDS.get(word, word)
        if mapped not in _ARTICLES:
            words.append(str(mapped))
    return " ".join(words)


def _has_word(response: str, answer: str) -> bool:
    return re.search(r"\b" + re.escape(answer) + r"\b", response) is not None


def _score_vqa(response: str, answer: str | list[Any]) -> float:
    normalized_response = _normalize_vqa_text(response)
    answers = answer if isinstance(answer, list) else [answer]
    return float(
        any(
            _has_word(normalized_response, _normalize_vqa_text(candidate))
            for candidate in answers
        )
    )


def _parse_counting_response(response: str) -> list[Any] | tuple[Any, ...] | None:
    cleaned = response.replace("json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        values = list(parsed.values())
        if not all(isinstance(value, list) for value in values):
            return None
        parsed = list(chain.from_iterable(values))
    if not isinstance(parsed, (list, tuple)):
        return None
    return parsed


def score_mm_niah_answer(task: str, response: str, answer: Any) -> float:
    """Return the official MM-NIAH task score for one text-needle example."""

    if task == "counting-text":
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                return 0.0
        if not isinstance(answer, list) or not answer:
            return 0.0
        parsed = _parse_counting_response(response)
        if parsed is None:
            return 0.0
        matches = sum(predicted == expected for predicted, expected in zip(parsed, answer))
        return matches / len(answer)

    if isinstance(answer, int):
        stripped = response.strip(".")
        if stripped.isdigit():
            return float(int(stripped) == answer)
        return 0.0

    return _score_vqa(response, answer)
