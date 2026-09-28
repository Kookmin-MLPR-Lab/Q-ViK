from qvik.eval.mm_niah_scoring import score_mm_niah_answer


def test_text_answer_uses_vqa_word_containment() -> None:
    assert (
        score_mm_niah_answer(
            "retrieval-text",
            "The melody of the song is the heart.",
            "heart",
        )
        == 1.0
    )
    assert score_mm_niah_answer("reasoning-text", "A dessert.", "Dessert") == 1.0
    assert score_mm_niah_answer("retrieval-text", "heartfelt", "heart") == 0.0


def test_counting_answer_uses_json_list_soft_accuracy() -> None:
    assert score_mm_niah_answer("counting-text", "[1, 2]", "[1, 2]") == 1.0
    assert score_mm_niah_answer("counting-text", "[1, 9]", "[1, 2]") == 0.5
    assert (
        score_mm_niah_answer(
            "counting-text",
            '```json\n{"first": [1], "second": [2]}\n```',
            "[1, 2]",
        )
        == 1.0
    )
    assert score_mm_niah_answer("counting-text", "There are two.", "[2]") == 0.0
