import unittest

from qvik.teacher.answer_correctness import (
    is_correct_prediction,
    normalize_answer,
)


class AnswerCorrectnessTest(unittest.TestCase):
    def test_normalize_vqa_answer(self):
        self.assertEqual(normalize_answer("The TWO, apples."), "2 apples")
        self.assertEqual(normalize_answer("1,000.5"), "1000.5")

    def test_gqa_exact_normalized_match(self):
        self.assertTrue(
            is_correct_prediction("gqa", "The answer is: red.", ["red"])
        )
        self.assertFalse(is_correct_prediction("gqa", "blue", ["red"]))

    def test_textvqa_accepts_any_annotator_answer(self):
        self.assertTrue(
            is_correct_prediction(
                "textvqa",
                "NOKIA",
                ["nokia", "nokia", "toshiba"],
            )
        )

    def test_scienceqa_letter_or_choice_text(self):
        choices = ["condensation", "evaporation", "freezing"]
        kwargs = {"answer_index": 1, "choices": choices}
        self.assertTrue(
            is_correct_prediction("scienceqa", "(B)", ["evaporation"], **kwargs)
        )
        self.assertTrue(
            is_correct_prediction(
                "scienceqa",
                "The answer is B.",
                ["evaporation"],
                **kwargs,
            )
        )
        self.assertTrue(
            is_correct_prediction(
                "scienceqa",
                "evaporation",
                ["evaporation"],
                **kwargs,
            )
        )
        self.assertFalse(
            is_correct_prediction("scienceqa", "C", ["evaporation"], **kwargs)
        )


if __name__ == "__main__":
    unittest.main()
