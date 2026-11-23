
import unittest

from reward import compute_score, extract_answer


class TestChestICPOReward(unittest.TestCase):

    def score(self, response, answer="B"):
        return compute_score(
            data_source="chest_icpo",
            solution_str=response,
            ground_truth=answer,
        )

    def test_correct_plain_letter(self):
        self.assertEqual(self.score("B"), 1.0)

    def test_incorrect_plain_letter(self):
        self.assertEqual(self.score("A"), 0.0)

    def test_correct_structured(self):
        response = """
<think>
The pleural effusion improved.
</think>
<answer>B</answer>
<evidence>[T3, T4]</evidence>
<flag>ok</flag>
"""
        self.assertEqual(self.score(response), 1.0)

    def test_incorrect_structured(self):
        response = """
<think>
The effusion worsened.
</think>
<answer>A</answer>
"""
        self.assertEqual(self.score(response), 0.0)

    def test_reasoning_does_not_count(self):
        response = """
<think>
Option B might be correct.
</think>
<answer>A</answer>
"""
        self.assertEqual(self.score(response), 0.0)

    def test_missing_answer(self):
        self.assertEqual(
            self.score("The effusion improved."),
            0.0,
        )

    def test_empty_response(self):
        self.assertEqual(self.score(""), 0.0)

    def test_incomplete_reasoning(self):
        response = "<think>Option B looks correct"
        self.assertEqual(self.score(response), 0.0)

    def test_case_insensitive(self):
        self.assertEqual(self.score("<answer>b</answer>"), 1.0)

    def test_multiple_answer_tags(self):
        response = "<answer>A</answer><answer>B</answer>"
        self.assertEqual(self.score(response), 1.0)

    def test_extract_answer(self):
        self.assertEqual(
            extract_answer("<think>A</think><answer>B</answer>"),
            "B",
        )

    def test_invalid_ground_truth(self):
        with self.assertRaises(ValueError):
            self.score("B", answer="INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
