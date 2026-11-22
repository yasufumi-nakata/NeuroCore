import unittest
from summarize_report import binary_subset


class SummaryTests(unittest.TestCase):
    def test_AI_predictions_remain_errors_and_true_AI_row_does_not_enter_precision(self):
        out = binary_subset({"class_order": ["ai_prompt_submit", "keyboard", "mouse_click"],
                             "confusion_rows_true_columns_predicted": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        self.assertEqual(out["samples"], 39)
        self.assertEqual(out["excluded_true_AI_examples"], 6)
        self.assertEqual(out["AI_predictions_on_common_action_examples_counted_as_errors"], 11)
        self.assertAlmostEqual(out["accuracy"], 14 / 39)
        self.assertAlmostEqual(out["per_class"]["keyboard"]["precision"], 5 / 13)
        self.assertAlmostEqual(out["per_class"]["keyboard"]["recall"], 5 / 15)
        self.assertAlmostEqual(out["balanced_accuracy"], (5 / 15 + 9 / 24) / 2)

    def test_pure_AI_predictions_do_not_create_empty_or_perfect_common_subset(self):
        out = binary_subset({"class_order": ["ai_prompt_submit", "keyboard", "mouse_click"],
                             "confusion_rows_true_columns_predicted": [[1, 0, 0], [20, 0, 0], [30, 0, 0]]})
        self.assertEqual(out["samples"], 50)
        self.assertEqual(out["accuracy"], 0)
        self.assertEqual(out["balanced_accuracy"], 0)
        self.assertEqual(out["macro_f1"], 0)


if __name__ == "__main__":
    unittest.main()
