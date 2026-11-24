import unittest
try:
    from .context_event_lookup import run, fit
except ImportError:
    from context_event_lookup import run, fit


def row(subject, split, context, target):
    return {"canonical_subject_id": subject, "split": split,
            "messages": [{"role": "user", "content": context},
                         {"role": "assistant", "content": target}]}


class EventLookupTest(unittest.TestCase):
    def test_train_only_context(self):
        train = [row("a", "train", "page-A", "keyboard"), row("a", "train", "page-A", "keyboard"),
                 row("b", "train", "page-A", "mouse_click"), row("b", "train", "page-B", "mouse_click"),
                 row("a", "train", "page-C", "ai_prompt_submit")]
        val = [row("c", "val", "page-A", "keyboard"), row("c", "val", "page-B", "mouse_click"),
               row("c", "val", "page-C", "ai_prompt_submit")]
        result = run(train, val)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["macro_balanced_accuracy"], 1.0)
        self.assertTrue(result["subject_disjoint"])

    def test_leakage_rejected(self):
        train = [row("same", "train", "x", "mouse_click")]
        val = [row("same", "val", "x", "mouse_click")]
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            run(train, val)

    def test_eeg_rejected(self):
        with self.assertRaisesRegex(ValueError, "not EEG"):
            fit([row("a", "train", "EEG_FIXED_Q16", "mouse_click")])

    def test_unknown_context_fallback_train_majority(self):
        train = [row("a", "train", "A", "keyboard"), row("a", "train", "A", "keyboard"),
                 row("a", "train", "B", "mouse_click")]
        val = [row("b", "val", "unseen", "keyboard"), row("b", "val", "A", "keyboard"),
               row("b", "val", "B", "mouse_click"), row("b", "val", "unseen", "ai_prompt_submit")]
        result = run(train, val)
        self.assertEqual(result["validation_unseen_contexts"], 1)
        self.assertEqual(result["correct"], 3)


if __name__ == "__main__":
    unittest.main()
