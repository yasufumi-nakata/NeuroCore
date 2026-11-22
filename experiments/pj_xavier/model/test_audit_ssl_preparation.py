import copy
import unittest

import audit_ssl_preparation as audit


def fixture():
    digest = "a" * 64
    row = {"waveform_sha256": digest, "recording_id": "recording", "ssl_split": "train", "participant_group": "group",
           "stored_signal_seconds": 70., "eeg_channels": [{"name": f"C{i}"} for i in range(8)],
           "waveform_member_bytes": 56000, "eeg_sample_bytes": 44800}
    accepted = {"id": "window", "source_sha256": digest, "source_recording_id": "recording", "split": "train", "subject_group": "group",
                "window_start_s": 5, "window_end_s": 35, "channels": [f"C{i}" for i in range(8)], "sample_rate_hz": 200,
                "unit": "V_x_1e4", "eeg_path": "waveforms/a.npy", "eeg_index": 0}
    cert = {"status": "completed", "source_sha256": digest, "source_recording_id": "recording", "split": "train", "subject_group": "group",
            "original_signal_seconds": 70., "accepted_windows": 1, "candidate_windows": 2, "accepted_unique_source_seconds": 30,
            "records": [accepted], "rejected_windows": [{"window_start_s": 35, "window_end_s": 65, "reasons": ["max_abs_exceeds_100uV"]}],
            "unit_check": {"status": "passed"}, "rejected_reason_counts": {"max_abs_exceeds_100uV": 1},
            "artifacts": {"waveforms/a.npy": {"shape": [1, 8, 30, 200], "dtype": "float32", "bytes": 192128}}}
    return row, cert


class AccountingTests(unittest.TestCase):
    def test_complete_disjoint_grid_and_quantities(self):
        row, cert = fixture()
        result = audit.audit_certificate(row, cert)
        self.assertEqual(result["candidate_windows"], 2)
        self.assertEqual(result["accepted_windows"], 1)
        self.assertEqual(result["rejected_windows"], 1)
        self.assertEqual(result["accepted_unique_source_seconds"], 30)
        self.assertEqual(result["original_source_seconds"], 70.)
        self.assertEqual(result["original_eeg_sample_bytes"], 44800)
        self.assertEqual(result["original_waveform_bytes"], 56000)

    def test_interval_gap_overlap_or_duplicate_is_not_a_complete_corpus(self):
        row, original = fixture()
        for error in ("gap", "overlap", "duplicate"):
            with self.subTest(error=error):
                cert = copy.deepcopy(original)
                if error == "gap":
                    cert["rejected_windows"] = []
                elif error == "overlap":
                    cert["rejected_windows"][0].update(window_start_s=5, window_end_s=35)
                else:
                    cert["records"].append(copy.deepcopy(cert["records"][0]))
                with self.assertRaises(ValueError):
                    audit.audit_certificate(row, cert)

    def test_private_label_fabricated_channel_and_unreferenced_slot_rejected(self):
        row, original = fixture()
        for error in ("label", "channel", "slot"):
            with self.subTest(error=error):
                cert = copy.deepcopy(original)
                if error == "label":
                    cert["records"][0]["label"] = "ai_prompt_submit"
                elif error == "channel":
                    cert["records"][0]["channels"][0] = "fabricated"
                else:
                    cert["artifacts"]["waveforms/a.npy"]["shape"][0] = 2
                with self.assertRaises(ValueError):
                    audit.audit_certificate(row, cert)


if __name__ == "__main__":
    unittest.main()
