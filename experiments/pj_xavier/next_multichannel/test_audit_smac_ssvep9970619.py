"""Regression checks for strict S Mac EEGLAB header audit boundaries."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import audit_smac_ssvep9970619 as audit


class TestSMACSSVEPMetadataGate(unittest.TestCase):
    def test_excluded_hosts(self):
        for host in ("smac19", "smac40", "smac31", "vpnclient231", "smac02-evil"):
            self.assertFalse(audit.approved_node(host), host)
        for host in ("smac02", "smac35.sfc.keio.ac.jp"):
            self.assertTrue(audit.approved_node(host), host)

    def test_auxiliary_channels_never_mapped_as_eeg(self):
        labels = [f"EEG_{i}" for i in range(64)] + [f"EXG{i}" for i in range(1, 9)] + ["Status"]
        ee = types.SimpleNamespace(chanlocs=[types.SimpleNamespace(labels=x) for x in labels])
        self.assertEqual(audit.checked_labels(ee), labels)
        bad = labels.copy()
        bad[64] = "EEG_64"
        ee.chanlocs = [types.SimpleNamespace(labels=x) for x in bad]
        with self.assertRaisesRegex(ValueError, "AUX/Status boundary"):
            audit.checked_labels(ee)

    def test_denied_workstation_before_file_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scipy.io.loadmat", side_effect=AssertionError("header unexpectedly opened")):
                with self.assertRaisesRegex(ValueError, "S Mac worker host"):
                    audit.inspect_headers(Path(tmp), host="vpnclient231")

    def test_missing_exact_header_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Expected 44"):
                audit.inspect_headers(Path(tmp), host="smac35")


if __name__ == "__main__":
    unittest.main()
