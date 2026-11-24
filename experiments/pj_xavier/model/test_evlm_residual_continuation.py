"""CPU guard tests; no EEG, model or heldout access."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlm_residual_continuation as gate


class ContinuationTest(unittest.TestCase):
    def test_parent_baseline_exact(self):
        m = {"n": 36, "valid_count": 36, "mse_on_valid_predictions": 14.9}
        gate.verify_baseline(m, m)
        for change in ({"valid_count": 35}, {"n": 35}, {"mse_on_valid_predictions": 15.0}):
            with self.assertRaises(ValueError):
                gate.verify_baseline(m, {**m, **change})

    def test_bound_receipt_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "receipt.json"
            p.write_bytes(b"{}")
            entry = {"path": str(p), "sha256": hashlib.sha256(b"{}").hexdigest()}
            self.assertEqual(gate.bound(entry), p)
            p.write_bytes(b"{ } ")
            with self.assertRaises(ValueError):
                gate.bound(entry)


if __name__ == "__main__":
    unittest.main()
