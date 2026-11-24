"""Synthetic-only tests for residual reconstruction, absolute scores and heldout guards."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evlm_residual_protocol as residual


class ResidualProtocolTest(unittest.TestCase):
    def test_synthetic_contract(self):
        residual.selftest()

    def test_invalid_output_does_not_become_zero(self):
        row = {"deterministic_baseline": [99, 0, -99, 1]}
        self.assertEqual(residual.absolute_output("[1,0,-1,0]", row), "[100,0,-100,1]")
        self.assertEqual(residual.absolute_output("[2,0,-1,0]", row), "INVALID")
        self.assertEqual(residual.absolute_output("[0,0,0,null]", row), "INVALID")

    def test_unpaired_rejected(self):
        with self.assertRaises(ValueError):
            residual.absolute_inputs([], [{}])


if __name__ == "__main__":
    unittest.main()
