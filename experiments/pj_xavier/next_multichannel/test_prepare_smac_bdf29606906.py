"""Synthetic-only tests for the sealed Figshare 29606906 offline SSL admission."""
from pathlib import Path
from unittest import TestCase, mock
import subprocess

import numpy as np

from prepare_smac_bdf29606906 import causal_windows, freeze_members, split_for_subject


class BDFPreparationTests(TestCase):
    def test_registered_subject_groups_have_frozen_metadata_only_partition(self):
        roster = [f"dataset/sub{n}/recording.bdf" for n in range(1, 21)]
        with mock.patch("prepare_smac_bdf29606906.subprocess.run", return_value=subprocess.CompletedProcess(
            args=["bsdtar"], returncode=0, stdout="\n".join(roster) + "\n")):
            members = freeze_members(Path("/nonexistent/dataset.rar"))
        self.assertEqual(len(members), 20)
        self.assertEqual(sum(x["split"] == "train" for x in members.values()), 14)
        self.assertEqual(sum(x["split"] == "val" for x in members.values()), 5)
        self.assertEqual(sum(x["split"] == "test" for x in members.values()), 1)
        self.assertTrue(all(not x["source_signal_opened"] for x in members.values()))
        self.assertEqual(members["sub1"]["split"], split_for_subject("sub1"))

    def test_rejects_unrecognized_heldout_filename(self):
        roster = [f"dataset/sub{n}/recording.bdf" for n in range(1, 21)]
        roster[-1] = "dataset/sub20/../../bad.bdf"
        with mock.patch("prepare_smac_bdf29606906.subprocess.run", return_value=subprocess.CompletedProcess(
            args=["bsdtar"], returncode=0, stdout="\n".join(roster) + "\n")):
            with self.assertRaises(Exception):
                freeze_members(Path("/nonexistent/dataset.rar"))

    def test_rejects_duplicate_subjects(self):
        roster = [f"dataset/sub{n}/recording.bdf" for n in range(1, 21)]
        roster[-1] = "dataset/sub1/other.bdf"
        with mock.patch("prepare_smac_bdf29606906.subprocess.run", return_value=subprocess.CompletedProcess(
            args=["bsdtar"], returncode=0, stdout="\n".join(roster) + "\n")):
            with self.assertRaises(Exception):
                freeze_members(Path("/nonexistent/dataset.rar"))

    def test_causal_windows_only_use_past_samples_and_fixed_qc(self):
        fs = 500
        t = np.arange(12 * fs) / fs
        x = np.stack([5e-6 * np.sin(2 * np.pi * 10 * t + i / 10) for i in range(16)])
        a, wa, rejects = causal_windows(x)
        self.assertEqual(rejects, {})
        self.assertEqual(a.shape[1:], (16, 3, 200))
        self.assertGreaterEqual(len(a), 2)
        modified = x.copy()
        modified[:, 10 * fs:] = 0.04  # Change only future samples well after first 3 s window.
        b, wb, _ = causal_windows(modified)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(wa[0], wb[0])
        self.assertLessEqual(wa[0]["latest_native_source_index"], 10 * fs)
        self.assertFalse(np.isnan(a).any())

    def test_flat_signal_is_not_promoted_to_training(self):
        signal = np.zeros((16, 500 * 8), dtype=np.float64)
        arr, windows, rejects = causal_windows(signal)
        self.assertEqual(arr.shape[0], 0)
        self.assertEqual(len(windows), 0)
        self.assertGreater(rejects.get("flat_below_fixed_0_01uV", 0), 0)
