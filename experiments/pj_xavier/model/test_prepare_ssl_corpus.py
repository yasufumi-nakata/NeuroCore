import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest import mock
import zipfile

import numpy as np

import prepare_ssl_corpus as p


def source_fixture():
    channels = [{"name": f"C{i}", "index_in_file": i, "gain_to_volts": 1e-7, "offset_volts": 0.} for i in range(32)]
    rows, sources = [], {}
    for device in ("BA", "EM", "LA"):
        key = hashlib.sha256(device.encode()).hexdigest()
        group = "zenodo:15698441:sub-fixture"
        row = {"waveform_sha256": key, "participant_group": group, "recording_id": device,
               "dataset_id": "zenodo:15698441", "device_session": device, "ssl_split": "train",
               "eeg_channels": copy.deepcopy(channels), "channels_in_file": 32}
        rows.append(row)
        sources[key] = {"split": "train", "subject_group": group, "source_recording_id": device, "dataset": row["dataset_id"]}
    return rows, sources


class PreparationTests(unittest.TestCase):
    def test_fixed_subject_split_is_order_independent_and_whole_group(self):
        ids = [f"zenodo:15698441:sub-{i:03d}" for i in range(1, 20)]
        left = p.participant_split(ids)
        self.assertEqual(left, p.participant_split(list(reversed(ids))))
        self.assertEqual(sum(v == "train" for v in left.values()), 15)
        self.assertEqual(sum(v == "val" for v in left.values()), 4)
        with self.assertRaisesRegex(ValueError, "19 groups"):
            p.participant_split(ids[:-1] + [ids[0]])

    def test_source_crossfold_and_native_channel_order_leakage_are_rejected(self):
        rows, sources = source_fixture()
        p.validate_source_declarations(rows, sources, set())
        bad = copy.deepcopy(sources)
        bad[rows[1]["waveform_sha256"]]["split"] = "val"
        changed = copy.deepcopy(rows)
        changed[1]["ssl_split"] = "val"
        with self.assertRaisesRegex(ValueError, "different folds"):
            p.validate_source_declarations(changed, bad, set())
        changed = copy.deepcopy(rows)
        changed[0]["eeg_channels"].reverse()
        with self.assertRaisesRegex(ValueError, "original header"):
            p.validate_source_declarations(changed, sources, set())
        with self.assertRaisesRegex(ValueError, "excluded source"):
            p.validate_source_declarations(rows, sources, {rows[0]["waveform_sha256"]})

    def test_uniba_original_heldout_is_forbidden_even_if_renamed_train(self):
        row = {"waveform_sha256": "a" * 64, "participant_group": "uniba:P001", "recording_id": "r",
               "dataset_id": "zenodo:20801101", "ssl_split": "train", "channels_in_file": 10,
               "source_supervised_partition": "test", "eeg_channels": source_fixture()[0][0]["eeg_channels"][:8]}
        sources = {row["waveform_sha256"]: {"split": "train", "subject_group": row["participant_group"], "source_recording_id": "r", "dataset": row["dataset_id"]}}
        with self.assertRaisesRegex(ValueError, "nontraining UniBa"):
            p.validate_source_declarations([row], sources, set())

    def test_grid_respects_both_edges_and_never_reuses_time(self):
        self.assertEqual(p.window_grid(39.999, 128), [])
        self.assertEqual(p.window_grid(40., 128), [(5, 35)])
        grid = p.window_grid(100.1, 500)
        self.assertEqual(grid, [(5, 35), (35, 65), (65, 95)])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(grid, grid[1:])))
        self.assertLessEqual(grid[-1][1], 100.1 - 5)
        with self.assertRaises(ValueError):
            p.window_grid(float("nan"), 250)

    def test_qc_has_explicit_physical_unit_limits(self):
        values = np.tile(np.array([-100e-6, 100e-6]), (8, 3000))
        self.assertEqual(p.qc_volts(values)[0], [])
        self.assertIn("max_abs_exceeds_100uV", p.qc_volts(values * 1.000001)[0])
        values[3] = 0.
        self.assertIn("a_channel_std_below_0p01uV", p.qc_volts(values)[0])
        values[2, 3] = float("nan")
        self.assertEqual(p.qc_volts(values)[0], ["nonfinite"])

    def test_resampling_voltage_scaling_patch_layout_and_overshoot_qc(self):
        for rate in (128, 250, 256, 500):
            with self.subTest(rate=rate):
                time = np.arange(rate * 30) / rate
                native = np.tile(20e-6 * np.sin(2 * np.pi * 10 * time), (8, 1))
                array, reasons, _ = p.preprocess_window(native, rate)
                self.assertEqual(reasons, [])
                self.assertEqual(array.shape, (8, 30, 200))
                self.assertEqual(array.dtype, np.float32)
                expected = .2 * np.sin(2 * np.pi * 10 * np.arange(6000) / 200)
                np.testing.assert_allclose(array[0].reshape(-1)[200:-200], expected[200:-200], atol=0.0006, rtol=0.)
        # A bounded native step overshoots after polyphase interpolation.
        # The post-resampling QC must reject it, despite its native <=100 uV.
        step = np.zeros((8, 128 * 30))
        step[:, 128 * 15:] = 100e-6
        array, reasons, stats = p.preprocess_window(step, 128)
        self.assertIsNone(array)
        self.assertIn("max_abs_exceeds_100uV", reasons)
        self.assertGreater(stats["max_abs_volts"], 100e-6)

    def test_independent_brainvision_header_voltage_check_catches_double_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digital = np.arange(120, dtype="<f4").reshape(12, 10)
            digital.tofile(root / "eeg.eeg")
            (root / "eeg.vhdr").write_text("BinaryFormat=IEEE_FLOAT_32\nDataOrientation=MULTIPLEXED\n")
            channels = [{"name": f"C{i}", "index_in_file": i, "gain_to_volts": 1e-7, "offset_volts": 0.} for i in range(8)]
            physical = digital.T[:8].astype(np.float64) * 1e-7
            raw = types.SimpleNamespace(n_times=12, get_data=lambda picks, start, stop: physical[:, start:stop])
            row = {"eeg_channels": channels, "channels_in_file": 10, "source_format": "BrainVision"}
            self.assertEqual(p.direct_unit_check(raw, row, root / "eeg.eeg", root / "eeg.vhdr")["sample_values_compared"], 96)
            raw.get_data = lambda picks, start, stop: physical[:, start:stop] * 1e-6
            with self.assertRaisesRegex(ValueError, "calibration mismatch"):
                p.direct_unit_check(raw, row, root / "eeg.eeg", root / "eeg.vhdr")

    def test_independent_edf_gain_and_offset_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n, header_bytes = 2, 768
            fixed = bytearray(b" " * 256)
            fixed[184:192] = str(header_bytes).encode().ljust(8)
            fields = bytearray(b" " * 512)
            for index in range(n):
                fields[216 * n + index * 8:216 * n + (index + 1) * 8] = b"8       "
            records = [np.arange(i * 16, (i + 1) * 16, dtype="<i2").reshape(2, 8) for i in range(3)]
            path = root / "eeg.edf"
            path.write_bytes(bytes(fixed + fields) + b"".join(record.tobytes() for record in records))
            channels = [{"name": "a", "index_in_file": 0, "gain_to_volts": 2e-7, "offset_volts": -3e-6},
                        {"name": "b", "index_in_file": 1, "gain_to_volts": 3e-7, "offset_volts": 4e-6}]
            native = np.concatenate(records, axis=1).astype(np.float64)
            physical = np.stack([native[i] * channel["gain_to_volts"] + channel["offset_volts"] for i, channel in enumerate(channels)])
            raw = types.SimpleNamespace(get_data=lambda picks, start, stop: physical[:, start:stop])
            result = p.direct_unit_check(raw, {"eeg_channels": channels, "channels_in_file": 2, "source_format": "EDF"}, path, path)
            self.assertEqual(result["sample_values_compared"], 24)
            self.assertLess(result["maximum_absolute_difference_volts"], 1e-20)

    def test_archive_members_cannot_escape_and_corruption_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"immutable EEG fixture"
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, "w") as archive:
                archive.writestr("../outside.eeg", data)
                archive.writestr("sub/eeg.eeg", data)
            bio.seek(0)
            with zipfile.ZipFile(bio) as archive, mock.patch.object(p, "capacity_guard"):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    p.extract_member(archive, "../outside.eeg", root, len(data))
                with self.assertRaisesRegex(ValueError, "SHA/size mismatch"):
                    p.extract_member(archive, "sub/eeg.eeg", root, len(data), "0" * 64)
            self.assertFalse((root.parent / "outside.eeg").exists())

    def test_deadline_reserve_and_existing_artifacts_fail_closed(self):
        with mock.patch.object(p.time, "time", return_value=p.DEADLINE):
            with self.assertRaisesRegex(ValueError, "absolute deadline"):
                p.capacity_guard(".")
        with mock.patch.object(p.time, "time", return_value=p.DEADLINE - 100), mock.patch.object(p.shutil, "disk_usage", return_value=types.SimpleNamespace(free=p.RESERVE + 99)):
            with self.assertRaisesRegex(ValueError, "reserve"):
                p.capacity_guard(".", 100)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, "capacity_guard"):
            root = Path(directory)
            first = p.save_shard(root, "a" * 64, 0, [np.zeros((8, 30, 200), dtype=np.float32)])
            original = (root / first[0]).read_bytes()
            with self.assertRaisesRegex(ValueError, "existing shard"):
                p.save_shard(root, "a" * 64, 0, [np.ones((8, 30, 200), dtype=np.float32)])
            self.assertEqual((root / first[0]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
