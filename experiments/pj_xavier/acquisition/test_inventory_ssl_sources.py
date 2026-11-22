import copy
import hashlib
import unittest

import inventory_ssl_sources as inventory


class SSLInventoryTests(unittest.TestCase):
    def test_brainvision_resolution_and_microvolt_conversion(self):
        header = b"Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nNumberOfChannels=2\n[Channel Infos]\nCh1=C3,,0.1,uV\nCh2=C4,,2,mV\n"
        _, rows = inventory.brainvision_scales(header)
        self.assertAlmostEqual(rows[0]["gain_to_volts"], 1e-7)
        self.assertAlmostEqual(rows[1]["gain_to_volts"], .002)
        self.assertEqual(rows[0]["offset_volts"], 0)
        with self.assertRaises(ValueError):
            inventory.brainvision_scales(header.replace(b"0.1,uV", b"0,uV"))
        with self.assertRaises(ValueError):
            inventory.brainvision_scales(header.replace(b"0.1,uV", b"0.1,"))

    def test_edf_physical_scale_offset_and_non_eeg_units(self):
        n = 3
        header = bytearray(b" " * (256 + 256 * n))
        header[252:256] = b"3   "
        def put(offset, width, values):
            for i, value in enumerate(values):
                start = 256 + offset * n + width * i
                header[start:start + width] = str(value).encode().ljust(width)
        put(0, 16, ["C3", "Status", "EDF Annotations"])
        put(96, 8, ["uV", "", ""])
        put(104, 8, [-100, 0, 0])
        put(112, 8, [100, 65535, 1])
        put(120, 8, [-32768, -32768, -32768])
        put(128, 8, [32767, 32767, 32767])
        result = inventory.edf_scales(bytes(header))
        first = result[0]
        self.assertAlmostEqual(-32768 * first["gain_to_volts"] + first["offset_volts"], -100e-6)
        self.assertAlmostEqual(32767 * first["gain_to_volts"] + first["offset_volts"], 100e-6)
        self.assertTrue(result[1]["physical_unit_not_voltage"])
        self.assertTrue(result[2]["annotation_channel"])

    def test_units_match_and_non_eeg_channels_are_excluded(self):
        scales = [{"name": "C3", "header_unit": "uV", "gain_to_volts": 1e-6, "offset_volts": 0.},
                  {"name": "Status", "header_unit": "", "physical_unit_not_voltage": True}]
        tsv = "name\ttype\tunits\tstatus\nC3\tEEG\tµV\tgood\nStatus\tMISC\tn/a\tgood\n".encode()
        result = inventory.eeg_channels(tsv, scales)
        self.assertEqual([item["name"] for item in result], ["C3"])
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            inventory.eeg_channels(tsv.replace("µV".encode(), b"mV"), scales)
        recorded = inventory.eeg_channels(tsv.replace("µV".encode(), b"V"), scales, document_unit_disagreement=True)
        self.assertFalse(recorded[0]["bids_and_header_unit_agree"])
        self.assertEqual(recorded[0]["header_unit"], "uV")
        self.assertEqual(recorded[0]["bids_unit"], "V")
        self.assertEqual(recorded[0]["gain_to_volts"], 1e-6)
        with self.assertRaises(ValueError):
            inventory.unit_to_volts("unknown")

    def test_three_devices_cannot_be_counted_as_three_participants(self):
        records = [{"participant_group": f"d:s{subject}", "device_session": device,
                    "waveform_sha256": hashlib.sha256(f"{subject}{device}".encode()).hexdigest()}
                   for subject in range(19) for device in ("BA", "EM", "LA")]
        self.assertEqual(len(inventory.validate_portable_groups(records)), 19)
        changed = [{**row, "participant_group": row["participant_group"] + row["device_session"]} for row in records]
        with self.assertRaises(ValueError):
            inventory.validate_portable_groups(changed)
        records[0]["device_session"] = "EM"
        with self.assertRaises(ValueError):
            inventory.validate_portable_groups(records)

    def test_only_uniba_train_sources_survive_with_no_itu(self):
        rows = []
        for index in range(758):
            source = index % 32
            rows.append({"id": str(index), "subject": "uniba:" + str(source), "eeg_source_sha256": hashlib.sha256(str(source).encode()).hexdigest(),
                         "split": "train" if source < 18 else "val" if source < 24 else "test"})
        train, excluded = inventory.select_uniba_training(rows)
        self.assertEqual(len(train), 18)
        self.assertEqual(len(excluded), 14)
        self.assertEqual({row["split"] for row in train}, {"train"})
        altered = copy.deepcopy(rows)
        altered[0]["subject"] = "itu:0"
        with self.assertRaisesRegex(ValueError, "ITU"):
            inventory.select_uniba_training(altered)
        altered = copy.deepcopy(rows)
        altered[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "crosses"):
            inventory.select_uniba_training(altered)

    def test_member_traversal_is_rejected(self):
        for bad in ("../other.eeg", "/absolute.eeg", "folder\\other.eeg"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                inventory.contained_member("raw data/sub-001", bad)


if __name__ == "__main__":
    unittest.main()
