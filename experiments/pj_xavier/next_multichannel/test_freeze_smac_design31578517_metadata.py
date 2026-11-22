import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import freeze_smac_design31578517_metadata as f


def fixture():
    files, rows = [], []
    for i in range(25):
        name = "sub-S%02d_task-art_eeg.edf" % i
        c = {
            "label": f.CHANNELS[:], "transducer": ["Ag/AgCl electrodes"] * 32,
            "unit": ["uV"] * 32, "pmin": ["-9830"] * 32, "pmax": ["9830"] * 32,
            "dmin": ["-32768"] * 32, "dmax": ["32767"] * 32,
            "prefilter": ["HP:1.0Hz;LP:49.0Hz"] * 32,
            "samples": ["12288"] * 32, "reserved": ["RESERVED"] * 32,
        }
        size = 8448 + 32 * 12288 * 2
        files.append({"name": name, "id": 800000 + i, "size": size,
                      "is_link_only": False, "computed_md5": "%032x" % (i+1),
                      "supplied_md5": "%032x" % (i+1)})
        rows.append({"relative_path": name, "file_sha256": "%064x" % (i+1),
                     "expected_bytes": size, "actual_bytes": size,
                     "header_sha256": "%064x" % (100+i), "n_channels": 32,
                     "header_bytes": 8448, "n_records": 1, "record_duration_s": 12.,
                     "duration_s": 12., "channels": c, "payload_read": False})
    metadata = {"id": 31578517, "version": 2, "is_public": True,
                "download_disabled": False, "is_confidential": False,
                "is_embargoed": False, "license": {"name": "CC BY 4.0"}, "files": files}
    evidence = {"source": f.SOURCE, "rows": rows, "signal_payload_read": False,
                "raw_signal_bytes_to_workstation": 0,
                "author_code": {"pre.py": "synthetic publisher script",
                                "README.txt": "25 subjects have complete EEG. Recorded EEG Channels: 32 channels. Approximately 1024 Hz."}}
    return metadata, evidence


class MetadataFreezeTests(unittest.TestCase):
    def test_all25_groups_fixed17_5_3_and_input_order_independent(self):
        m, h = fixture()
        a = f.make_plan(m, h)
        m["files"].reverse()
        h["rows"].reverse()
        self.assertEqual(a, f.make_plan(m, h))
        self.assertEqual(a["split_counts"], {"train": 17, "val": 5, "test": 3})
        self.assertEqual(len({r["subject_group"] for r in a["records"]}), 25)
        self.assertTrue(all(not r["signal_opened_before_freeze"] for r in a["records"]))
        self.assertFalse(a["online_harness_allowed"])
        self.assertTrue(a["unresolved_cross_dataset_participant_linkage"])

    def test_duplicate_group_file_and_payload_metadata_require_review(self):
        for field in ("file_sha256", "relative_path"):
            m, h = fixture()
            h["rows"][1][field] = h["rows"][0][field]
            with self.assertRaises(ValueError):
                f.make_plan(m, h)
        m, h = fixture()
        m["files"][1]["computed_md5"] = m["files"][0]["computed_md5"]
        m["files"][1]["supplied_md5"] = m["files"][0]["computed_md5"]
        with self.assertRaisesRegex(ValueError, "must-link"):
            f.make_plan(m, h)

    def test_actual_fractional_rate_is_retained_and_bad_unit_fails(self):
        m, h = fixture()
        h["rows"][0]["record_duration_s"] = 11.999999
        h["rows"][0]["duration_s"] = 11.999999
        plan = f.make_plan(m, h)
        row = next(r for r in plan["records"] if r["relative_path"] == h["rows"][0]["relative_path"])
        self.assertNotEqual(row["native_sample_rate_hz"], 1024.)
        self.assertEqual(row["native_rate_numerator"] / row["native_rate_denominator"],
                         row["native_sample_rate_hz"])
        h["rows"][0]["channels"]["unit"][0] = "mV"
        with self.assertRaisesRegex(ValueError, "physical unit"):
            f.make_plan(m, h)

    def test_source_conditions_header_geometry_and_prior_signal_reads_fail_closed(self):
        edits = [
            lambda m, h: m.update(download_disabled=True),
            lambda m, h: m.update(version=3),
            lambda m, h: h.update(signal_payload_read=True),
            lambda m, h: h["rows"][0].update(payload_read=True),
            lambda m, h: h["rows"][0].update(actual_bytes=1),
        ]
        for edit in edits:
            m, h = fixture()
            edit(m, h)
            with self.assertRaises(ValueError):
                f.make_plan(m, h)

    def test_freeze_writes_exact_seal_and_timestamp_without_signal_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m, h = fixture()
            mp, hp = root/"metadata.json", root/"header_evidence.json"
            mp.write_bytes(f.canonical(m)); hp.write_bytes(f.canonical(h))
            with mock.patch.multiple(f, METADATA_SHA256=f.sha(mp), HEADER_EVIDENCE_SHA256=f.sha(hp)):
                receipt = f.freeze(mp, hp, root/"freeze")
                self.assertEqual(receipt["plan_sha256"], f.sha(root/"freeze/frozen_plan.json"))
                plan = json.loads((root/"freeze/frozen_plan.json").read_text())
                self.assertEqual(receipt["frozen_at_utc"], plan["frozen_at_utc"])
                self.assertEqual(receipt["metadata_signal_reads"], 0)
                with self.assertRaisesRegex(ValueError, "immutable"):
                    f.freeze(mp, hp, root/"freeze")
                mp.write_text("{}")
                with self.assertRaisesRegex(ValueError, "seal differs"):
                    f.freeze(mp, hp, root/"second_freeze")


if __name__ == "__main__":
    unittest.main()
