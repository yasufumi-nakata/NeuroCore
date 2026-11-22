import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import audit_eegmmidb_zip as a


def make_edf(*, patient="S001", declared=2, annotation_duration="1", offset=0):
    def field(value, width):
        value = str(value).encode("ascii")
        assert len(value) <= width
        return value.ljust(width, b" ")
    fixed = b"".join([field(0, 8), field(patient, 80), field("test", 80), field("01.01.00", 8),
                      field("00.00.00", 8), field(1024, 8), field("EDF+C", 44),
                      field(declared, 8), field(1, 8), field(3, 4)])
    columns = [(16, ["C3", "C4", "EDF Annotations"]), (80, ["", "", ""]),
               (8, ["uV", "uV", ""]), (8, [-100, -100, -1]), (8, [100, 100, 1]),
               (8, [-32768, -32768, -32768]), (8, [32767, 32767, 32767]),
               (80, ["", "", ""]), (8, [4, 8, 32]), (32, ["", "", ""])]
    signal = b"".join(field(value, width) for width, values in columns for value in values)
    records = []
    for i in range(2):
        values = struct.pack("<12h", *(offset + j + i for j in range(12)))
        annotation = (f"+{i}\x14\x14\x00+{i}\x15{annotation_duration}\x14T{i + 1}\x14\x00").encode()
        records.append(values + annotation.ljust(64, b"\x00"))
    return fixed + signal + b"".join(records)


def make_release(root, *, second=False, mismatch=False, prefix="release/"):
    members = {"S001/S001R04.edf": make_edf(), "S001/S001R04.edf.event": b"event companion"}
    if second:
        members["S002/S002R04.edf"] = make_edf(patient="S002")
        members["S002/S002R04.edf.event"] = b"event companion"
    edfs = sorted(p for p in members if p.endswith(".edf"))
    members["RECORDS"] = ("\n".join(edfs) + "\n").encode()
    text = "".join(hashlib.sha256(value).hexdigest() + " " + key + "\n" for key, value in members.items())
    checksum = root / "official_SHA256SUMS.txt"
    checksum.write_text(text)
    records = root / "official_RECORDS"
    records.write_bytes(members["RECORDS"])
    members["SHA256SUMS.txt"] = text.encode()
    archive = root / "release.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(prefix + name, data + b"changed" if mismatch and name.endswith(".event") else data)
    return archive, checksum, records


class EEGAuditTests(unittest.TestCase):
    def test_channel_samples_duration_and_annotation_bytes_are_disjoint(self):
        data = make_edf()
        row = a.audit_edf(io.BytesIO(data), len(data), "S001/S001R04.edf")
        self.assertEqual(row["eeg_channel_count"], 2)
        self.assertEqual(row["annotation_channel_count"], 1)
        self.assertEqual(row["eeg_sample_values"], 24)
        self.assertEqual(row["eeg_sample_bytes"], 48)
        self.assertEqual(row["annotation_payload_bytes"], 128)
        self.assertEqual(row["signal_duration_seconds_decimal"], "2")
        self.assertEqual(row["channels"][0]["sample_rate_hz"], 4)
        self.assertEqual(row["channels"][1]["sample_rate_hz"], 8)
        self.assertEqual(row["annotation_counts"], {"T1": 1, "T2": 1})
        self.assertEqual(row["semantic_annotation_counts"], {"imagined_left_fist": 1, "imagined_right_fist": 1})
        self.assertFalse(row["annotation_errors"])
        self.assertEqual(row["header_bytes"] + row["eeg_sample_bytes"] + row["annotation_payload_bytes"], len(data))

    def test_task_mapping_does_not_turn_execution_into_imagery_or_intent(self):
        self.assertEqual(a.semantic_label(3, "T1"), "executed_left_fist")
        self.assertEqual(a.semantic_label(6, "T1"), "imagined_both_fists")
        self.assertEqual(a.semantic_label(6, "T2"), "imagined_both_feet")
        self.assertEqual(a.semantic_label(1, "T0"), "baseline_eyes_open")
        self.assertEqual(a.semantic_label(2, "T1"), "unmapped_annotation")
        with self.assertRaises(ValueError):
            a.run_task(15)

    def test_short_body_wrong_record_count_and_trailing_bytes_rejected(self):
        for data in (make_edf()[:-1], make_edf(declared=4), make_edf() + b"extra"):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                a.audit_edf(io.BytesIO(data), len(data), "S001/S001R04.edf")

    def test_unspecified_record_count_derived_from_exact_payload_and_reported(self):
        data = make_edf(declared=-1)
        row = a.audit_edf(io.BytesIO(data), len(data), "S001/S001R04.edf")
        self.assertTrue(row["record_count_inferred_from_size"])
        self.assertEqual(row["data_records"], 2)

    def test_out_of_bounds_and_overlap_annotations_remain_visible(self):
        data = make_edf(annotation_duration="2")
        row = a.audit_edf(io.BytesIO(data), len(data), "S001/S001R04.edf")
        self.assertEqual(row["annotation_errors"]["annotation_outside_signal_duration"], 1)
        self.assertEqual(row["annotation_errors"]["overlapping_annotation_intervals"], 1)

    def test_complete_release_hashes_headers_and_duplicate_payload_not_double_added(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, checksums, records = make_release(root, second=True)
            out = root / "audit"
            result = a.audit(archive, checksums, records, out, prefix="release/", expected_archive_sha256=a.sha_file(archive))
            self.assertEqual(result["eeg_recordings"], 2)
            self.assertEqual(result["publisher_subject_groups"], 2)
            self.assertEqual(result["unique_EEG_payload_recordings"], 1)
            self.assertEqual(result["all_recording_signal_seconds_decimal"], "4")
            self.assertEqual(result["content_unique_signal_seconds_decimal"], "2")
            self.assertTrue(result["all_published_member_hashes_verified"])
            self.assertEqual(result["unlisted_members"], [])
            self.assertTrue((out / "completion.json").is_file())
            with self.assertRaises(ValueError):
                a.audit(archive, checksums, records, out, prefix="release/")

    def test_bad_member_hash_does_not_publish_completed_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, checksums, records = make_release(root, mismatch=True)
            out = root / "audit"
            with self.assertRaisesRegex(ValueError, "official member SHA mismatch"):
                a.audit(archive, checksums, records, out, prefix="release/")
            self.assertFalse((out / "summary.json").exists())
            self.assertFalse((out / "completion.json").exists())

    def test_record_coverage_and_expanded_budget_fail_closed(self):
        for reason in ("coverage", "budget", "deadline", "archive_sha"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive, checksums, records = make_release(root)
                kw = {"prefix": "release/"}
                if reason == "coverage":
                    records.write_text("S001/S001R04.edf\nS002/S002R04.edf\n")
                elif reason == "budget":
                    kw["max_uncompressed_bytes"] = 1
                elif reason == "deadline":
                    kw["deadline"] = 1
                else:
                    kw["expected_archive_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    a.audit(archive, checksums, records, root / "audit", **kw)
                self.assertFalse((root / "audit/completion.json").exists())

    def test_paths_checksum_duplicates_and_malformed_TAL_rejected(self):
        for name in ("/escape", "a/../b", "a//b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                a.safe_name(name)
        with self.assertRaises(ValueError):
            a.checksums(("a" * 64 + " x\n") * 2)
        with self.assertRaises(ValueError):
            a.tal_annotations(b"+0\x14T1\x00")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, checksums, records = make_release(root)
            with zipfile.ZipFile(archive, "a") as z:
                z.writestr("release/../outside", b"no")
            with self.assertRaises(ValueError):
                a.audit(archive, checksums, records, root / "audit", prefix="release/")


if __name__ == "__main__":
    unittest.main()
