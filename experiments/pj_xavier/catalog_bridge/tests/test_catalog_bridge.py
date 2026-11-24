"""Only synthetic catalogue metadata; never touches EEG or live S Mac jobs."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catalog_identity as identity
import catalog_diff_standalone as bridge


class IdentityTests(unittest.TestCase):
    def test_doi_and_provider(self):
        self.assertEqual(identity.norm_doi("https://doi.org/10.18112/openneuro.ds005841.v1.0.0"),
                         "10.18112/openneuro.ds005841.v1.0.0")
        self.assertIn("openneuro:ds005841", identity.keys(url="https://openneuro.org/datasets/ds005841"))
        self.assertIn("zenodo:19440997", identity.keys(doi="10.5281/zenodo.19440997"))
        self.assertNotEqual(identity.keys(url="https://example.org/a"), identity.keys(url="https://example.org/b"))

    def test_no_false_match_on_title_alone(self):
        candidate = {"high:0": {"title": "EEG Similar Title", "url": "https://a.org/signal",
                                "keys": sorted(identity.keys(url="https://a.org/signal")),
                                "collection": "high", "access": "すぐに使える", "eeg_text_match": True}}
        root = [{"title": "EEG Similar Title", "source_url": "https://b.org/signal", "status": "unknown"}]
        result = bridge.assess(candidate, iter(root))
        self.assertEqual(result["matches"].get("high_unmatched"), 1)
        self.assertEqual(result["exact_title_url_aliases"], 0)

    def test_high_confidence_and_heldout_metadata(self):
        candidate = {
            "high:0": {"title": "Demo", "url": "https://zenodo.org/records/1234", "keys": list(identity.keys(url="https://zenodo.org/records/1234")), "collection": "high", "access": "すぐに使える"},
            "global:0": {"title": "Unresolved", "url": "https://example.org/unresolved", "keys": list(identity.keys(url="https://example.org/unresolved")), "collection": "global", "access": "immediate", "eeg_text_match": True},
            "global:1": {"title": "Account only", "url": "https://example.org/registered", "keys": list(identity.keys(url="https://example.org/registered")), "collection": "global", "access": "account_required", "eeg_text_match": True},
        }
        root = [{"title": "Demo", "source_url": "https://zenodo.org/records/1234", "source_family_key": "zenodo_record:1234", "status": "resolved"}]
        result = bridge.assess(candidate, iter(root))
        self.assertEqual(result["matches"]["high_matched"], 1)
        self.assertEqual(result["matches"]["global_unmatched"], 2)
        self.assertEqual(len(result["candidate_review_queue"]), 1)
        self.assertFalse(result["candidate_review_queue"][0]["download_authorized"])

    def test_exact_title_url_alias(self):
        candidate = {"high:0": {"title": "ABC BCI", "url": "https://m.example.org/project/", "keys": [], "collection": "high"}}
        root = [{"title": "ABC BCI", "source_url": "https://m.example.org/project", "status": "metadata_only"}]
        out = bridge.assess(candidate, iter(root))
        self.assertEqual(out["exact_title_url_aliases"], 1)
        self.assertEqual(out["matches"]["high_matched"], 1)

    def test_csv_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audit/global_eeg").mkdir(parents=True)
            with (root / "eeg_dataset_summary_ja.csv").open("w", encoding="utf-8", newline="") as fp:
                wr = csv.DictWriter(fp, fieldnames=["データセット名", "公開URL", "DOI", "アクセス区分"])
                wr.writeheader()
                wr.writerow({"データセット名": "Demo", "公開URL": "https://openneuro.org/datasets/ds005841", "DOI": "", "アクセス区分": "すぐに使える"})
            with (root / "audit/global_eeg/global_eeg_merged.csv").open("w", encoding="utf-8", newline="") as fp:
                wr = csv.DictWriter(fp, fieldnames=["title", "url", "doi", "access", "eeg_text_match"])
                wr.writeheader()
                wr.writerow({"title": "Demo", "url": "https://openneuro.org/datasets/ds005841", "access": "immediate", "eeg_text_match": "True"})
            previous = identity.ROOT
            try:
                identity.ROOT = root
                rows = identity.build_candidates()
            finally:
                identity.ROOT = previous
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("openneuro:ds005841" in row["keys"] for row in rows.values()))


if __name__ == "__main__":
    unittest.main()
