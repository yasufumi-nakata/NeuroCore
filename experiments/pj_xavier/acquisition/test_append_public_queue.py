import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import acquire
import append_public_queue as a


def release(cid="000010", source="10", **changes):
    row = {"catalog_id": cid, "provider": "zenodo", "source_id": source, "source_version": "1", "status": "resolved",
           "scope_decision": "include_provisional", "training_use_status": "license_and_source_terms_review_pending",
           "title": "Human EEG dataset", "source_description": "This dataset contains recorded EEG data from human participants.",
           "metadata_body_sha256": "a" * 64, "manifest_evidence_url": "https://zenodo.org/api/records/" + source,
           "files": [acquire.make_file("signal.eeg", "https://zenodo.org/records/" + source + "/files/signal.eeg", 10,
                                       "md5:" + "b" * 32, evidence_url="https://zenodo.org/api/records/" + source)]}
    row.update(changes)
    return row


class AppendQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "ledger.sqlite"
        self.db = acquire.init_db(self.db_path)
        self.number = 0

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def plan(self, rows):
        self.number += 1
        source = self.root / "fresh.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        catalog = self.root / "catalog.csv"
        with catalog.open("w", newline="") as f:
            writer = csv.writer(f);writer.writerow(["ID"]);writer.writerows([[row["catalog_id"]] for row in rows])
        output = self.root / ("plan" + str(self.number))
        result = a.make_plan(source, self.root / "snapshots", catalog, self.db_path, output, 0)
        return output, result

    def apply(self, output, suffix=""):
        return a.apply_plan(output, self.db_path, acquire.hash_file(output / "plan.json"), self.root / ("receipt" + suffix + ".json"), 0)

    def test_append_preserves_every_existing_field_including_running_and_complete(self):
        old, new = release(), release("000011", "11")
        acquire.ingest(self.db, [old])
        self.db.execute("UPDATE downloads SET status='running',bytes_present=7,attempts=3,lease_until=999999,error='old'")
        self.db.commit()
        before = self.db.execute("SELECT * FROM downloads").fetchone()
        output, plan = self.plan([old, new])
        self.assertEqual(plan["new_job_candidates"], 1)
        result = self.apply(output)
        self.assertEqual(result["inserted_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT * FROM downloads WHERE id=?", (before[0],)).fetchone(), before)
        self.assertEqual(self.apply(output, "repeat")["inserted_jobs"], 0)
        self.assertEqual(self.db.execute("SELECT count(*) FROM downloads").fetchone()[0], 2)

    def test_same_pinned_size_change_holds_one_release_not_other_additions(self):
        old = release()
        acquire.ingest(self.db, [old])
        changed = copy.deepcopy(old);changed["files"][0]["size_bytes"] = 99
        output, plan = self.plan([changed, release("000011", "11")])
        self.assertEqual(plan["counts"]["metadata_conflict_held"], 1)
        self.assertEqual(self.apply(output)["inserted_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT expected_bytes FROM downloads WHERE catalog_id='000010'").fetchone()[0], 10)

    def test_same_name_new_provider_id_preserves_existing_path_and_namespaces_new_file(self):
        row = release(provider="figshare")
        row["files"] = [acquire.make_file("same.eeg", "https://api.figshare.com/file/download/1", 10, "md5:" + "b" * 32,
                                          evidence_url="https://api.figshare.com/v2/articles/10", provider_file_id="1")]
        acquire.ingest(self.db, [row])
        old = self.db.execute("SELECT * FROM downloads").fetchone()
        expanded = copy.deepcopy(row)
        expanded["files"].append(acquire.make_file("same.eeg", "https://api.figshare.com/file/download/2", 20, "md5:" + "c" * 32,
                                                   evidence_url="https://api.figshare.com/v2/articles/10", provider_file_id="2"))
        output, plan = self.plan([expanded])
        self.assertEqual(plan["new_job_candidates"], 1)
        self.assertEqual(self.apply(output)["inserted_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT * FROM downloads WHERE id=?", (old[0],)).fetchone(), old)
        new_path = self.db.execute("SELECT relative_path FROM downloads WHERE provider_file_id='2'").fetchone()[0]
        self.assertTrue(new_path.startswith("provider_files/figshare_2_"))

    def test_unknown_size_is_metadata_blocked(self):
        row = release();row["files"][0]["size_bytes"] = None
        output, _ = self.plan([row])
        result = self.apply(output)
        self.assertEqual(result["inserted_unknown_size_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT status,expected_bytes FROM downloads").fetchone(), ("metadata_blocked", None))

    def test_seed_releases_and_unresolved_scopes_are_not_silently_added(self):
        candidate = release("000001", "19440997", scope_decision="unresolved",
                            title="A steady-state visual evoked potential (SSVEP)-based BCI dataset in children")
        seed = release("010914", "20801101", scope_decision="component_only_provisional")
        output, plan = self.plan([candidate, seed])
        self.assertEqual(plan["new_job_candidates"], 0)
        review = list(a.read_rows(output / "scope_review.jsonl"))[0]
        self.assertEqual(review["priority"], "official_description_EEG_data_candidate")
        self.assertEqual(review["training_approval"], "not_granted")
        self.assertFalse(review["automatically_ingested"])

    def test_file_extension_alone_is_not_EEG_scope_evidence(self):
        row = release(scope_decision="unresolved", title="Mixed files", source_description="No signal content is described.")
        review = a.source_review(row)
        self.assertIsNone(review["proposed_acquisition_scope"])

    def test_snapshot_and_plan_tampering_fail_before_database_changes(self):
        output, plan = self.plan([release()])
        self.root.joinpath("fresh.jsonl").write_text("replaced mutable upstream file")
        self.assertEqual(acquire.hash_file(plan["source_snapshot"]), plan["source_snapshot_sha256"])
        with (output / "releases.jsonl").open("a") as f:f.write("\n")
        with self.assertRaisesRegex(ValueError, "plan artifact changed"):
            self.apply(output)
        self.assertEqual(self.db.execute("SELECT count(*) FROM downloads").fetchone()[0], 0)

    def test_changed_metadata_since_plan_is_held_without_overwrite(self):
        old = release();acquire.ingest(self.db, [old])
        output, _ = self.plan([old, release("000011", "11")])
        self.db.execute("UPDATE downloads SET expected_bytes=11 WHERE catalog_id='000010'");self.db.commit()
        result = self.apply(output)
        self.assertEqual(len(result["held_releases"]), 1)
        self.assertEqual(result["inserted_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT expected_bytes FROM downloads WHERE catalog_id='000010'").fetchone()[0], 11)

    def test_official_access_hold_applies_to_new_jobs_without_worker_start(self):
        row = release();output, _ = self.plan([row])
        self.db.execute("CREATE TABLE dataset_access_holds(dataset_key TEXT PRIMARY KEY,reason TEXT)")
        self.db.execute("INSERT INTO dataset_access_holds VALUES(?,?)", (acquire.dataset_storage_key(row), "guestbook_required"));self.db.commit()
        result = self.apply(output)
        self.assertEqual(result["access_requirement_held_new_jobs"], 1)
        self.assertEqual(self.db.execute("SELECT status FROM downloads").fetchone()[0], "access_action_required")
        self.assertEqual(result["worker_start_stop_operations"], 0)

    def test_sql_authorizer_prevents_download_update_side_effect_from_trigger(self):
        old = release();acquire.ingest(self.db, [old])
        self.db.execute("CREATE TRIGGER unexpected_update AFTER INSERT ON downloads BEGIN UPDATE downloads SET status='bad' WHERE catalog_id='000010'; END")
        self.db.commit()
        output, _ = self.plan([old, release("000011", "11")])
        result = self.apply(output)
        self.assertEqual(result["inserted_jobs"], 0)
        self.assertEqual(len(result["held_releases"]), 1)
        self.assertEqual(self.db.execute("SELECT status FROM downloads").fetchone()[0], "queued")


if __name__ == "__main__":
    unittest.main()
