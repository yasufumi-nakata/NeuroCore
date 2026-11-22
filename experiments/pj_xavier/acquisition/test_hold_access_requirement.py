import json
from pathlib import Path
import tempfile
import unittest

import acquire
import hold_access_requirement as h


class AccessHoldTests(unittest.TestCase):
    def test_release_hold_preserves_complete_running_and_unrelated_http400(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            db = acquire.init_db(p / "ledger.sqlite")
            files = [acquire.make_file(str(i) + ".eeg", "https://dataverse.harvard.edu/api/access/datafile/" + str(i), 10,
                                       evidence_url="https://dataverse.harvard.edu/api/datasets/1") for i in range(4)]
            row = {"catalog_id": "000037", "provider": "dataverse", "source_id": "doi:test", "source_version": "1.1", "status": "resolved", "files": files}
            acquire.ingest(db, [row])
            for i, status in enumerate(("queued", "retry", "complete", "running")):
                db.execute("UPDATE downloads SET status=?,bytes_present=? WHERE relative_path=?", (status, i, str(i) + ".eeg"))
            acquire.ingest(db, [{**row, "catalog_id": "000038", "source_id": "doi:other", "files": [files[0]]}])
            db.execute("UPDATE downloads SET status='retry',error='HTTPError:status=400' WHERE catalog_id='000038'")
            db.commit()
            before = {r[0]: r for r in db.execute("SELECT id,status,bytes_present,error FROM downloads WHERE status IN ('complete','running') OR catalog_id='000038'")}
            evidence = {"probe_http_status": 400, "probe_response": {"status": "ERROR", "message": "You may not download this file without the required Guestbook response for guestbookID 217."},
                        "guestbook_id": 217, "dataset_source_url": "https://dataverse.harvard.edu/api/datasets/1", "published_file_ids": [0, 1, 2, 3],
                        "dataset_pid": "doi:test", "source_version": "1.1", "submission_performed": False}
            (p / "evidence.json").write_text(json.dumps(evidence))
            receipt = h.hold(p / "ledger.sqlite", p / "evidence.json")
            self.assertEqual(receipt["rows_changed"], 2)
            self.assertEqual(h.hold(p / "ledger.sqlite", p / "evidence.json")["rows_changed"], 0)
            after = {r[0]: r for r in db.execute("SELECT id,status,bytes_present,error FROM downloads WHERE status IN ('complete','running') OR catalog_id='000038'")}
            self.assertEqual(before, after)

    def test_generic_http400_is_not_sufficient_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(json.dumps({"probe_http_status": 400, "probe_response": {"status": "ERROR", "message": "bad request"},
                                        "dataset_source_url": "https://dataverse.harvard.edu/api/datasets/1"}))
            with self.assertRaises(ValueError):
                h.hold(Path(tmp) / "not-created.sqlite", path)
            self.assertFalse((Path(tmp) / "not-created.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
