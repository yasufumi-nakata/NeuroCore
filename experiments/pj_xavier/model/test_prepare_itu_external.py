import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import prepare_itu_external as external


PROTOCOL = json.loads(Path(external.__file__).with_name("itu_external_v3_protocol.json").read_text())


def clock(verified=True):
    return {"verified": verified, "max_error_ms": 1.0 if verified else 4302., "drift_ppm": 1.,
            "explicit_phase_channel_scale": 1, "anchor_count": 7, "anchor_span_s": 180.,
            "scale_eeg_per_behavior_second": 1., "offset_eeg_seconds": 0., "reason": "fixture"}


class ExternalCohortTests(unittest.TestCase):
    def test_canonical_ids_do_not_claim_distinct_humans(self):
        self.assertEqual(external.canonical_subject("itu", "sub-P001"), "itu:sub-P001")
        with self.assertRaises(ValueError): external.canonical_subject("itu", "uniba:sub-P001")
        value = external.compare_cross_site([{"site": "uniba", "subject": "uniba:sub-P001", "eeg_source_sha256": "a" * 64}],
                                            [{"site": "itu", "subject": "itu:sub-P001", "binary_sha256": "b" * 64}])
        self.assertEqual(value["shared_unqualified_participant_labels"], 1)
        self.assertFalse(value["human_identity_disjointness_verified"])
        self.assertEqual(value["exact_binary_hash_matches"], 0)

    def test_identical_raw_binary_across_sites_fails(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            external.compare_cross_site([{"site": "uniba", "subject": "uniba:sub-X", "eeg_source_sha256": "same"}],
                                        [{"site": "itu", "subject": "itu:sub-Y", "binary_sha256": "same"}])

    def test_clock_fail_cannot_be_promoted_or_rescaled(self):
        row = {"site": "itu", "partition": "external_test", "subject": "itu:sub-P012", "status": "blocked", "clock_sync": clock(False)}
        self.assertFalse(external.clock_eligible(row, PROTOCOL))
        row["status"] = "alignment_passed_external_candidate"
        with self.assertRaisesRegex(ValueError, "disagree"): external.clock_eligible(row, PROTOCOL)
        row["clock_sync"]["verified"] = True
        with self.assertRaisesRegex(ValueError, "thresholds"): external.clock_eligible(row, PROTOCOL)

    def test_current_phase_member_must_match_extraction_and_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);name = "data/01_raw/itu/sub-P001/phase.tsv";path = root / name
            path.parent.mkdir(parents=True);path.write_text("onset\n0\n")
            inventory = {name: {"sha256": external.digest(path), "bytes": path.stat().st_size}}
            external.checked_member(root, name, inventory, {})
            path.write_text("onset\n1\n")
            with self.assertRaisesRegex(ValueError, "changed"): external.checked_member(root, name, inventory, {})

    def test_future_history_and_exact_cutoff_are_excluded_by_frozen_helper(self):
        natural, _, _ = external.load_helpers(PROTOCOL)
        available = {k: True for k in natural.EVENT_KINDS}
        past = {k: np.array([1., 3., 9.]) for k in natural.EVENT_KINDS}
        changed = {k: np.array([1., 3., 9., 10., 10.1, 50.]) for k in natural.EVENT_KINDS}
        self.assertEqual(natural.history_features(past, available, 10.), natural.history_features(changed, available, 10.))

    def test_train_partition_or_postcutoff_history_is_rejected(self):
        row = {"site": "itu", "split": "external_test", "clock_uncertainty_guard_s": .01,
               "window_start_s": 3.25, "window_end_s": 5., "event_time_s": 9.52,
               "decision_time_s": 9.125, "environment_time_s": 3.,
               "history_pre_eeg": {"cutoff_s": 3.24, "max_source_onset_used_s": 3., "target_or_future_events_included": False},
               "history_pre_decision_guarded": {"cutoff_s": 9.115, "max_source_onset_used_s": 9., "target_or_future_events_included": False}}
        external.validate_timing(row)
        row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "partition"): external.validate_timing(row)
        row["split"] = "external_test";row["history_pre_eeg"]["max_source_onset_used_s"] = 3.24
        with self.assertRaisesRegex(ValueError, "Future"): external.validate_timing(row)

    def test_unsafe_or_cross_site_member_is_rejected(self):
        for name in ("../outside.eeg", "/absolute.eeg", "data/01_raw/uniba/source.eeg"):
            with self.subTest(name=name), self.assertRaises(ValueError): external.member_path(Path("/tmp"), name)

    @unittest.skipUnless(importlib.util.find_spec("mne"), "MNE is required for the real BrainVision fixture")
    def test_complete_cpu_fixture_has_external_only_and_all_record_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);bids = root / "bids";bids.mkdir()
            records=[];members=[]
            for number, passed in [(1, True), (12, False)]:
                sub=f"sub-P{number:03d}";stem=f"{sub}_ses-S001_task-programmingwithai_run-01_eeg"
                parent=Path("data/01_raw/itu") / sub / "ses-S001";eeg=bids/parent/"eeg";beh=bids/parent/"beh"
                eeg.mkdir(parents=True);beh.mkdir()
                header=eeg/(stem+".vhdr");binary=eeg/(stem+".eeg");marker=eeg/(stem+".vmrk")
                names=PROTOCOL["signal_confirmation"]["channels"]+["otree_phase"]
                data=np.full((51200,9),number*1e-6,dtype='<f4');data.tofile(binary)
                header.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile="+binary.name+"\nMarkerFile="+marker.name+"\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=9\nSamplingInterval=3906.25\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n"+"".join(f"Ch{i}={name},,1,V\n" for i,name in enumerate(names,1)))
                marker.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile="+binary.name+"\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")
                prefix=f"{sub}_ses-S001_task-programmingwithai_acq-"
                phase=beh/(prefix+"otree_run-01_events.tsv");phase.write_text("onset\ttrial_type\tai\n0\ttaskBegin\ttrue\n")
                (beh/(prefix+"gpt_run-01_events.tsv")).write_text("onset\ttrial_type\n80\tGPTSentPrompt\n")
                (beh/(prefix+"keypress_run-01_events.tsv")).write_text("onset\taction\n20\tdown\n40\tdown\n120\tdown\n150\tdown\n")
                (beh/(prefix+"mouseclick_run-01_events.tsv")).write_text("onset\n30\n50\n130\n160\n")
                for path in (bids/parent).rglob('*'):
                    if path.is_file():members.append({"path":str(path.relative_to(bids)),"sha256":external.digest(path),"bytes":path.stat().st_size})
                records.append({"site":"itu","partition":"external_test","subject":"itu:"+sub,"session":"ses-S001",
                    "header_member":str(header.relative_to(bids)),"header_sha256":external.digest(header),"binary_sha256":external.digest(binary),
                    "recording_duration_s":200.,"behavior_phase_member":str(phase.relative_to(bids)),"behavior_phase_sha256":external.digest(phase),
                    "status":"alignment_passed_external_candidate" if passed else "blocked","clock_sync":clock(passed)})
            protocol=copy.deepcopy(PROTOCOL);protocol['inclusion'].update(audit_recordings=2,clock_passed_recordings_expected=1)
            source={'verified':True,'sources':[{'filename':name,'sha256':protocol['frozen_inputs'][key],'ledger_status':'complete'} for name,key in [('bids_eeg_itu.tar.gz','itu_eeg_archive_sha256'),('bids_behavioral_itu.tar.gz','itu_behavior_archive_sha256')]]}
            external.write_json(bids/'source_verification.json',source);external.write_json(bids/'extracted_members.json',{'members':members})
            audit_path=root/'clock.json';external.write_json(audit_path,{'recordings':records})
            uniba=root/'uniba.jsonl';external.write_jsonl(uniba,[{'site':'uniba','subject':'uniba:sub-P001','eeg_source_sha256':'0'*64}])
            for key,path in [('itu_source_verification_sha256',bids/'source_verification.json'),('itu_extracted_members_sha256',bids/'extracted_members.json'),('itu_alignment_audit_sha256',audit_path),('uniba_v3_manifest_sha256',uniba)]: protocol['frozen_inputs'][key]=external.digest(path)
            protocol_path=root/'protocol.json';external.write_json(protocol_path,protocol)
            args=SimpleNamespace(bids_root=str(bids),output=str(root/'output'),protocol=str(protocol_path),expected_protocol_sha256=external.digest(protocol_path),parent_protocol=str(Path(external.__file__).with_name('natural_cohort_v3_protocol.json')),uniba_manifest=str(uniba),alignment_audit=str(audit_path),min_free_gib=0,run_token='fixture')
            before={str(p):external.digest(p) for p in bids.rglob('*') if p.is_file()}
            result=external.prepare(args)
            self.assertEqual(result['recordings_audited'],2);self.assertEqual(result['clock_blocked_recordings'],1)
            self.assertEqual(result['selected_windows'],5);self.assertEqual(result['counts']['train'],{});self.assertEqual(result['counts']['val'],{})
            rows=[json.loads(x) for x in (root/'output/manifest.jsonl').read_text().splitlines()]
            self.assertEqual({r['split'] for r in rows},{'external_test'})
            self.assertEqual({r['subject'] for r in rows},{'itu:sub-P001'})
            self.assertFalse(result['model_loaded']);self.assertFalse(result['training_started']);self.assertFalse(result['gpu_execution'])
            for path,value in before.items():self.assertEqual(external.digest(path),value)
            with self.assertRaisesRegex(ValueError,'already exists'):external.prepare(args)


if __name__ == '__main__':
    unittest.main()
