"""Causality, calibration, partition, timing and provenance regression tests."""
import io
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import prep_causal_v2 as p


def synthetic_edf(native_fs=160):
    # Three seconds; differing per-channel digital/physical calibration and a TAL channel.
    def field(value, n):
        return str(value).encode().ljust(n)
    duration, ns, nrecords = 1, 65, 3
    prefix = b"0       " + b"".ljust(80) + b"".ljust(80) + b"01.01.00" + b"00.00.00"
    prefix += field(256*(ns+1), 8) + b"EDF+C".ljust(44) + field(nrecords,8) + field(duration,8) + field(ns,4)
    labels = ["EEG"+str(i) for i in range(64)] + ["EDF Annotations"]
    fields = [(labels,16), ([""]*ns,80), (["uV"]*64+[""],8),
              ([-200]*64+[-1],8), ([400]*64+[1],8), ([-10000]*64+[-32768],8),
              ([20000]*64+[32767],8), ([""]*ns,80), ([native_fs]*64+[32],8), ([""]*ns,32)]
    header = prefix + b"".join(field(v,w) for vals,w in fields for v in vals)
    payload = bytearray()
    eeg_hash = hashlib.sha256()
    expected = np.zeros((64,native_fs*nrecords), dtype=np.float64)
    for r in range(nrecords):
        for c in range(64):
            digital = (np.arange(native_fs) + r*3 + c*7).astype("<i2")
            raw = digital.tobytes()
            eeg_hash.update(raw)
            payload.extend(raw)
            expected[c,r*native_fs:(r+1)*native_fs] = digital.astype(float)*.02*1e-6
        payload.extend(b"+0\x14\x14\x00".ljust(64,b"\x00"))
    data = header + payload
    record = {"member":"S001/S001R04.edf", "source_recording_id":"fixture",
              "source_eeg_sha256":hashlib.sha256(data).hexdigest(), "eeg_payload_sha256":eeg_hash.hexdigest(),
              "header_sha256":hashlib.sha256(header).hexdigest(), "channels":labels[:64],
              "native_sample_rate_hz":native_fs, "source_seconds":3.0}
    return data, record, expected


class TestCausalPreparation(unittest.TestCase):
    def test_future_changes_do_not_change_past_at_both_native_rates(self):
        rng = np.random.default_rng(8)
        for fs in (160,128):
            with self.subTest(fs=fs):
                x = rng.normal(0,10e-6,(4,fs*8))
                future = x.copy()
                cutoff = fs*4+3
                future[:,cutoff:] = rng.normal(0,100,(4,x.shape[1]-cutoff))
                nout = (cutoff*200+fs-1)//fs
                a, b = p.causal_stream(x,fs), p.causal_stream(future,fs)
                np.testing.assert_array_equal(a[:,:nout],b[:,:nout])
                np.testing.assert_array_equal(a[:,:nout],p.causal_stream(x[:,:cutoff],fs))

    def test_impulse_cannot_precede_input(self):
        for fs in (160,128):
            x=np.zeros((1,fs*4)); x[0,fs*2+1]=1
            y=p.causal_stream(x,fs)
            before=(fs*2+1)*200//fs
            self.assertTrue(np.all(y[:,:before]==0))
            self.assertTrue(np.any(y[:,before:]!=0))

    def test_EDF_conversion_excludes_annotations_and_verifies_payload(self):
        helper=p.header_helper()
        data,record,expected=synthetic_edf()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"raw.zip"
            with zipfile.ZipFile(path,"w") as z:z.writestr("files/"+record["member"],data)
            with zipfile.ZipFile(path) as z:
                decoded,fs,_=p.decode_member(z,record,helper)
                np.testing.assert_allclose(decoded,expected,rtol=0,atol=3e-20)
                self.assertEqual(fs,160)
                bad={**record,"eeg_payload_sha256":"0"*64}
                with self.assertRaisesRegex(ValueError,"payload"):
                    p.decode_member(z,bad,helper)

    def test_MNE_independent_unit_conversion_for_fixture(self):
        import mne
        data,_,expected=synthetic_edf()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"fixture.edf";path.write_bytes(data)
            got=mne.io.read_raw_edf(path,preload=True,verbose="ERROR").get_data()
            np.testing.assert_allclose(got,expected,rtol=0,atol=3e-20)

    def test_short_annotation_not_repaired_and_grid_is_ceiled(self):
        short=p.interval(20,2.975,30)
        self.assertIn("window_exceeds_annotation",short["reasons"])
        span=p.interval(10.001,4.1,20)
        self.assertEqual(span["sample_start"],2201)
        self.assertAlmostEqual(span["grid_shift_s"],.004)
        self.assertFalse(span["reasons"])

    def test_control_block_does_not_remove_main(self):
        main=p.interval(4.2,4.1,125)
        shifted=p.interval(4.2,4.1,125,shifted=True)
        self.assertFalse(main["reasons"])
        self.assertIn("insufficient_past_filter_history",shifted["reasons"])

    def test_native_dependency_bound_is_no_future(self):
        span=p.interval(10.001,4.1,125)
        for fs in(160,128):
            t=p.timing(span,fs,"test")
            self.assertLessEqual(t["causal_history"]["last_sample_time_s"],t["feature_available_time_s"])
            self.assertEqual(t["raw_interval"]["start_s"],span["window_start_s"])
            self.assertEqual(t["causal_history"]["native_sample_start"],0)

    def test_QC_maximum_absolute_not_peak_to_peak(self):
        x=np.tile(80e-6*np.sin(np.arange(600)*2*np.pi/50),(64,1))
        self.assertTrue(p.qc(x)["passed"])
        x[2,1]=101e-6
        self.assertTrue(p.qc(x)["passed"])
        self.assertTrue(p.qc(x)["quality_flag_above_initial_100uV"])
        x[2,1]=501e-6
        self.assertEqual(p.qc(x)["reasons"],["absolute_amplitude_above_500uV"])
        x[2,:]=0
        self.assertIn("channel_population_std_below_0p01uV",p.qc(x)["reasons"])
        x[4,4]=np.nan
        self.assertEqual(p.qc(x)["reasons"],["nonfinite"])

    def test_hash_split_counts_and_all_runs_grouped(self):
        a=p.split_assignments()
        from collections import Counter
        self.assertEqual(Counter(a.values()),{"train":76,"val":16,"test":17})
        self.assertEqual(a,p.split_assignments())
        self.assertEqual(a["physionet:eegmmidb:1.0.0:S001"],"train")

    def test_main_and_control_artifacts_are_distinct_slots(self):
        x=np.zeros((64,600),np.float32)
        with tempfile.TemporaryDirectory() as d:
            meta={"CPU_absolute_deadline_epoch":p.time.time()+60,"incremental_output_budget_bytes":10**7,
                  "disk_reserve_bytes":0,"reserved_future_bulk_growth_bytes":0}
            a=p.save_array(d,"waveforms/r.npy",[x,x+1],meta,d)
            b=p.save_array(d,"controls/r.npy",[x],meta,d)
            self.assertEqual(a["shape"],[2,64,600]);self.assertEqual(b["shape"],[1,64,600])
            with self.assertRaisesRegex(ValueError,"Existing prepared"):
                p.save_array(d,"waveforms/r.npy",[x],meta,d)

    def test_original_deadline_and_budget_are_hard_limits(self):
        with tempfile.TemporaryDirectory() as d:
            meta={"CPU_absolute_deadline_epoch":0,"incremental_output_budget_bytes":10,
                  "disk_reserve_bytes":0,"reserved_future_bulk_growth_bytes":0}
            with self.assertRaisesRegex(ValueError,"deadline"):p.guard(meta,d)
            meta["CPU_absolute_deadline_epoch"]=p.time.time()+20
            with self.assertRaisesRegex(ValueError,"budget"):p.guard(meta,d,11)

    def test_immutable_write_and_interval_union(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"x.json";p.save_json(path,{"a":1})
            with self.assertRaisesRegex(ValueError,"Immutable"):p.save_json(path,{"a":2})
            self.assertEqual(json.loads(path.read_text()),{"a":1})
        rows=[{"source_eeg_sha256":"a","window_start_s":a,"window_end_s":b} for a,b in [(0,3),(2,5),(8,11)]]
        self.assertEqual(p.union_seconds(rows),8)


if __name__=="__main__":unittest.main()
