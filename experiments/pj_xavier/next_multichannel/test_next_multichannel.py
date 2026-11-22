"""Focused CPU contract tests for the next PJ-Xavier multichannel stage."""
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import admission
import causal_shards as cs
import joint_alignment as ja


class TestAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=admission.REPO/"private/xavier/phase4_automation_1615/source_admission.private.jsonl"
        with cls.manifest.open() as fh:
            cls.rows=[json.loads(line) for line in fh]

    def test_no_premature_admission(self):
        self.assertGreaterEqual(len(self.rows),1000)
        for r in self.rows:
            self.assertFalse(r["fit_allowed"])
            self.assertFalse(r["validation_gold_allowed"])
            self.assertFalse(r["test_gold_allowed"])
            self.assertFalse(r["raw_label"]["original_supervised_gold"])
            self.assertIsNone(r["ai_weak_label"]["value"])

    def test_deterministic_must_link(self):
        groups=defaultdict(set)
        for r in self.rows:
            groups[r["component_id_sha256"]].add(r["metadata_only_proposed_split"])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        self.assertLess(len(groups),len(self.rows))

    def test_old_split_kept_untouched(self):
        d=json.loads((admission.REPO/admission.EXISTING_SPLIT).read_text())
        self.assertEqual({"train":76,"val":16,"test":17},
                         {x:list(d["assignments"].values()).count(x) for x in ("train","val","test")})

    def test_explicit_cross_id_duplicate_pairs(self):
        d=json.loads((admission.REPO/admission.FIF_MUST_LINK).read_text())
        by_group={r["source_group"]:r["component_id_sha256"] for r in self.rows}
        for p in d["pairs"]:
            l="small_bci_mixed|"+p["left_group"]
            rr="small_bci_mixed|"+p["right_group"]
            self.assertEqual(by_group[l],by_group[rr])

    def test_label_input_separation(self):
        for r in self.rows:
            if r["source"]=="p30032":
                self.assertIn("TextToSpell",r["raw_label"]["oracle"])
                self.assertIn("SelectedTarget",r["raw_label"]["decoder_output"])
                self.assertFalse(r["validation_gold_allowed"])

    def test_sha_attestation(self):
        a=json.loads((admission.REPO/"experiments/pj_xavier/reports/next_multichannel_admission_20261122_1615.aggregate.json").read_text())
        self.assertEqual(a["private_manifest_sha256"],admission.file_digest(self.manifest))
        self.assertEqual(a["records_checked"],len(self.rows))


class TestCausalShards(unittest.TestCase):
    def montage(self,c,fs):
        return cs.Montage(tuple("E"+str(i) for i in range(c)),"average",fs,"raw-native-confirmed")

    def processor(self,m):
        return cs.CausalRecordStream(m,unit_is_verified_physical_volts=True,source_is_raw_causal=True)

    def test_all_channel_shapes_and_causality(self):
        rng=np.random.default_rng(1729)
        for channels in (14,16,27,32,64):
            for fs in (128,160,256,512):
                with self.subTest(channels=channels,fs=fs):
                    m=self.montage(channels,fs)
                    x=rng.normal(0,4e-6,size=(channels,fs*4)).astype(np.float64)
                    a=self.processor(m)
                    whole=a.push(x)
                    b=self.processor(m)
                    parts=[]
                    offsets=[0,7,fs//2,fs+9,fs*2-3,fs*4]
                    for p,q in zip(offsets[:-1],offsets[1:]):
                        parts.append(b.push(x[:,p:q]))
                    pieces=[np.concatenate([part[i] for part in parts],axis=-1) for i in range(3)]
                    np.testing.assert_array_equal(whole[1],pieces[1])
                    np.testing.assert_array_equal(whole[2],pieces[2])
                    np.testing.assert_allclose(whole[0],pieces[0],rtol=1e-5,atol=1e-10)
                    self.assertEqual(a.next_j,b.next_j)
                    self.assertTrue(np.all(whole[2] <= np.floor(whole[1]*fs/200)))
                    self.assertTrue(np.all(whole[2] < x.shape[1]))
                    self.assertTrue(np.all(whole[2] >= fs*2))

    def test_no_future_dependency(self):
        m=self.montage(14,256)
        x=np.ones((14,256*4),dtype=np.float64)*2e-6
        a=self.processor(m).push(x)
        altered=x.copy()
        altered[:,-20:]*=7
        b=self.processor(m).push(altered)
        earlier=a[2]<x.shape[1]-20
        np.testing.assert_array_equal(a[0][:,earlier],b[0][:,earlier])

    def test_fail_closed(self):
        m=self.montage(14,256)
        with self.assertRaises(ValueError):
            cs.CausalRecordStream(m,unit_is_verified_physical_volts=False,source_is_raw_causal=True)
        with self.assertRaises(ValueError):
            cs.CausalRecordStream(m,unit_is_verified_physical_volts=True,source_is_raw_causal=False)
        with self.assertRaises(ValueError):
            self.processor(m).push(np.ones((13,256)))
        with self.assertRaises(ValueError):
            self.processor(m).push(np.full((14,256),np.nan))
        with self.assertRaises(ValueError):
            cs.Montage(tuple("E"+str(i) for i in range(13)),"average",256,"raw")

    def test_same_montage_bucket_and_patch(self):
        m=self.montage(14,256)
        v,j,n=self.processor(m).push(np.ones((14,256*7))*4e-6)
        s=cs.patch_shards(v,j,n,m)
        self.assertGreaterEqual(len(s),1)
        stack=cs.assert_same_montage_batch(s)
        self.assertEqual(stack.shape[1:],(14,3,200))
        other=dict(s[0],montage_key="other")
        with self.assertRaises(ValueError):
            cs.assert_same_montage_batch([s[0],other])

    def test_bounded_stream_assembler(self):
        m=self.montage(16,512)
        x=np.random.default_rng(73).normal(0,5e-6,size=(16,512*9))
        expected_v,expected_j,expected_n=self.processor(m).push(x)
        expected=cs.patch_shards(expected_v,expected_j,expected_n,m)
        proc=self.processor(m)
        assembler=cs.StreamingShardAssembler(m)
        got=[]
        begin=0
        while begin<x.shape[1]:
            end=min(x.shape[1],begin+137)
            got.extend(assembler.append(*proc.push(x[:,begin:end])))
            begin=end
        self.assertEqual(len(expected),len(got))
        for a,b in zip(expected,got):
            self.assertEqual(a["target_last"],b["target_last"])
            np.testing.assert_allclose(a["data_V"],b["data_V"],atol=1e-10,rtol=1e-5)
        self.assertLess(assembler.pending_y.shape[1],600)
        with self.assertRaises(ValueError):
            assembler.append(np.zeros((16,1)),np.array([assembler.last_target_index+2]),np.array([512*9]))

    def test_CBraMod_one_scale_and_range_rejection(self):
        x=np.ones((14,3,200),dtype=np.float32)*4e-6
        self.assertAlmostEqual(float(cs.cbramod_encoder_input(x,physical_unit_verified=True)[0,0,0]),0.04,places=6)
        with self.assertRaises(ValueError):
            cs.cbramod_encoder_input(x,physical_unit_verified=False)
        with self.assertRaises(ValueError):
            cs.cbramod_encoder_input(np.ones((14,3,200))*0.025,physical_unit_verified=True)

    @unittest.skipIf(cs.torch is None,"PyTorch unavailable")
    def test_resampler_gradients_all_channels(self):
        torch=cs.torch
        mod=cs.ChannelQueryResampler(32,8,4)
        for c in (14,16,27,32,64):
            x=torch.randn(2,c,3,32,requires_grad=True)
            y=mod(x,montage_keys=["same_verified_montage"]*2)
            self.assertEqual(tuple(y.shape),(2,8,32))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            with self.assertRaises(ValueError):
                mod(x,montage_keys=["montage_A","montage_B"])
            mod.zero_grad(set_to_none=True)


class TestAlignment(unittest.TestCase):
    def test_counterbalanced_original_not_four_independent_eeg(self):
        a=ja.make_counterbalanced_trials("train-only-synthetic-trial","A",allowed_split="train")
        b=ja.make_counterbalanced_trials("train-only-synthetic-trial","B",allowed_split="train")
        self.assertEqual(ja.verify_train_counterbalance(a)["independent_signal_multiplier"],1)
        self.assertEqual(len(a),4)
        self.assertEqual(sum(r["synthetic_mapping_action_gold"]=="X" for r in a),2)
        self.assertEqual(sum(r["synthetic_mapping_action_gold"]=="Y" for r in a),2)
        for r in a+b:
            self.assertNotIn("original_source_cue_gold",r["input_environment"])
            self.assertNotIn("synthetic_mapping_action_gold",r["input_environment"])
            ja.validate_input_context(r["input_environment"])
            self.assertIsNone(r["free_intent_gold"])

    def test_context_fail_closed_and_controls(self):
        x=ja.mapping("trial","normal")
        with self.assertRaises(ValueError):
            ja.validate_input_context(dict(x,target="X"))
        with self.assertRaises(ValueError):
            ja.synthetic_action_gold("spontaneous_intent",x)
        self.assertIn("context_only",ja.BENCHMARK_SPEC["negative_controls"])
        self.assertIn("T0_idle_diagnostic_not_natural_idle_false_actions_per_min",ja.BENCHMARK_SPEC["negative_controls"])
        self.assertEqual(ja.BENCHMARK_SPEC["new_model_performance"],"NOT_EVALUATED")


if __name__=="__main__":
    unittest.main()
