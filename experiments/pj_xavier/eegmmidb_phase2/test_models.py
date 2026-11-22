"""CPU-only tests for phase2 gradients, leakage guards and exact restart."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from eegmmidb_phase2 import data_contract as data
from eegmmidb_phase2 import model as models
from eegmmidb_phase2 import runtime
from eegmmidb_phase2 import train_models as train

torch.set_num_threads(2)


def fixture(root):
    root = Path(root)
    rows, artifacts = [], {}
    for part_index, part in enumerate(("train", "val", "test")):
        name = part + ".npy"
        values = np.random.default_rng(part_index).normal(0, 2e-5, (4, 64, 600)).astype(np.float32)
        np.save(root/name, values, allow_pickle=False)
        artifacts[name] = {"sha256": data.sha256(root/name), "bytes": (root/name).stat().st_size,
                           "shape": list(values.shape), "dtype": "float32"}
        source = hashlib.sha256(part.encode()).hexdigest()
        payload = hashlib.sha256(("payload" + part).encode()).hexdigest()
        index = 0
        for family, labels in data.TASKS.items():
            for label in labels:
                rows.append({
                    "id": f"{part}_{family}_{label}", "split": part,
                    "subject_group": f"physionet:eegmmidb:1.0.0:S{part_index+1:03d}",
                    "source_eeg_sha256": source, "source_sha256": source,
                    "eeg_payload_sha256": payload,
                    "task_family": family, "label": label, "label_source": "recorded_event",
                    "unit": "V", "sample_rate_hz": 200, "channels": [f"EEG{i}" for i in range(64)],
                    "eeg_path": name, "eeg_index": index,
                    "window_start_s": index*4., "window_end_s": index*4.+3,
                    "decision_time_s": index*4.+3, "feature_available_time_s": index*4.+2.995,
                    "preprocessing": {"online_causality_verified": True}, "annotation_review_required": False,
                })
                index += 1
    save_rows(root, rows)
    data.write_json(root/"artifacts.json", artifacts)
    for name in ("metadata_protocol.json","preprocessing_protocol.json","subject_split.json","corpus_summary.json"):
        data.write_json(root/name, {"fixture": True, "schema": name})
    return rows, artifacts


def save_rows(root, rows):
    Path(root, "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


def tiny_base():
    config = LlamaConfig(
        vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        attention_dropout=0.2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config)


class InputIdentityTests(unittest.TestCase):
    def test_complete_realistic_fixture_and_union(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root)
            rows, _, audit = data.load_corpus(Path(root)/"manifest.jsonl", Path(root)/"artifacts.json")
            self.assertEqual(audit["partition_counts"], {"train":4, "val":4, "test":4})
            self.assertEqual(data.source_interval_union(rows),36.)
            self.assertEqual(data.EEGStore().get(rows[0]).shape,(64,600))

    def test_cross_split_subject_source_and_payload_are_rejected(self):
        for key in ("subject_group","source_eeg_sha256","eeg_payload_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                rows,_=fixture(root)
                rows[4][key]=rows[0][key]
                if key=="source_eeg_sha256":
                    rows[4]["source_sha256"]=rows[0][key]
                save_rows(root,rows)
                with self.assertRaises(ValueError):
                    data.load_corpus(Path(root)/"manifest.jsonl",Path(root)/"artifacts.json")

    def test_duplicate_fractional_boolean_negative_slot_rejected(self):
        for value in (0,1.5,True,-1,4):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as root:
                rows,_=fixture(root)
                rows[1]["eeg_index"]=value
                save_rows(root,rows)
                with self.assertRaises(ValueError):
                    data.load_corpus(Path(root)/"manifest.jsonl",Path(root)/"artifacts.json")

    def test_waveform_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(root)
            values=np.load(Path(root)/"train.npy")
            values[0,0,0]+=1e-6
            np.save(Path(root)/"train.npy",values)
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                data.load_corpus(Path(root)/"manifest.jsonl",Path(root)/"artifacts.json")

    def test_unit_channel_and_future_availability_rejected(self):
        for key,value in (("unit","uV"),("channels",["X"]*64),("feature_available_time_s",100)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as root:
                rows,_=fixture(root)
                rows[0][key]=value
                save_rows(root,rows)
                with self.assertRaises(ValueError):
                    data.load_corpus(Path(root)/"manifest.jsonl",Path(root)/"artifacts.json")

    def test_training_batch_rejects_test_before_waveform_read(self):
        store=mock.Mock()
        with self.assertRaisesRegex(ValueError,"Test waveform"):
            train.batch_rows([{"split":"test"}],store,torch.device("cpu"))
        store.get.assert_not_called()

    def test_subject_macro_primary_selection_not_pooled(self):
        rows=[]
        probabilities=[]
        for subject,n,label,p in (("A",1,"left_fist",[.8,.2]),("A",1,"right_fist",[.8,.2]),
                                 ("B",10,"left_fist",[.8,.2]),("B",10,"right_fist",[.2,.8])):
            for _ in range(n):
                rows.append({"task_family":"left_right","subject_group":subject,"label":label})
                probabilities.append(p)
        for label,p in zip(data.TASKS["fists_feet"],([1.,0.],[0.,1.])):
            rows.append({"task_family":"fists_feet","subject_group":"A","label":label})
            probabilities.append(p)
        result=data.partition_metrics(rows,probabilities)
        self.assertAlmostEqual(result["selection_score"],.75)
        self.assertGreater(result["tasks"]["left_right"]["balanced_accuracy"],.9)


class GradientTests(unittest.TestCase):
    def test_cuda_allocator_receives_explicit_device_index(self):
        capacity = 24 * 1024**3
        with mock.patch.object(torch.cuda, "get_device_properties", return_value=argparse.Namespace(total_memory=capacity)), \
             mock.patch.object(torch.cuda, "current_device", return_value=0), \
             mock.patch.object(torch.cuda, "set_per_process_memory_fraction") as setter, \
             mock.patch.object(torch.cuda, "reset_peak_memory_stats"):
            result = runtime.apply_budget(torch.device("cuda"), 21)
        setter.assert_called_once_with(21 / 24, 0)
        self.assertEqual(result["budget_bytes"], 21 * 1024**3)

    def test_physical_scaling_and_full_channel_shape(self):
        encoder=models.EEGEncoder().eval()
        observed=[]
        handle=encoder.temporal.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
        x=torch.full((2,64,600),.0005)
        with torch.no_grad():
            y=encoder(x)
        handle.remove()
        self.assertEqual(y.shape,(2,18,32))
        self.assertTrue(torch.equal(observed[0],torch.full((2,1,64,600),5.)))
        with self.assertRaises(ValueError):
            encoder(x[:,:8])

    def test_explicit_adaptive_bins_match_reference(self):
        encoder=models.EEGEncoder().eval()
        x=torch.randn(2,64,600)*1e-5
        with torch.no_grad():
            reference=F.adaptive_avg_pool1d(encoder(x).transpose(1,2),8).transpose(1,2)
            actual=encoder.prefix_features(x)
        torch.testing.assert_close(actual,reference)

    def test_scratch_both_task_heads_and_encoder_receive_gradients(self):
        torch.manual_seed(3)
        model=models.ScratchDecoder().train()
        x=torch.randn(4,64,600)*1e-5
        labels=torch.tensor([0,1,0,1])
        logits=model(x,["left_right","left_right","fists_feet","fists_feet"])
        F.cross_entropy(logits,labels).backward()
        self.assertGreater(model.encoder.temporal.convolution.weight.grad.abs().sum().item(),0)
        for head in model.heads.values():
            self.assertGreater(head.weight.grad.abs().sum().item(),0)

    def test_frozen_llm_prefix_encoder_gradient_and_next_symbol_loss(self):
        torch.manual_seed(4)
        base=tiny_base()
        model=models.FusedDecoder(base).train()
        self.assertFalse(model.base.training)
        x=torch.randn(2,64,600)*1e-5
        logits=model(x,[[1,5,7,9],[1,5,10,11,12]])
        self.assertEqual(logits.shape,(2,2))
        labels=torch.tensor([0,1])
        loss=F.cross_entropy(logits,labels)
        expected=-torch.log_softmax(logits,-1)[torch.arange(2),labels].mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in base.parameters()))
        self.assertGreater(model.projector[1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.encoder.temporal.convolution.weight.grad.abs().sum().item(),0)

    def test_prompt_has_only_static_task_mapping(self):
        for task in data.TASKS:
            text=models.static_prompt(task)
            for forbidden in ("S001","T1","T2","trial_index","source_recording_id","window_start_s"):
                self.assertNotIn(forbidden,text)
        self.assertEqual(models.static_prompt("left_right","eeg_only"),models.static_prompt("fists_feet","eeg_only"))
        with self.assertRaises((ValueError,TypeError)):
            models.static_prompt({"task_family":"left_right","label":"left_fist"})


class RestartTests(unittest.TestCase):
    def basic_config(self):
        return {"kind":"scratch","architecture":models.ARCHITECTURE,"task_mapping":{k:list(v) for k,v in data.TASKS.items()}}

    def test_checkpoint_weights_only_and_exact_partial_epoch_next_step(self):
        with tempfile.TemporaryDirectory() as root:
            train.seed_everything(8)
            first=models.ScratchDecoder().train()
            optimizer=torch.optim.AdamW(first.parameters(),lr=.001)
            x=torch.randn(4,64,600)*1e-5
            task=["left_right","fists_feet","left_right","fists_feet"]
            y=torch.tensor([0,1,1,0])
            optimizer.zero_grad()
            F.cross_entropy(first(x,task),y).backward()
            optimizer.step()
            config=self.basic_config()
            state=train.training_state(config,"a"*64,0,4,1,None,None,{"a","b","c","d"},4,1.,False)
            path=Path(root)/"partial.pt"
            train.save_checkpoint(path,first,optimizer,state)
            raw=torch.load(path,map_location="cpu",weights_only=True)
            self.assertEqual(raw["next_position"],4)
            self.assertFalse(raw["epoch_completed"])
            optimizer.zero_grad()
            F.cross_entropy(first(x,task),y).backward()
            optimizer.step()
            expected=train.clone_cpu(first.state_dict())
            second=models.ScratchDecoder().train()
            other=torch.optim.AdamW(second.parameters(),lr=.001)
            restored=train.restore_training(path,second,other,config,"a"*64)
            self.assertEqual(restored["seen_train_ids"],["a","b","c","d"])
            other.zero_grad()
            F.cross_entropy(second(x,task),y).backward()
            other.step()
            for key,value in second.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)

    def test_resume_without_improvement_preserves_old_best_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            first,second=Path(root)/"first.pt",Path(root)/"second.pt"
            first.write_bytes(b"old-best")
            second.write_bytes(b"new-worse")
            score,old_hash,changed=train.update_best(root,first,.75,None)
            self.assertTrue(changed)
            score,new_hash,changed=train.update_best(root,second,.74,score)
            self.assertFalse(changed)
            self.assertEqual(old_hash,new_hash)
            self.assertEqual((Path(root)/"best.pt").read_bytes(),b"old-best")
            self.assertEqual(train.epoch_order(10,20261122,2)[4:],train.epoch_order(10,20261122,2)[4:])

    def test_source_or_config_tampering_and_inconsistent_restart_rejected(self):
        for modification in ("source","config","inconsistent"):
            with self.subTest(modification=modification),tempfile.TemporaryDirectory() as root:
                model=models.ScratchDecoder()
                optimizer=torch.optim.AdamW(model.parameters())
                config=self.basic_config()
                state=train.training_state(config,"a"*64,0,0,0,None,None,set(),0,0,False)
                path=Path(root)/"partial.pt"
                train.save_checkpoint(path,model,optimizer,state)
                raw=torch.load(path,map_location="cpu",weights_only=True)
                if modification=="source":raw["source_sha256"]["model.py"]="0"*64
                if modification=="config":raw["config"]["kind"]="fused"
                if modification=="inconsistent":raw["resumable"]=False
                torch.save(raw,path)
                with self.assertRaises(ValueError):
                    train.restore_training(path,model,optimizer,config,"a"*64)

    def test_small_real_training_loop_never_reads_test_and_saves_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            prepared=root/"prepared";prepared.mkdir()
            fixture(prepared)
            plan=train.read_json(Path(__file__).with_name("protocol.training.v1.json"))
            plan["stages"]["scratch"].update(max_epochs=2,capacity_batch_candidates=[4])
            protocol=root/"protocol.json";data.write_json(protocol,plan)
            contract_path=root/"contract.json"
            contract=train.register(prepared,protocol,contract_path)
            report={"kind":"scratch","execution_contract_sha256":data.sha256(contract_path),
                    "source_sha256":runtime.source_seals(),"batch_size":4,"status":"passed",
                    "completed_optimizer_steps":50}
            receipt=root/"capacity.json";data.write_json(receipt,report)
            args=argparse.Namespace(command="train",prepared=str(prepared),contract=str(contract_path),
                  run_dir=str(root/"run"),kind="scratch",device="cpu",scratch_run=None,
                  local_files_only=True,absolute_deadline_epoch=None,capacity_reports=[str(receipt)],resume=None)
            actual_get=data.EEGStore.get
            calls=[]
            def monitored_get(store,row):
                calls.append(row["split"])
                return actual_get(store,row)
            with mock.patch.object(data.EEGStore,"get",monitored_get):
                result=train.train(args)
            self.assertEqual(result["status"],"completed")
            self.assertEqual(result["completed_epochs"],2)
            self.assertEqual(result["optimizer_steps"],2)
            self.assertEqual(result["unique_source_interval_union_seconds"],12.)
            self.assertNotIn("test",calls)
            self.assertEqual(set(calls),{"train","val"})
            state=torch.load(root/"run/best.pt",weights_only=True,map_location="cpu")
            self.assertEqual(state["config"],train.read_json(root/"run/run_config.json"))
            self.assertEqual(state["training_membership_sha256"],data.sha256(root/"run/training_membership.json"))
            self.assertEqual(len(list((root/"run").glob("epoch_*.pt"))),2)
            # Exact state-dict restoration does not change eval logits.
            first=models.ScratchDecoder().eval();second=models.ScratchDecoder().eval()
            models.restore_learned(first,"scratch",state["learned"])
            models.restore_learned(second,"scratch",state["learned"])
            x=torch.randn(2,64,600)*1e-5
            torch.testing.assert_close(first(x,["left_right","fists_feet"]),second(x,["left_right","fists_feet"]),rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()

