"""Phase3 continuous train/validation EEG preparation; no cue/action labels.

The phase2 operator and header parser are immutable dependencies. Test members
are never opened by this program. A new bounded resource receipt is mandatory.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

PHASE2_OPERATOR_SHA = "99ce752a91043a6e1b0b9a89d7c984d6fead9478076a8fb555747c123a758d95"
DATASET = "physionet:eegmmidb:1.0.0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    write_bytes(path, (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def write_bytes(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".part")
    require(not path.exists() and not temp.exists(), "Existing immutable output: "+str(path))
    with temp.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def write_rows(path, rows):
    write_bytes(path,b"".join((json.dumps(r,sort_keys=True,allow_nan=False)+"\n").encode() for r in rows))


def load_operator(path):
    require(sha(path)==PHASE2_OPERATOR_SHA,"Pinned phase2 causal operator changed")
    spec=importlib.util.spec_from_file_location("phase3_fixed_phase2_operator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def continuous_spans(recording_seconds):
    # Fixed grid anchored at recording +5s; never at an annotation/cue.
    n=max(0,math.floor((recording_seconds-5)/3))
    return [{"sample_start":1000+600*k,"sample_stop":1600+600*k,
             "window_start_s":5.0+3*k,"window_end_s":8.0+3*k,
             "grid_shift_s":0.0,"reasons":[]} for k in range(n)]


def validate_allowlist(rows, denied, split_assignments):
    require(rows,"Empty continuous source allowlist")
    ids=set();sources={};payloads={};groups=defaultdict(set)
    forbidden_ids={r["source_recording_id"] for r in denied}
    forbidden_sha={r["source_eeg_sha256"] for r in denied}
    forbidden_payload={r["eeg_payload_sha256"] for r in denied}
    for row in rows:
        require(row["split"] in ("train","val"),"Test is forbidden in SSL preparation")
        require(row["status"]=="continuous_SSL_source_allowed","Unapproved source role")
        require(row["source_recording_id"] not in ids and row["source_recording_id"] not in forbidden_ids,"Repeated/denied recording")
        ids.add(row["source_recording_id"])
        require(row["source_eeg_sha256"] not in forbidden_sha and row["eeg_payload_sha256"] not in forbidden_payload,"Denied raw/payload in allowlist")
        require(row["split"]==split_assignments[row["subject_group"]],"Original participant partition changed")
        require(row["run"] in range(1,15),"Run outside original full14")
        require(len(row["channels"])==len(set(row["channels"]))==64,"64 unique channels required")
        require(not {"events","label","task_family","gold_annotation_code"}&set(row),"Supervision metadata must not enter SSL source manifest")
        groups[row["subject_group"]].add(row["split"])
        for key, mapping in (("source_eeg_sha256",sources),("eeg_payload_sha256",payloads)):
            old=mapping.setdefault(row[key],row["split"])
            require(old==row["split"],"Original signal crosses train/validation")
    require(all(len(x)==1 for x in groups.values()),"Participant crosses partitions")
    return {"records":len(rows),"record_counts":dict(Counter(r["split"] for r in rows)),
            "groups":{s:len({r["subject_group"] for r in rows if r["split"]==s}) for s in ("train","val")}}


def guard(receipt, output, required=0):
    require(receipt["status"]=="authorized_registered", "New resource receipt not registered")
    require(time.time()<receipt["absolute_deadline_epoch"],"Phase3 absolute deadline reached")
    usage=sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file()) if Path(output).exists() else 0
    require(usage+required<=receipt["max_output_bytes"],"Phase3 derived-byte budget exceeded")
    parent=Path(output) if Path(output).exists() else Path(output).parent
    require(shutil.disk_usage(parent).free-required>=receipt["disk_reserve_bytes"],"Phase3 free-space reserve crossed")


def run(args):
    started=time.time()
    source_root=Path(args.contract_dir);output=Path(args.output)
    require(not output.exists(),"Output already exists; do not overwrite a completed or failed attempt")
    proto=json.loads((source_root/"metadata_protocol.json").read_text())
    receipt=json.loads(Path(args.resource_receipt).read_text())
    require(proto["status"]=="frozen_metadata_before_continuous_signal_processing", "Metadata contract not frozen")
    require(receipt["protocol_sha256"]==sha(source_root/"metadata_protocol.json"),"Budget authorizes a different metadata contract")
    require(receipt["prep_source_sha256"]==sha(__file__),"Budget authorizes a different preparation source")
    require(receipt["max_output_bytes"]>=proto["capacity_upper_bounds"]["array_payload_bytes"],"Budget below conservative data payload upper bound")
    require(receipt["cpu_threads"]==2,"CPU thread limit must remain2")
    output.parent.mkdir(parents=True,exist_ok=True)
    guard(receipt,output)
    for name,key in (("source_allowlist.jsonl","source_allowlist_sha256"),("source_denylist.jsonl","source_denylist_sha256"),("subject_split.json","subject_split_sha256")):
        require(sha(source_root/name)==proto[key],"Frozen metadata bytes changed")
    records=[json.loads(x) for x in (source_root/"source_allowlist.jsonl").read_text().splitlines()]
    denied=[json.loads(x) for x in (source_root/"source_denylist.jsonl").read_text().splitlines()]
    split=json.loads((source_root/"subject_split.json").read_text())["assignments"]
    allow_audit=validate_allowlist(records,denied,split)
    require(len(records)==1238 and len(denied)==288 and allow_audit["record_counts"]=={"train":1014,"val":224},"Full fixed source coverage differs")
    require(len({r["source_recording_id"] for r in records+denied})==1526,"Original record coverage differs")
    op=load_operator(args.phase2_source);helper=op.header_helper()
    require(op.MAX_ABS_V==5e-4 and op.MIN_STD_V==1e-8,"Fixed QC thresholds differ")
    archive=Path(args.archive)
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==proto["source_archive_bytes"],"Original ZIP size differs")
    require(sha(archive)==proto["source_archive_sha256"],"Original whole ZIP SHA differs")
    # Exact runtime evidence is sealed before any signal/member decoding.
    output.mkdir()
    runtime={"schema":"pj-xavier-phase3-preparation-runtime-v1","status":"sealed_before_member_decoding",
             "protocol_sha256":sha(source_root/"metadata_protocol.json"),"resource_receipt_sha256":sha(args.resource_receipt),
             "prep_source_sha256":sha(__file__),"phase2_operator_sha256":PHASE2_OPERATOR_SHA,
             "python":sys.version.split()[0],"numpy":np.__version__,"scipy":scipy.__version__,
             "versions_may_differ_from_phase2_but_downstream_phase2_arrays_remain_unchanged":True,
             "created_epoch":time.time()}
    # Real numeric causality verification under this runtime, before raw decoding.
    for fs in (128,160):
        rng=np.random.default_rng(20261122);x=rng.normal(0,1e-5,(2,fs*6));cutoff=fs*3+1
        modified=x.copy();modified[:,cutoff:]=rng.normal(0,1,(2,x.shape[1]-cutoff))
        n=(cutoff*200+fs-1)//fs
        y=op.causal_stream(x,fs)
        require(np.array_equal(y[:,:n],op.causal_stream(modified,fs)[:,:n]),"Future modification affected past")
        require(np.array_equal(y[:,:n],op.causal_stream(x[:,:cutoff],fs)),"Prefix invariance failed")
    runtime["future_and_prefix_invariance_native_rates"]=[128,160]
    write_json(output/"runtime_seal.json",runtime)
    runtime_sha=sha(output/"runtime_seal.json")
    manifest,artifacts,source_ledger,rejections=[],{},[],[]
    with zipfile.ZipFile(archive) as z,threadpool_limits(limits=2):
        for rec in records:
            guard(receipt,output)
            raw,fs,evidence=op.decode_member(z,rec,helper)
            stream=op.causal_stream(raw,fs)
            accepted,rows=[],[]
            relative="waveforms/"+Path(rec["member"]).stem+".npy"
            for span in continuous_spans(rec["source_seconds"]):
                x,quality=op.evaluate_window(stream,span)
                identity=rec["source_recording_id"]+":continuous"+str(span["sample_start"])
                if x is None:
                    rejections.append({"id":identity,"split":rec["split"],"source_recording_id":rec["source_recording_id"],**quality})
                    continue
                rows.append({"id":identity,"dataset":DATASET,"subject_group":rec["subject_group"],"split":rec["split"],
                    "source_recording_id":rec["source_recording_id"],"source_eeg_sha256":rec["source_eeg_sha256"],
                    "source_sha256":rec["source_eeg_sha256"],"eeg_payload_sha256":rec["eeg_payload_sha256"],
                    "channels":rec["channels"],"sample_rate_hz":200,"native_sample_rate_hz":fs,"unit":"V",
                    "eeg_path":relative,"eeg_index":len(accepted),"QC":quality,
                    "data_origin":"real_recording","target_kind":"self_supervised_waveform_reconstruction",
                    "preprocessing":{"online_causality_verified":True,"runtime_seal_sha256":runtime_sha,
                        "protocol_sha256":runtime["protocol_sha256"],"future_samples_used":False,"normalization":"none"},
                    **op.timing(span,fs,rec["source_recording_id"])})
                accepted.append(x)
            if accepted:
                array=np.stack(accepted).astype(np.float32,copy=False)
                guard(receipt,output,array.nbytes+4096)
                target=output/relative;target.parent.mkdir(exist_ok=True)
                temp=target.with_suffix(".part")
                with temp.open("xb") as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
                os.link(temp,target);temp.unlink()
                artifacts[relative]={"sha256":sha(target),"bytes":target.stat().st_size,"shape":list(array.shape),"dtype":"float32"}
                for row in rows:row["array_sha256"]=artifacts[relative]["sha256"]
            manifest.extend(rows)
            source_ledger.append({**evidence,"split":rec["split"],"subject_group":rec["subject_group"],
                                 "original_seconds":rec["source_seconds"],"candidates":len(continuous_spans(rec["source_seconds"])),"accepted":len(rows)})
            if len(source_ledger)%50==0:
                print(json.dumps({"stage":"continuous_SSL_preparation","records":len(source_ledger),"windows":len(manifest),"elapsed_s":time.time()-started}),flush=True)
        for rec in records:
            guard(receipt,output)
            h=hashlib.sha256()
            with z.open("files/"+rec["member"]) as f:
                for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
            require(h.hexdigest()==rec["source_eeg_sha256"],"Original allowed EDF member final SHA differs")
    require(sha(archive)==proto["source_archive_sha256"],"Whole ZIP final SHA differs")
    write_rows(output/"manifest.jsonl",manifest);write_json(output/"artifacts.json",artifacts)
    write_rows(output/"source_records.jsonl",source_ledger);write_rows(output/"rejected_windows.jsonl",rejections)
    for name in ("metadata_protocol.json","source_allowlist.jsonl","source_denylist.jsonl","subject_split.json"):
        write_bytes(output/name,(source_root/name).read_bytes())
    write_bytes(output/"resource_receipt.json",Path(args.resource_receipt).read_bytes())
    summary={"schema":"pj-xavier-phase3-continuous-SSL-corpus-v1","status":"completed",
        "protocol_sha256":runtime["protocol_sha256"],"runtime_seal_sha256":runtime_sha,
        "prep_source_sha256":sha(__file__),"source_archive_before_after_SHA":proto["source_archive_sha256"],
        "all_allowed_members_before_after_SHA_verified":True,"source_counts":allow_audit,
        "source_original_hours":{s:sum(r["source_seconds"] for r in records if r["split"]==s)/3600 for s in ("train","val")},
        "window_counts":dict(Counter(r["split"] for r in manifest)),
        "unique_target_interval_hours":{s:sum(r["window_end_s"]-r["window_start_s"] for r in manifest if r["split"]==s)/3600 for s in ("train","val")},
        "participant_groups_with_windows":{s:len({r["subject_group"] for r in manifest if r["split"]==s}) for s in ("train","val")},
        "accepted_windows":len(manifest),"rejected_windows":len(rejections),
        "rejection_reason_counts":dict(Counter(x for r in rejections for x in r["reasons"])),
        "quality_flag_above_initial_100uV":dict(Counter(r["split"] for r in manifest if r["QC"]["quality_flag_above_initial_100uV"])),
        "array_bytes":sum(x["bytes"] for x in artifacts.values()),"array_count":len(artifacts),
        "manifest_sha256":sha(output/"manifest.jsonl"),"artifacts_sha256":sha(output/"artifacts.json"),
        "source_allowlist_sha256":proto["source_allowlist_sha256"],"source_denylist_sha256":proto["source_denylist_sha256"],
        "annotation_or_action_labels_read":False,"test_members_opened":0,"model_forward_calls":0,
        "source_raw_hours_not_equal_QC_accepted_hours":True,"new_acquired_independent_raw_bytes":0,
        "preparation_does_not_prove_training_completed":True,"elapsed_seconds":time.time()-started,
        "completed_epoch":time.time(),"absolute_deadline_epoch":receipt["absolute_deadline_epoch"]}
    guard(receipt,output)
    write_json(output/"corpus_summary.json",summary)
    write_json(output/"completion.json",{"status":"completed","corpus_summary_sha256":sha(output/"corpus_summary.json"),
        "manifest_sha256":sha(output/"manifest.jsonl"),"completed_epoch":time.time()})
    print(json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    for name in ("archive","contract-dir","output","resource-receipt","phase2-source"):
        p.add_argument("--"+name,required=True)
    run(p.parse_args())


if __name__=="__main__":main()
