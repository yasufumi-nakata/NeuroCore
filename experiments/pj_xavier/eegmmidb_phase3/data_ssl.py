"""All-slot and original-source validation for label-free continuous EEG."""
from __future__ import annotations
from collections import defaultdict,Counter
import json,math
from pathlib import Path
import numpy as np
from eegmmidb_phase3.common import (
 EEGStore,contained,is_sha,object_hash,require,sha256,source_interval_union,
)

def read_rows(path):
 return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def load_corpus(root,verify_bytes=True):
 root=Path(root).resolve()
 rows=read_rows(root/"manifest.jsonl");artifacts=json.loads((root/"artifacts.json").read_text())
 allow=read_rows(root/"source_allowlist.jsonl");deny=read_rows(root/"source_denylist.jsonl")
 assignment=json.loads((root/"subject_split.json").read_text())["assignments"]
 by_source={r["source_eeg_sha256"]:r for r in allow}
 require(len(by_source)==len(allow),"Repeated original source in allowlist")
 denied_sha={r["source_eeg_sha256"] for r in deny};denied_payload={r["eeg_payload_sha256"] for r in deny}
 forbidden_groups={g for g,s in assignment.items() if s=="test"}
 require(rows and {r["split"] for r in rows}=={"train","val"},"Both allowed SSL partitions required")
 ids=set();slots=defaultdict(set);groups=defaultdict(set);sources=defaultdict(set)
 payloads=defaultdict(set);files=defaultdict(set);intervals=defaultdict(list)
 channels=rows[0]["channels"]
 require(len(channels)==64 and len(set(channels))==64,"Full64 unique channel order required")
 for row in rows:
  require(not {"label","action_label","events","gold_annotation_code","task_family"} & row.keys(),"Action or cue supervision in SSL manifest")
  identity=row.get("id");require(isinstance(identity,str) and identity and identity not in ids,"Repeated/missing SSL ID");ids.add(identity)
  source=row["source_eeg_sha256"];payload=row["eeg_payload_sha256"]
  require(is_sha(source) and is_sha(payload) and row.get("source_sha256")==source,"Raw identity invalid")
  require(source in by_source and source not in denied_sha and payload not in denied_payload,"Denied source or payload")
  original=by_source[source]
  for key in ("subject_group","source_recording_id","split","eeg_payload_sha256","channels"):
   require(row[key]==original[key],"SSL row misrepresents original source: "+key)
  require(row["subject_group"] not in forbidden_groups and assignment[row["subject_group"]]==row["split"],"Test participant or changed split")
  require(row["channels"]==channels and row["unit"]=="V" and row["sample_rate_hz"]==200,"Channel/physical-unit identity changed")
  require(row.get("preprocessing",{}).get("online_causality_verified") is True,"Causality proof missing")
  begin,end,available,decision=[float(row[k]) for k in ("window_start_s","window_end_s","feature_available_time_s","decision_time_s")]
  require(all(math.isfinite(v) for v in (begin,end,available,decision)) and begin>=5 and abs(end-begin-3)<1e-8 and available<=decision+1e-8 and end<=decision+1e-8,"Invalid observed interval")
  index=row.get("eeg_index");name=row.get("eeg_path")
  require(isinstance(index,int) and not isinstance(index,bool) and index>=0,"Index must be an integer")
  require(name in artifacts and index not in slots[name],"Duplicate/missing array slot")
  seal=artifacts[name]
  require(seal["dtype"]=="float32" and len(seal["shape"])==3 and seal["shape"][1:]==[64,600] and index<seal["shape"][0],"Wrong array shape")
  require(all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in seal["shape"]),"Invalid array dimensions")
  require(row["array_sha256"]==seal["sha256"] and is_sha(seal["sha256"]),"Array SHA declaration differs")
  slots[name].add(index)
  groups[row["subject_group"]].add(row["split"]);sources[source].add(row["split"]);payloads[payload].add(row["split"]);files[name].add(row["split"])
  intervals[source].append((begin,end));row["_path"]=str(contained(root,name))
 for mapping in (groups,sources,payloads,files):
  require(all(len(v)==1 for v in mapping.values()),"Participant/source/payload/file crosses split")
 for intervals0 in intervals.values():
  spans=sorted(intervals0)
  require(all(a[1]<=b[0]+1e-8 for a,b in zip(spans,spans[1:])),"Overlapping original windows")
 require(set(slots)==set(artifacts),"Extra or missing artifacts")
 for name,seal in artifacts.items():
  require(slots[name]==set(range(seal["shape"][0])),"Every array slot must be uniquely accounted")
  p=contained(root,name)
  if verify_bytes:require(p.stat().st_size==seal["bytes"] and sha256(p)==seal["sha256"],"Actual waveform bytes differ")
  array=np.load(p,mmap_mode="r",allow_pickle=False)
  require(array.dtype==np.float32 and list(array.shape)==seal["shape"],"Array header changed")
 audit={
  "schema":"xavier-phase3-independent-SSL-input-audit-v1","status":"passed",
  "manifest_sha256":sha256(root/"manifest.jsonl"),"artifacts_sha256":sha256(root/"artifacts.json"),
  "source_allowlist_sha256":sha256(root/"source_allowlist.jsonl"),
  "source_denylist_sha256":sha256(root/"source_denylist.jsonl"),
  "subject_split_sha256":sha256(root/"subject_split.json"),
  "window_counts":dict(Counter(r["split"] for r in rows)),
  "source_interval_union_seconds":{s:source_interval_union([r for r in rows if r["split"]==s]) for s in ("train","val")},
  "array_count":len(artifacts),"array_bytes":sum(s["bytes"] for s in artifacts.values()),
  "test_participant_groups_excluded":len(forbidden_groups),
  "all_slots_unique_and_complete":True,"all_used_sources_allowed_and_not_denied":True,
  "test_or_action_label_forward_calls":0,
 }
 return rows,artifacts,audit

def membership(rows,audit):
 parts={s:sorted(r["id"] for r in rows if r["split"]==s) for s in ("train","val")}
 return {
  "schema":"xavier-phase3-SSL-membership-v1",**audit,
  "training_ids":parts["train"],"validation_ids":parts["val"],
  "ids_sha256":object_hash(parts),
  "original_source_sha256":{s:sorted({r["source_eeg_sha256"] for r in rows if r["split"]==s}) for s in parts},
  "payload_sha256":{s:sorted({r["eeg_payload_sha256"] for r in rows if r["split"]==s}) for s in parts},
  "subject_groups":{s:sorted({r["subject_group"] for r in rows if r["split"]==s}) for s in parts},
  "test_used_for_tuning":False,"action_labels_used":False,
 }

class SSLStore(EEGStore):
 def get(self,row):
  require(row["split"] in ("train","val"),"Test data cannot enter SSL")
  values=super().get(row)
  require(float(np.abs(values).max())<=.00050001,"Prepared amplitude exceeds fixed500microvolt QC")
  return values

