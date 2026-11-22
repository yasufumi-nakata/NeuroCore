#!/usr/bin/env python3
"""Standalone, read-only catalog bridge for smac31; stdlib-only.
Never edits the root acquisition catalog, downloads DB, download guardians, or any EEG payload.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
from pathlib import Path
import catalog_identity as lib

def canon_url(url):
    try:
        p=urllib.parse.urlsplit(str(url or ""))
        return ((p.hostname or "").lower(),urllib.parse.unquote(p.path).rstrip("/").lower())
    except ValueError:return ("","")

def assess(candidate_data,catalog):
    # catalog is an iterable of already parsed public source metadata
    inverted={}
    by_title={}
    source_status=collections.Counter()
    nodes=0
    for d in catalog:
        nodes+=1
        source_status[str(d.get("status","unknown"))]+=1
        keys=lib.keys(doi=d.get("doi") or d.get("source_doi"),url=d.get("source_url"),family=d.get("source_family_key"),concept=d.get("source_concept_key"))
        for key in keys:inverted.setdefault(key,set()).add(nodes)
        t=" ".join(str(d.get("title","")).lower().split())
        if t:by_title.setdefault(t,[]).append({"source_index":nodes,"status":d.get("status"),"url":d.get("source_url"),"source_family_key":d.get("source_family_key")})
    exact_match=set()
    for rid,rec in candidate_data.items():
        if set(rec["keys"]) & set(inverted):
            exact_match.add(rid)
    alias_match={}
    for rid,rec in candidate_data.items():
        if rid in exact_match:continue
        title=" ".join(rec["title"].lower().split())
        hits=[h for h in by_title.get(title,[]) if rec.get("url") and canon_url(h["url"])==canon_url(rec["url"])]
        if len(hits)==1:
            alias_match[rid]={"status":hits[0]["status"],"match_basis":"exact_title_and_exact_normalized_url"}
    matched=exact_match|set(alias_match)
    counts=collections.Counter()
    for rid,rec in candidate_data.items():
        counts[rec["collection"]+"_matched" if rid in matched else rec["collection"]+"_unmatched"]+=1
    queue=[]
    for rid,rec in candidate_data.items():
        if rid in matched:continue
        if rec["collection"]=="global" and rec.get("eeg_text_match") and rec["access"]=="immediate" and rec["keys"]:
            queue.append({"title":rec["title"],"doi":rec.get("doi", ""),"source_url_candidate":rec["url"],
                          "status":"requires_official_public_manifest_and_license_eeg_payload_verification",
                          "download_authorized":False})
    return {"root_catalog_rows":nodes,"root_source_status":dict(source_status),
            "matches":dict(counts),"exact_key_matches":len(exact_match),
            "exact_title_url_aliases":len(alias_match),"candidate_review_queue":queue}

def verify_manifest(path):
    if not path.exists():return None
    d=json.loads(path.read_text())
    files=d["files"]
    if d["dataset_id"]!="ds005841" or d["license"]!="CC0" or sum(x["size_bytes"] for x in files)!=d["total_bytes"]:
        raise RuntimeError("official metadata manifest validation failed")
    if d["file_count"]!=len(files) or len([f for f in files if f["path"].lower().endswith(".bdf")])!=288:
        raise RuntimeError("official metadata manifest inconsistent")
    return {"dataset":"ds005841","objects":len(files),"bdf_objects":288,
            "bytes":d["total_bytes"],"official_metadata_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "downloaded_bytes_by_this_bridge":0}

def main():
    ap=argparse.ArgumentParser()
    base=Path.home()/"pj-xavier/catalog-bridge-v1"
    ap.add_argument("--input-dir",type=Path,default=base)
    ap.add_argument("--root-catalog",type=Path,default=Path.home()/"pj-xavier/control_20261122/acquisition/catalog_fresh_resolution.jsonl")
    ap.add_argument("--out-dir",type=Path,default=base/"reports")
    ap.add_argument("--check-only",action="store_true")
    args=ap.parse_args()
    # Output path must be an isolated destination, regardless of caller arguments.
    if os.path.commonpath([str(args.out_dir.expanduser().resolve()),str(base.resolve())])!=str(base.resolve()):
        raise RuntimeError("output outside isolated namespace prohibited")
    if not args.root_catalog.is_file():raise FileNotFoundError(args.root_catalog)
    for name in ("eeg_dataset_summary_ja.csv","audit/global_eeg/global_eeg_merged.csv"):
        if not (args.input_dir/name).is_file():raise FileNotFoundError(args.input_dir/name)
    if os.getloadavg()[0]>=4:raise RuntimeError("smac31 load too high for optional catalog comparison")
    if shutil.disk_usage(base.parent).free<150*(1024**3):raise RuntimeError("disk reserve insufficient")
    lib.ROOT=args.input_dir
    candidates=lib.build_candidates()
    with args.root_catalog.open(encoding="utf-8") as f:
        d=assess(candidates,(json.loads(line) for line in f))
    if d["root_catalog_rows"]<1:raise RuntimeError("empty or invalid root catalog")
    d["official_dataset_probe"]=verify_manifest(args.input_dir/"ds005841_official_s3_manifest_metadata.json")
    d.update(schema="xavier-standalone-catalog-diff-v1",ran_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             high_confidence_source_count=sum(x["collection"]=="high" for x in candidates.values()),broader_discovery_count=sum(x["collection"]=="global" for x in candidates.values()),
             applies_to_root=False,raw_data_downloads_started=0,train_fit_started=0,
             explanation="Discovery-only unmatched is not proof of downloadable EEG. Exact URL/title aliases count as existing. All source promotions need official manifest/license/EEG checks.")
    summary={k:v for k,v in d.items() if k!="candidate_review_queue"}
    summary["candidate_review_queue_count"]=len(d["candidate_review_queue"])
    if not args.check_only:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        fn=args.out_dir/("catalog_diff_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
        with fn.open("x",encoding="utf-8") as f:json.dump(d,f,ensure_ascii=False,indent=2)
        summary["report_path"]=str(fn)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
