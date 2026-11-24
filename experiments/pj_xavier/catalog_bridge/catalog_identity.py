#!/usr/bin/env python3
"""Public, stdlib-only EEG source identity helpers. No SSH or downloader.

Do not use catalogue row indexes/IDs as identity. These helpers do not
certify a source license, EEG payload, or training eligibility.
"""
import csv
import re
import urllib.parse
from pathlib import Path

ROOT=Path.cwd()
DOI_RE=re.compile(r"10\.\d{4,9}/[^\s?#]+",re.I)
DS_RE=re.compile(r"\bds\d{5,7}\b",re.I)

def norm_doi(s):
    s=urllib.parse.unquote(str(s or "")).strip().lower()
    s=re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", s)
    m=DOI_RE.search(s)
    return m.group(0).rstrip(" .;,)") if m else ""

def keys(doi="",url="",family="",concept=""):
    k=set()
    for d in (doi,url):
        value=norm_doi(d)
        if value:
            k.add("doi:"+value)
            z=re.fullmatch(r"10\.5281/zenodo\.(\d+)",value)
            if z:k.add("zenodo:"+z.group(1))
            ds=DS_RE.search(value)
            if ds:k.add("openneuro:"+ds.group().lower())
    for u in (url,):
        u=str(u or "").strip()
        if not u:continue
        try:
            p=urllib.parse.urlsplit(u)
            h=(p.hostname or "").lower().removeprefix("www.")
            path=urllib.parse.unquote(p.path).lower().rstrip("/")
            if h and path:k.add("url:"+h+path)
            z=re.search(r"(?:/records/|/record/|zenodo\.)(\d+)",path)
            if h=="zenodo.org" and z:k.add("zenodo:"+z.group(1))
            ds=DS_RE.search(u)
            if ds and (h=="openneuro.org" or "openneuro" in u.lower()):k.add("openneuro:"+ds.group().lower())
            if h in ("figshare.com","figshare.org") or h.endswith(".figshare.com"):
                q=re.search(r"/(?:articles|datasets)/[^?#]*?/(\d+)(?:/|$)",path)
                if q:k.add("figshare:"+q.group(1))
        except ValueError:pass
    for x in (family,concept):
        x=str(x or "").lower().strip()
        if x:
            k.add("family:"+x)
            m=re.match(r"zenodo[_-](?:record|concept):(\d+)",x)
            if m:k.add("zenodo:"+m.group(1))
            m=DS_RE.search(x)
            if m and "openneuro" in x:k.add("openneuro:"+m.group().lower())
    return k

def build_candidates():
    candidates={}
    for rel,src in [("eeg_dataset_summary_ja.csv","high"),("audit/global_eeg/global_eeg_merged.csv","global")]:
        with (ROOT/rel).open(encoding="utf-8-sig",newline="") as f:
            for n,row in enumerate(csv.DictReader(f)):
                title=row.get("データセット名") if src=="high" else row.get("title")
                url=row.get("公開URL") if src=="high" else row.get("url")
                doi=row.get("DOI") if src=="high" else row.get("doi")
                access=row.get("アクセス区分") if src=="high" else row.get("access")
                kk=sorted(keys(doi=doi,url=url))
                candidates[src+":"+str(n)]=dict(title=(title or "")[:180],doi=norm_doi(doi),url=url or "",access=access or "",keys=kk,collection=src,eeg_text_match=(row.get('eeg_text_match','').lower()=='true' if src=='global' else True))
    return candidates
