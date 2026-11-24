#!/usr/bin/env python3
"""Resolve metadata-only official OpenNeuro S3 manifest; NEVER downloads EEG objects."""
import datetime,hashlib,json,urllib.parse,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
DATASET="ds005841"
BASE="https://s3.amazonaws.com/openneuro.org"
OUT=Path.home()/"pj-xavier-stage/catalog-bridge-v1"
NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
def get_small(url,maximum=350000):
 req=urllib.request.Request(url,headers={"User-Agent":"PJ-Xavier-MetadataOnly/1.0"})
 with urllib.request.urlopen(req,timeout=15) as r:
  if r.status!=200:raise RuntimeError("unexpected HTTP status "+str(r.status))
  data=r.read(maximum+1)
  if len(data)>maximum:raise RuntimeError("metadata budget exceeded")
  return data
def main():
 description=json.loads(get_small(BASE+"/"+DATASET+"/dataset_description.json",10000))
 if description.get("License")!="CC0" or "ds005841" not in description.get("DatasetDOI",""):
  raise RuntimeError("official dataset/license identity mismatch")
 files=[];seen_tokens=set();token=None;pages=0
 while True:
  q={"list-type":"2","prefix":DATASET+"/","max-keys":"1000"}
  if token:q["continuation-token"]=token
  raw=get_small(BASE+"?"+urllib.parse.urlencode(q),350000)
  root=ET.fromstring(raw);pages+=1
  for x in root.findall("s:Contents",NS):
   key=x.findtext("s:Key",default="",namespaces=NS)
   size=int(x.findtext("s:Size",default="-1",namespaces=NS))
   etag=x.findtext("s:ETag",default="",namespaces=NS).strip('"')
   if not key.startswith(DATASET+"/") or ".." in key.split("/") or size<0:raise RuntimeError("invalid object identity")
   files.append({"path":key,"size_bytes":size,"provider_etag_unverified":etag,
      "last_modified":x.findtext("s:LastModified",default="",namespaces=NS),
      "checksum_algorithms":[v.text for v in x.findall("s:ChecksumAlgorithm",NS)]})
  more=(root.findtext("s:IsTruncated",default="false",namespaces=NS) or "").lower()=="true"
  if not more:break
  token=root.findtext("s:NextContinuationToken",default="",namespaces=NS)
  if not token or token in seen_tokens:raise RuntimeError("S3 pagination invalid")
  seen_tokens.add(token)
  if pages>=30:raise RuntimeError("S3 pagination budget exceeded")
 total=sum(f["size_bytes"] for f in files)
 if len({f["path"] for f in files})!=len(files):raise RuntimeError("duplicate S3 paths")
 suffixes={}
 for f in files:
  ext=Path(f["path"]).suffix.lower()
  suffixes[ext]=suffixes.get(ext,0)+1
 report={"schema":"xavier-openneuro-metadata-only-v1","observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
   "dataset_id":DATASET,"source":"public_s3_list_objects_v2","source_url":BASE,
   "license":description["License"],"doi":description.get("DatasetDOI"),"dataset_title":description.get("Name"),
   "pages":pages,"file_count":len(files),"total_bytes":total,"extension_counts":suffixes,
   "files":files,"download_authorized_by_this_receipt":False,
   "warnings":["S3 ETag is not universally a content checksum","raw data has not been fetched or QC checked",
       "dataset is only a source candidate until version, train split, must-link, and EEG signal acceptance are sealed"]}
 OUT.mkdir(parents=True,exist_ok=True)
 serialized=(json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode()
 path=OUT/"ds005841_official_s3_manifest_metadata.json"
 with path.open("xb") as f:f.write(serialized)
 print(json.dumps({"path":str(path),"sha256":hashlib.sha256(serialized).hexdigest(),"dataset_id":DATASET,
      "license":description["License"],"file_count":len(files),"pages":pages,"total_bytes":total,
      "extension_counts":suffixes},ensure_ascii=False))
if __name__=="__main__":main()
