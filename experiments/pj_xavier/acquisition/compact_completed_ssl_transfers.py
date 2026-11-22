"""Reversible, byte-exact compaction of this project's completed SSL transfer packs.

Only corpus_*.tar containers of private/xavier/transfers/ssl_v1 are eligible.
Original prepared arrays, raw EEG, models, manifests, and transfer receipts stay
in place. A recipe keeps exact TAR headers/padding and small metadata, referring
to hash-verified immutable array bytes. Every pack is reconstructed and compared
byte-for-byte before any transfer pack is removed.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def contained(root, rel):
    p=Path(rel)
    require(not p.is_absolute() and ".." not in p.parts, "Untrusted relative path")
    target=Path(root)/p
    require(target.resolve().is_relative_to(Path(root).resolve()) and not target.is_symlink(), "Escaped or symlink path")
    return target

def emit_pack(repo, pack, destination):
    target=Path(destination)
    require(not target.exists(), "Preserve existing restoration target")
    whole=hashlib.sha256()
    size=0
    with target.open("xb") as output:
        for segment in pack["segments"]:
            if segment["kind"]=="literal":
                value=base64.b64decode(segment["base64"],validate=True)
                require(len(value)==segment["bytes"],"Literal size differs")
                output.write(value);whole.update(value);size+=len(value)
            elif segment["kind"]=="prepared_array":
                source=contained(repo,segment["path"])
                require(source.is_file() and source.stat().st_size==segment["bytes"],"Retained array missing or changed")
                before=source.stat();digest=hashlib.sha256();count=0
                with source.open("rb") as f:
                    for value in iter(lambda:f.read(1048576),b""):
                        digest.update(value);whole.update(value);output.write(value)
                        count+=len(value);size+=len(value)
                after=source.stat()
                require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),"Retained array changed while reading")
                require(count==segment["bytes"] and digest.hexdigest()==segment["sha256"],"Retained array SHA differs")
            else:
                raise ValueError("Unknown recipe segment")
        output.flush();os.fsync(output.fileno())
    require(size==pack["bytes"] and whole.hexdigest()==pack["sha256"],"Byte-exact reconstruction failed")
    return {"bytes":size,"sha256":whole.hexdigest()}

def compare(a,b):
    with Path(a).open("rb") as x,Path(b).open("rb") as y:
        while True:
            left=x.read(1048576);right=y.read(1048576)
            require(left==right,"Independent byte comparison failed")
            if not left:
                return

def prepare(repo, output):
    repo=Path(repo).resolve();output=Path(output).resolve()
    require(not output.exists(),"Preserve existing recipe")
    source=repo/"private/xavier/transfers/ssl_v1"
    manifest_path=source/"transfer_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    require(manifest["schema"]=="xavier-ssl-workstation-to-Colab-transfer-v1" and len(manifest["packs"])==30,"Unregistered transfer family")
    output.mkdir(parents=True)
    packs=[]
    for item in manifest["packs"]:
        require(re.fullmatch(r"corpus_\d{3}\.tar",item["name"]) is not None,"Only completed corpus transfer packs")
        archive=source/item["name"]
        require(not archive.is_symlink() and archive.stat().st_size==item["bytes"] and sha(archive)==item["sha256"],"Original transfer identity differs")
        before=archive.stat();segments=[]
        def literal(data):
            if data:
                segments.append({"kind":"literal","bytes":len(data),"base64":base64.b64encode(data).decode("ascii")})
        with tarfile.open(archive,"r:") as tar,archive.open("rb") as raw:
            members=tar.getmembers()
            require([m.name for m in members]==item["members"] and all(m.isfile() for m in members),"Transfer membership/type differs")
            cursor=0
            for member in members:
                require(member.offset_data>=cursor and member.size==manifest["files"][member.name]["bytes"],"TAR offset/member size differs")
                raw.seek(cursor);literal(raw.read(member.offset_data-cursor))
                if member.name.startswith("prepared/ssl_portable_uniba_v1/waveforms/") and member.name.endswith(".npy"):
                    target=contained(repo,"private/xavier/"+member.name)
                    require(target.is_file() and target.stat().st_size==member.size,"Retained prepared array missing")
                    segments.append({"kind":"prepared_array","path":str(target.relative_to(repo)),"bytes":member.size,"sha256":manifest["files"][member.name]["sha256"]})
                else:
                    require(member.size<16*1048576,"Unreviewed non-array payload")
                    raw.seek(member.offset_data);value=raw.read(member.size)
                    require(len(value)==member.size and hashlib.sha256(value).hexdigest()==manifest["files"][member.name]["sha256"],"Small historical metadata differs from original transfer manifest")
                    literal(value)
                cursor=member.offset_data+member.size
            raw.seek(cursor);literal(raw.read())
        after=archive.stat()
        require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),"Transfer changed during recipe creation")
        pack={"path":str(archive.relative_to(repo)),"bytes":item["bytes"],"sha256":item["sha256"],"stat":[before.st_ino,before.st_size,before.st_mtime_ns],"segments":segments}
        require(shutil.disk_usage(output).free>35*1024**3+item["bytes"],"Roundtrip would breach retained free space")
        with tempfile.TemporaryDirectory(prefix="roundtrip_",dir=output) as temporary:
            restored=Path(temporary)/item["name"];emit_pack(repo,pack,restored);compare(archive,restored)
        packs.append(pack)
    recipe={"schema":"xavier-reversible-SSL-transfer-compaction-v1","created_epoch":time.time(),"source_sha256":sha(__file__),"manifest_path":str(manifest_path.relative_to(repo)),"manifest_sha256":sha(manifest_path),"packs":packs,"all_30_actual_roundtrips_byte_compared":True,"original_raw_models_arrays_and_receipts_unmodified":True}
    path=output/"recipe.json";write_new(path,recipe)
    return {"status":"prepared_and_actual_roundtrips_verified","packs":len(packs),"recoverable_transfer_bytes":sum(x["bytes"] for x in packs),"recipe_bytes":path.stat().st_size,"recipe_sha256":sha(path),"removals":0}

def compact(repo, recipe_path, expected):
    repo=Path(repo).resolve();recipe_path=Path(recipe_path)
    require(sha(recipe_path)==expected,"Recipe identity differs")
    recipe=json.loads(recipe_path.read_text())
    require(recipe["schema"]=="xavier-reversible-SSL-transfer-compaction-v1" and recipe["all_30_actual_roundtrips_byte_compared"] and recipe["source_sha256"]==sha(__file__),"Recipe lacks exact roundtrip proof")
    require(sha(contained(repo,recipe["manifest_path"]))==recipe["manifest_sha256"],"Original transfer manifest changed")
    receipt=recipe_path.parent/"compaction_receipt.json";journal=recipe_path.parent/"compaction_actions.jsonl"
    require(not receipt.exists() and not journal.exists(),"Preserve previous compaction record")
    for pack in recipe["packs"]:
        target=contained(repo,pack["path"])
        require(target.parent==repo/"private/xavier/transfers/ssl_v1" and re.fullmatch(r"corpus_\d{3}\.tar",target.name),"Deletion outside owned transient transfer scope")
        st=target.stat()
        require([st.st_ino,st.st_size,st.st_mtime_ns]==pack["stat"] and sha(target)==pack["sha256"],"Original transfer changed before compaction")
        for segment in pack["segments"]:
            if segment["kind"]=="prepared_array":
                p=contained(repo,segment["path"])
                require(p.is_file() and p.stat().st_size==segment["bytes"] and sha(p)==segment["sha256"],"Immutable retained source changed")
    free_before=shutil.disk_usage(repo).free
    with journal.open("x") as stream:
        for pack in recipe["packs"]:
            target=contained(repo,pack["path"]);target.unlink()
            stream.write(json.dumps({"path":pack["path"],"removed_transfer_bytes":pack["bytes"],"sha256":pack["sha256"],"recipe_sha256":expected,"epoch":time.time()})+"\n")
            stream.flush();os.fsync(stream.fileno())
    result={"schema":"xavier-reversible-SSL-transfer-compaction-receipt-v1","status":"completed","recipe_sha256":expected,"removed_derived_transfer_packs":len(recipe["packs"]),"removed_transfer_bytes":sum(p["bytes"] for p in recipe["packs"]),"preserved_raw_arrays_models_manifest_and_receipts":True,"free_disk_bytes_before":free_before,"free_disk_bytes_after":shutil.disk_usage(repo).free,"other_concurrent_downloads_can_affect_free_space_delta":True,"completed_epoch":time.time()}
    write_new(receipt,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","compact","restore"])
    p.add_argument("--repo",required=True);p.add_argument("--output");p.add_argument("--recipe");p.add_argument("--expected-recipe-sha256");p.add_argument("--pack")
    a=p.parse_args()
    if a.action=="prepare":
        result=prepare(a.repo,a.output)
    elif a.action=="compact":
        result=compact(a.repo,a.recipe,a.expected_recipe_sha256)
    else:
        require(sha(a.recipe)==a.expected_recipe_sha256,"Recipe SHA differs")
        recipe=json.loads(Path(a.recipe).read_text());pack=next(x for x in recipe["packs"] if Path(x["path"]).name==a.pack)
        result=emit_pack(Path(a.repo).resolve(),pack,a.output)
    print(json.dumps(result,allow_nan=False))

if __name__=="__main__":
    main()
