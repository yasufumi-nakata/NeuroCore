#!/usr/bin/env python3
"""S Mac-only numerical diagnostic for pre-frozen Figshare 9970619 TRAIN FDT.

Never interprets numeric values as volts, updates a model, or opens val/test signals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket

from freeze_smac_ssvep9970619_metadata import SOURCE, approved_host, sha


def train_rows(manifest: dict, *, expected_metadata_manifest_sha256: str) -> list[dict]:
    if (manifest.get('schema') != 'xavier-figshare9970619-metadata-only-split-v1'
            or manifest.get('source') != SOURCE or manifest.get('fit_allowed') is not False
            or manifest.get('test_signal_files_opened') != 0
            or manifest.get('validation_signal_files_opened') != 0):
        raise ValueError('untrusted or incorrectly admitted frozen split')
    rows = manifest.get('items', [])
    if len(rows)!=44 or {x.get('split') for x in rows}!={'train','val','test'}:
        raise ValueError('incorrect source partitions')
    actual=sha(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())
    if len(expected_metadata_manifest_sha256)!=64 or actual!=expected_metadata_manifest_sha256 or manifest.get('metadata_manifest_sha256')!=actual:
        raise ValueError('immutable metadata or independent expected SHA mismatch')
    for entry in rows:
        name=entry.get('fdt_name')
        header=entry.get('set_name')
        if (not isinstance(name,str) or not isinstance(header,str) or
                Path(name).name!=name or Path(header).name!=header or
                not name.endswith('.fdt') or Path(name).stem!=Path(header).stem or
                not header.endswith('.set') or entry.get('source')!=SOURCE or
                entry.get('fdt_expected_bytes',0)<=0):
            raise ValueError('unsafe source or unpaired metadata path')
    train=[r for r in rows if r['split']=='train']
    if len(train)!=28 or sum(r['split']=='test' for r in rows)!=8 or sum(r['split']=='val' for r in rows)!=8:
        raise ValueError('unexpected heldout/training count')
    if any(r.get('fdt_content_opened') is not False for r in rows):
        raise ValueError('prior heldout content access not allowed')
    return train


def diagnose(directory: Path, frozen_receipt: Path, *, host: str, expected_split_sha256: str, sample_count: int=8192) -> dict:
    if not approved_host(host) or directory.is_symlink() or directory.name!=SOURCE:
        raise PermissionError('S Mac source-root only; smac19 excluded')
    if sample_count<128 or sample_count>16384:
        raise ValueError('diagnostic sample limit exceeded')
    import numpy as np
    receipt=json.loads(frozen_receipt.read_text())
    rows=train_rows(receipt,expected_metadata_manifest_sha256=expected_split_sha256)
    results=[]
    seen_hashes={}
    for entry in rows:
        fdt=directory/entry['fdt_name']
        if fdt.is_symlink() or not fdt.is_file() or fdt.stat().st_size!=entry['fdt_expected_bytes']:
            raise ValueError('train FDT missing or changed')
        header=directory/entry['set_name']
        if header.is_symlink() or not header.is_file() or sha(header.read_bytes())!=entry['set_header_sha256']:
            raise ValueError('frozen train header has changed')
        count=fdt.stat().st_size//(73*4)
        raw=np.memmap(fdt,mode='r',dtype='<f4',shape=(count,73))
        idx=np.linspace(0,count-1,min(sample_count,count),dtype='int64')
        eeg=np.asarray(raw[idx,:64],dtype='float64')
        if not np.isfinite(eeg).all():raise ValueError('nonfinite numerical EEG values')
        mean_ref=eeg-eeg.mean(axis=1,keepdims=True)
        centred=eeg-np.median(eeg,axis=0,keepdims=True)
        def quant(x,p):return float(np.quantile(np.abs(x),p))
        # A content SHA is calculated only for train FDT; no heldout signal opened.
        h=hashlib.sha256()
        with fdt.open('rb') as stream:
            while True:
                chunk=stream.read(8*1024*1024)
                if not chunk:break
                h.update(chunk)
        digest=h.hexdigest()
        if digest in seen_hashes:
            # Potential same-signal aliases in TRAIN must-link together.
            raise ValueError('duplicate train FDT content: must-link audit required')
        seen_hashes[digest]=True
        results.append({'fdt_name':entry['fdt_name'],'train_group_sha256':entry['group_token_sha256'],
                        'train_raw_sha256':digest,'sampled_eeg_values':int(eeg.size),
                        'absolute_q50_native_numeric':quant(eeg,0.5),
                        'absolute_q99_native_numeric':quant(eeg,0.99),
                        'absolute_q99_post_channel_median_subtraction_numeric':quant(centred,0.99),
                        'absolute_q99_post_common_average_numeric':quant(mean_ref,0.99)})
    def stats(field):return {'median':float(np.median([x[field] for x in results])),
                             'maximum':max(x[field] for x in results)}
    return {'schema':'xavier-ssvep9970619-train-only-numeric-diagnostic-v1',
            'frozen_metadata_manifest_sha256':receipt['metadata_manifest_sha256'],
            'train_files_hashed':len(results),'test_signal_files_opened':0,'validation_signal_files_opened':0,
            'new_gradient_windows':0,'fit_allowed':False,'online_harness_allowed':False,
            'voltage_conversion_factor':None,'source_preprocessing_causality_proven':False,
            'unit_inference':'UNKNOWN; original FDT raw numeric values, not volts or microvolts',
            'native_abs_q50':stats('absolute_q50_native_numeric'),
            'native_abs_q99':stats('absolute_q99_native_numeric'),
            'channel_median_subtracted_abs_q99':stats('absolute_q99_post_channel_median_subtraction_numeric'),
            'common_average_abs_q99':stats('absolute_q99_post_common_average_numeric'),
            'items':results}


def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True)
    a.add_argument('--split',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True)
    a.add_argument('--expected-split-sha256',required=True)
    args=a.parse_args()
    r=diagnose(args.source,args.split,host=socket.gethostname(),expected_split_sha256=args.expected_split_sha256)
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError('immutable train numeric diagnostic exists')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='items'},sort_keys=True))

if __name__=='__main__':main()
