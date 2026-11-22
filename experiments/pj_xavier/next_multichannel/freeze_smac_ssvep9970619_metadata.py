#!/usr/bin/env python3
"""Immutable, metadata-only S Mac group split for Figshare 9970619.

This is NOT signal-unit certification, training admission, or online causality.
Never decode heldout .fdt. Writes private receipts on a verified smac worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket

SOURCE = 'figshare-9970619-1'
SALT = 'PJ-Xavier|Figshare9970619|v1|freeze|2026-11-23'
EXPECTED_GROUPS = 11
RECORDS_PER_GROUP = 4


def approved_host(host: str) -> bool:
    short = host.split('.')[0].lower()
    return bool(re.fullmatch(r'smac[0-9]{2}', short)) and short not in {'smac19','smac31','smac40'}


def sha(v: bytes) -> str:
    return hashlib.sha256(v).hexdigest()


def split_tokens(tokens: list[str]) -> dict[str, str]:
    if len(tokens) != EXPECTED_GROUPS or len(set(tokens)) != EXPECTED_GROUPS:
        raise ValueError('expected 11 unique filename subject tokens')
    ranked = sorted(tokens, key=lambda x: (sha((SALT+'|'+x).encode()), x))
    return {token: ('train' if i < 7 else 'val' if i < 9 else 'test')
            for i, token in enumerate(ranked)}


def discover_headers(directory: Path, *, host: str) -> list[tuple[str, Path, Path]]:
    if not approved_host(host):
        raise PermissionError('verified S Mac download worker required; smac19 forbidden')
    if directory.is_symlink() or not directory.is_dir() or directory.name != SOURCE:
        raise ValueError('wrong or symbolic-link dataset source')
    sets = sorted(directory.glob('*.set'))
    if len(sets) != 44:
        raise ValueError('source has wrong number of .set headers')
    records = []
    by_token = {}
    for s in sets:
        f = s.with_suffix('.fdt')
        if (s.is_symlink() or f.is_symlink() or not s.is_file() or not f.is_file()
                or s.stat().st_size == 0 or f.stat().st_size == 0):
            raise ValueError('missing or symbolic-link source pair')
        if f.stat().st_size % (73 * 4) != 0:
            raise ValueError('inconsistent 73-channel float32 external FDT geometry')
        token = s.name.split('_')[0]
        if not token or not re.fullmatch('[A-Za-z0-9-]+', token):
            raise ValueError('invalid original subject token')
        by_token[token] = by_token.get(token, 0) + 1
        records.append((token, s, f))
    if len(by_token) != EXPECTED_GROUPS or set(by_token.values()) != {RECORDS_PER_GROUP}:
        raise ValueError('non-uniform subject-token grouping; cannot freeze')
    return records


def prepare(directory: Path, *, host: str) -> dict:
    records = discover_headers(directory, host=host)
    mapping = split_tokens(sorted({token for token,_,_ in records}))
    items = []
    for token, header, raw in records:
        items.append({
            'group_token_sha256': sha((SALT+'|'+token).encode()),
            'set_name': header.name,
            'set_header_sha256': sha(header.read_bytes()),
            'fdt_name': raw.name,
            'fdt_expected_bytes': raw.stat().st_size,
            'fdt_content_opened': False,
            'split': mapping[token],
            'source': SOURCE,
            'duration_nominal_seconds_from_73ch_200hz': raw.stat().st_size / (73*4*200),
        })
    splits = {k: [row for row in items if row['split'] == k] for k in ('train','val','test')}
    public = {
        'record_count': len(items),
        'group_tokens': len(mapping),
        'train_group_tokens': 7, 'val_group_tokens': 2, 'test_group_tokens': 2,
        'train_records': len(splits['train']),
        'val_records': len(splits['val']),
        'test_records': len(splits['test']),
        'nominal_train_seconds': sum(x['duration_nominal_seconds_from_73ch_200hz'] for x in splits['train']),
        'nominal_val_seconds': sum(x['duration_nominal_seconds_from_73ch_200hz'] for x in splits['val']),
        'nominal_test_seconds': sum(x['duration_nominal_seconds_from_73ch_200hz'] for x in splits['test']),
    }
    if sum(public[k] for k in ('train_records','val_records','test_records')) != 44:
        raise AssertionError('split coverage')
    digest = sha(json.dumps(items,sort_keys=True,separators=(',',':')).encode())
    return {
        'schema': 'xavier-figshare9970619-metadata-only-split-v1',
        'source': SOURCE,
        'policy': 'group-token split without fdt signal reads or prior cue knowledge',
        'metadata_manifest_sha256': digest,
        'metadata_only': True,
        'fit_allowed': False,
        'online_harness_allowed': False,
        'test_signal_files_opened': 0,
        'validation_signal_files_opened': 0,
        'train_signal_files_opened': 0,
        'unit_voltage_scale': None,
        'source_noncausal_preprocessing_unresolved': True,
        'cross_id_payload_must_link_unresolved': True,
        'event_gold_available': False,
        'blocked_reason': 'physical_V_calibration_missing_and_cross_id_must_link_unresolved',
        'aggregate': public,
        'items': items,
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--source',type=Path,required=True)
    a.add_argument('--receipt',type=Path,required=True)
    x=a.parse_args()
    result=prepare(x.source,host=socket.gethostname())
    if x.receipt.is_symlink() or x.receipt.exists():
        raise FileExistsError('cannot overwrite frozen split')
    x.receipt.parent.mkdir(parents=True,exist_ok=True)
    with x.receipt.open('x') as f:
        json.dump(result,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False))


if __name__=='__main__':
    main()
