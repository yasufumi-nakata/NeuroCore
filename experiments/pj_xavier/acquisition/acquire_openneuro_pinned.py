"""Acquire one authorized, version-pinned OpenNeuro release without alias loss.

Preserves the original queue's Git-annex/Git blob checksum notation. The
independent worker interprets and verifies that notation explicitly; an
unrecognized digest is a hard failure, never an unchecked download.
"""
from __future__ import annotations
import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit

CATALOG = '000233'
RELEASE = 'openneuro-ds005028-current-public-S3-e61cb6abf724'
DELEGATED = 'delegated_P300_v1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def immutable(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temp = path.with_name(path.name + '.part')
    require(not path.exists() and not temp.exists(), 'Receipt already exists')
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()


def checksum_semantics(checksum, expected_size):
    require(isinstance(expected_size, int) and not isinstance(expected_size, bool)
            and expected_size >= 0, 'Published size is not an integer')
    require(isinstance(checksum, dict) and checksum.get('algorithm') == 'unidentified',
            'Unexpected original OpenNeuro checksum scheme')
    value = checksum.get('value')
    require(isinstance(value, str), 'Published checksum value missing')
    match = re.fullmatch(r'sha256e-s(\d+)--([0-9a-f]{64})\.(dat|edf)', value)
    if match:
        require(int(match.group(1)) == expected_size, 'Git-annex key size differs')
        return {'algorithm': 'sha256', 'digest': match.group(2), 'scope': 'raw_file_bytes',
                'source_format': 'Git_annex_SHA256E'}
    if re.fullmatch(r'[0-9a-f]{40}', value):
        return {'algorithm': 'sha1', 'digest': value, 'scope': 'git_blob_header_plus_file_bytes',
                'source_format': 'Git_blob_SHA1'}
    raise ValueError('Unknown published checksum is not permitted')


def verify_file(path, expected_size, checksum):
    spec = checksum_semantics(checksum, expected_size)
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected_size,
            'File absent, symlink, or published size differs')
    before = path.stat()
    raw = hashlib.sha256()
    published = hashlib.new(spec['algorithm'])
    if spec['scope'] == 'git_blob_header_plus_file_bytes':
        published.update(b'blob ' + str(expected_size).encode('ascii') + b'\0')
    count = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            count += len(block)
            raw.update(block)
            published.update(block)
    after = path.stat()
    require(count == expected_size and (before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during verification')
    require(published.hexdigest() == spec['digest'], 'Published checksum mismatch')
    return raw.hexdigest()


IDENTITY_FIELDS = ('id', 'catalog_id', 'dataset_key', 'relative_path', 'url',
                   'expected_bytes', 'checksum_json', 'original_relative_path',
                   'provider_file_id', 'provider_file_identity')


def identity(row):
    return {k: row.get(k) for k in IDENTITY_FIELDS}


def validate_source_job(row, acquire):
    require(row['catalog_id'] == CATALOG and row['dataset_key'] == RELEASE, 'Source scope differs')
    url = urlsplit(row['url'])
    query = parse_qs(url.query, keep_blank_values=True)
    require(url.scheme == 'https' and url.netloc == 's3.amazonaws.com'
            and url.path.startswith('/openneuro.org/ds005028/') and not url.fragment
            and set(query) == {'versionId'} and len(query['versionId']) == 1
            and query['versionId'][0] not in ('', 'null'), 'Unpinned or noncanonical source URL')
    require(row['provider_file_identity'] == 'openneuro:url:' + row['url'], 'Provider identity differs')
    acquire.safe_relative(row['relative_path'])
    require(row['id'] == hashlib.sha256((RELEASE + '/' + row['relative_path']).encode()).hexdigest(),
            'Storage identity differs')
    spec = checksum_semantics(json.loads(row['checksum_json']), row['expected_bytes'])
    suffix = Path(row['relative_path']).suffix
    require((suffix in ('.dat', '.edf')) == (spec['source_format'] == 'Git_annex_SHA256E'),
            'Raw-data checksum representation differs')
    return spec


def delegate(original, selected, raw_root, acquire, output):
    require(selected and len({x['id'] for x in selected}) == len(selected), 'Empty or repeated jobs')
    original.row_factory = sqlite3.Row
    original.execute('BEGIN IMMEDIATE')
    try:
        saved = []
        for item in selected:
            found = original.execute('SELECT * FROM downloads WHERE id=?', (item['id'],)).fetchone()
            require(found is not None and identity(dict(found)) == identity(item), 'Original pinned identity changed')
            row = dict(found)
            require(row['status'] == 'queued' and row['lease_until'] == 0, 'Original job is not unleased queued')
            path = acquire.destination(raw_root, row['dataset_key'], row['relative_path'])
            require(not any(p.exists() or p.is_symlink() for p in
                            (path, path.with_name(path.name+'.part'), path.with_name(path.name+'.part.meta.json'))),
                    'Existing file or partial prevents new delegation')
            saved.append(row)
        immutable(Path(output)/'original_state.json', {'original_rows': saved, 'target_status': DELEGATED,
                                                     'created_epoch': time.time()})
        for item in selected:
            changed = original.execute("UPDATE downloads SET status=?,updated_at=? WHERE id=? AND status='queued' AND lease_until=0",
                                       (DELEGATED, acquire.now(), item['id'])).rowcount
            require(changed == 1, 'Concurrent queue update')
        original.commit()
    except BaseException:
        original.rollback()
        raise
    immutable(Path(output)/'delegation_receipt.json', {'status': 'delegated', 'job_ids': [x['id'] for x in selected],
              'source_state_sha256': sha(Path(output)/'original_state.json'), 'other_original_rows_changed': 0,
              'target_status': DELEGATED, 'completed_epoch': time.time()})


def reconcile(original, verified, raw_root, acquire, output):
    require(verified and len({x['id'] for x in verified}) == len(verified), 'Empty or repeated reconciliation')
    checked = []
    for row in verified:
        path = acquire.destination(raw_root, row['dataset_key'], row['relative_path'])
        digest = verify_file(path, row['expected_bytes'], json.loads(row['checksum_json']))
        require(row['status'] == 'complete' and row['sha256'] == digest, 'Downloaded content not verified complete')
        st = path.stat()
        checked.append((row, path, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)))
    original.row_factory = sqlite3.Row
    original.execute('BEGIN IMMEDIATE')
    try:
        for item, path, stat in checked:
            found = original.execute('SELECT * FROM downloads WHERE id=?', (item['id'],)).fetchone()
            require(found is not None and identity(dict(found)) == identity(item), 'Original pinned identity changed')
            row = dict(found)
            require(row['status'] == DELEGATED and row['lease_until'] == 0, 'Original job no longer delegated')
            st = path.stat()
            require(stat == (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns), 'Verified file changed')
            changed = original.execute("UPDATE downloads SET status='complete',bytes_present=?,sha256=?,error=NULL,updated_at=? WHERE id=? AND status=? AND lease_until=0",
                        (st.st_size, item['sha256'], acquire.now(), item['id'], DELEGATED)).rowcount
            require(changed == 1, 'Reconciliation race')
        original.commit()
    except BaseException:
        original.rollback()
        raise
    result = {'status': 'reconciled', 'files': len(checked), 'bytes': sum(x[0]['expected_bytes'] for x in checked),
              'other_original_rows_changed': 0, 'completed_epoch': time.time(),
              'checksum_semantics': {'data': 'Git-annex SHA256E content hash and embedded size',
                                     'metadata': 'SHA1(blob <size> NUL + bytes); not raw-file SHA1'}}
    immutable(Path(output)/'original_reconciliation.json', result)
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'worker.lock').open('xb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(Path(args.resource).read_text())
        frozen = json.loads(Path(args.frozen_jobs).read_text())
        require(budget['status'] == 'authorized_registered' and time.time() < budget['absolute_deadline_epoch'],
                'Budget absent or expired')
        require(budget['catalog_ids'] == [CATALOG] and budget['release_key'] == RELEASE
                and budget['concurrent_workers'] == 1, 'Budget scope differs')
        require(frozen['declared_public'] is True and frozen['license'] == 'CC0', 'Public licensing evidence missing')
        selected = frozen['jobs']
        require(len(selected) == budget['expected_files'] and sum(x['expected_bytes'] for x in selected)
                == budget['known_source_bytes'], 'Frozen file count or whole release bytes differ')
        require(len({x['id'] for x in selected}) == len(selected), 'Duplicate frozen jobs')
        reserve = budget['effective_download_reserve_bytes']
        require(shutil.disk_usage(args.raw_root).free >= reserve + budget['known_source_bytes'], 'Release does not fit reserve')
        spec = importlib.util.spec_from_file_location('fixed_p300_acquirer', args.acquire_source)
        acquire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(acquire)
        require(sha(args.acquire_source) == frozen['acquire_source_sha256'], 'Acquirer source changed')
        formats = collections.Counter(validate_source_job(x, acquire)['source_format'] for x in selected)
        acquire.verify_file = verify_file
        db = acquire.init_db(output/'downloads.sqlite')
        columns = [x[1] for x in db.execute('PRAGMA table_info(downloads)')]
        for job in selected:
            require(job['status'] == 'queued' and job['lease_until'] == 0, 'Frozen original state not queued')
            db.execute('INSERT INTO downloads ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',
                       [job[k] for k in columns])
        db.commit()
        original = sqlite3.connect(args.original_db, timeout=30)
        delegate(original, selected, args.raw_root, acquire, output)
        immutable(output/'worker_launch.json', {'pid': os.getpid(), 'started_epoch': time.time(),
                  'absolute_deadline_epoch': budget['absolute_deadline_epoch'], 'source_sha256': sha(__file__),
                  'acquire_source_sha256': sha(args.acquire_source), 'resource_sha256': sha(args.resource),
                  'frozen_jobs_sha256': sha(args.frozen_jobs), 'checksum_formats': dict(formats),
                  'expected_file_bytes': budget['known_source_bytes'], 'reserve_bytes': reserve,
                  'maximum_received_payload_bytes': budget['maximum_received_payload_bytes']})
        traffic = 0
        attempts = []
        while time.time() < budget['absolute_deadline_epoch'] and traffic < budget['maximum_received_payload_bytes']:
            result = acquire.worker(db, args.raw_root, reserve_gib=reserve/2**30,
                       seconds=max(1, math.floor(budget['absolute_deadline_epoch']-time.time())),
                       max_bytes=budget['maximum_received_payload_bytes']-traffic)
            traffic += result['bytes_transferred_this_run']
            attempts.append(result)
            states = {x[0] for x in db.execute('SELECT status FROM downloads')}
            print(json.dumps({'epoch': time.time(), 'received_bytes': traffic, 'states': sorted(states)}), flush=True)
            if states == {'complete'} or states-{'queued', 'complete', 'paused', 'retry'}:
                break
            next_retry = db.execute("SELECT min(retry_at) FROM downloads WHERE status IN ('queued','paused','retry')").fetchone()[0]
            if next_retry is None:
                break
            delay = min(30, max(1, next_retry-time.time()), max(0, budget['absolute_deadline_epoch']-time.time()))
            if delay:
                time.sleep(delay)
        db.row_factory = sqlite3.Row
        final = [dict(x) for x in db.execute('SELECT * FROM downloads ORDER BY id')]
        complete = all(x['status'] == 'complete' for x in final)
        if complete:
            for row in final:
                require(verify_file(acquire.destination(args.raw_root, row['dataset_key'], row['relative_path']),
                        row['expected_bytes'], json.loads(row['checksum_json'])) == row['sha256'], 'Final hash differs')
        receipt = {'status': 'all_files_downloaded_checksum_verified' if complete else 'incomplete',
                   'files': len(final), 'complete_files': sum(x['status'] == 'complete' for x in final),
                   'received_payload_bytes': traffic, 'attempts': attempts, 'header_audit_pending': True,
                   'original_reconciliation_performed': False, 'completed_epoch': time.time()}
        immutable(output/'download_completion.json', receipt)
        return receipt


def main():
    p = argparse.ArgumentParser()
    for name in ('frozen-jobs', 'resource', 'acquire-source', 'original-db', 'raw-root', 'output'):
        p.add_argument('--'+name, required=True)
    print(json.dumps(run(p.parse_args()), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
