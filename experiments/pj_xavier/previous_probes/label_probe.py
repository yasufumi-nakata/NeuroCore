#!/usr/bin/env python3
"""Read-only EEG annotation probe. Never infers user intent from a clinical label."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def chbmit_events(text: str, edf_filename: str) -> list[tuple[float, float]]:
    # CHB-MIT summary has a separate "File Name:" section for each EDF.
    sections = re.split(r'(?=^File Name:\s*)', text, flags=re.M)
    matched = []
    for section in sections:
        title = re.search(r'^File Name:\s*(\S+)', section, flags=re.M)
        if not title or title.group(1) != edf_filename:
            continue
        count_match = re.search(r'^Number of Seizures in File:\s*(\d+)', section, flags=re.M)
        if not count_match:
            raise ValueError('Missing seizure count in summary')
        expected = int(count_match.group(1))
        starts = [int(v) for v in re.findall(r'^Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s+seconds', section, flags=re.M)]
        ends = [int(v) for v in re.findall(r'^Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s+seconds', section, flags=re.M)]
        if len(starts) != expected or len(ends) != expected:
            raise ValueError(f'Annotation mismatch: expected={expected}; starts={starts}; ends={ends}')
        if any(b <= a for a, b in zip(starts, ends)):
            raise ValueError('Invalid seizure interval')
        matched.append(list(zip(starts, ends)))
    if len(matched) != 1:
        raise ValueError(f'Expected exactly one section for {edf_filename}, found {len(matched)}')
    return matched[0]


def event_window_classification(duration_s: float, events: list[tuple[float, float]], window_s=10.0, min_fraction=0.5) -> dict:
    if window_s <= 0 or not (0 < min_fraction <= 1):
        raise ValueError('Invalid window settings')
    windows = int(duration_s // window_s)
    labels = []
    for i in range(windows):
        start, stop = i * window_s, (i + 1) * window_s
        overlap = sum(max(0, min(stop, e) - max(start, s)) for s, e in events)
        overlap = min(overlap, window_s)
        # If the boundary window is only partly positive, mark unknown instead of inventing a label.
        label = 'positive' if overlap / window_s >= min_fraction else ('negative' if overlap == 0 else 'ambiguous')
        labels.append({'window_index': i, 'start_seconds': start, 'duration_seconds': window_s, 'label': label})
    return {
        'n_windows': len(labels),
        'positive': sum(x['label'] == 'positive' for x in labels),
        'negative': sum(x['label'] == 'negative' for x in labels),
        'ambiguous': sum(x['label'] == 'ambiguous' for x in labels),
        'positive_window_starts_seconds': [x['start_seconds'] for x in labels if x['label'] == 'positive'],
        'ambiguous_window_starts_seconds': [x['start_seconds'] for x in labels if x['label'] == 'ambiguous'],
    }


def analyze(edf: Path, summary: Path) -> dict:
    import mne
    import numpy as np
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose='ERROR')
    duration = raw.n_times / raw.info['sfreq']
    events = chbmit_events(summary.read_text(errors='replace'), edf.name)
    if any(not (0 <= start < stop <= duration) for start, stop in events):
        raise ValueError('Annotations outside the EDF duration')
    sample = raw.get_data(start=0, stop=int(raw.info['sfreq'] * 10))
    return {
        'source': 'CHB-MIT Scalp EEG Database v1.0.0',
        'official_url': 'https://physionet.org/content/chbmit/1.0.0/',
        'file': edf.name,
        'sha256': sha256_file(edf),
        'summary_sha256': sha256_file(summary),
        'sampling_rate_hz': float(raw.info['sfreq']),
        'recording_seconds': duration,
        'channels': len(raw.ch_names),
        'eeg_channels': sum(t == 'eeg' for t in raw.get_channel_types()),
        'channels_bipolar_montage': sum('-' in name for name in raw.ch_names),
        'window_shape_channels_times_points': list(sample.shape),
        'first_window_is_finite': bool(np.isfinite(sample).all()),
        'event_intervals_seconds': [{'start':a,'end':b} for a,b in events],
        'label_kind': 'clinical_annotation_not_user_intent',
        'window_labels': event_window_classification(duration, events),
        'can_reuse_directly_as_action_intent_supervision': False,
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--edf',type=Path,required=True)
    ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=analyze(args.edf,args.summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
