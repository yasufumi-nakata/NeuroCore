#!/usr/bin/env python3
"""Inspect observed action markers in an EEGLAB recording; no inferred intention."""
from __future__ import annotations
import argparse
import bisect
import json
from collections import Counter
from pathlib import Path


def class_of_press(description: str) -> str | None:
    txt = description.strip()
    if not txt.endswith(' pressed'):
        return None
    if txt.startswith('MouseButton'):
        return 'mouse_click'
    if txt.startswith('MouseWheel'):
        return 'mouse_scroll'
    return 'keyboard'


def count_preaction_windows(annotations: list[tuple[float, str]], duration: float, history_seconds: float = 2, guard_seconds: float = 0.25) -> dict:
    if not (history_seconds > guard_seconds >= 0):
        raise ValueError('Invalid history and guard time')
    times = sorted(t for t,_ in annotations)
    presses = [(t, cls) for t,desc in annotations if (cls:=class_of_press(desc)) and history_seconds <= t <= duration]
    valid = Counter()
    clean = Counter()
    for t, cls in presses:
        valid[cls] += 1
        # Nothing else occurred in the EEG history; otherwise it may reflect the previous physical action.
        a=bisect.bisect_left(times,t-history_seconds)
        b=bisect.bisect_left(times,t-guard_seconds)
        if a==b:
            clean[cls] += 1
    return {
        'observed_action_targets':sum(valid.values()),
        'targets_by_class':dict(valid),
        'history_with_no_other_recorded_events':sum(clean.values()),
        'clean_history_by_class':dict(clean),
        'history_seconds':history_seconds,
        'guard_seconds':guard_seconds,
        'interpretation':'observed next action, not free-form user intent; no environment snapshots in this record',
    }


def inspect(set_path: Path) -> dict:
    import mne
    import numpy as np
    raw=mne.io.read_raw_eeglab(str(set_path),preload=False,verbose='ERROR')
    ann=[(float(v['onset']),str(v['description'])) for v in raw.annotations]
    duration=raw.n_times/float(raw.info['sfreq'])
    first=raw.get_data(start=0,stop=round(float(raw.info['sfreq'])*10))
    return {
        'dataset':'EEGlass PC tasks, Zenodo 4453396',
        'url':'https://zenodo.org/records/4453396',
        'file':set_path.name,
        'sampling_frequency':float(raw.info['sfreq']),
        'channels':len(raw.ch_names),
        'duration_seconds':duration,
        'first_10s_shape':list(first.shape),
        'first_10s_finite':bool(np.isfinite(first).all()),
        'all_events':len(ann),
        'top_event_types':Counter(d for _,d in ann).most_common(15),
        'action_label_probe':count_preaction_windows(ann,duration),
        'subjects_verified_in_this_release':1,
        'free_text_goal_labels':False,
        'screen_snapshots_in_this_release':False,
        'learning_target_type':'observed action, not user intent',
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--eeglab',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    d=inspect(args.eeglab)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(d,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
