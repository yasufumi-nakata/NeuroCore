#!/usr/bin/env python3
"""Apple-Silicon smoke test: real EEG -> trainable cross-attention -> observed actions.

The environment token is a placeholder, not recorded screen context.  This is
NOT UniMind/REVE and does NOT evaluate intent accuracy or generalization.
"""
from __future__ import annotations

import argparse
import bisect
import json
import time
from collections import defaultdict
from pathlib import Path

from eeglab_action_probe import class_of_press


def prepare_observed_action_data(set_file: Path, items_per_class: int = 16) -> tuple:
    import mne
    import numpy as np
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose='ERROR')
    sr = int(raw.info['sfreq'])
    if sr != 250:
        raise RuntimeError('This read-only smoke test expects the 250 Hz EEGlass release')
    ann = [(float(a['onset']), a['description']) for a in raw.annotations]
    all_event_times = sorted(t for t, _ in ann)
    classes = ('keyboard', 'mouse_click', 'mouse_scroll')
    samples: dict[str, list] = defaultdict(list)
    for t, desc in ann:
        cls = class_of_press(desc)
        if cls not in classes or t < 2 or len(samples[cls]) >= items_per_class:
            continue
        if bisect.bisect_left(all_event_times, t - 2) != bisect.bisect_left(all_event_times, t - .25):
            continue
        begin = round((t - 2) * sr)
        end = begin + round(1.75 * sr)
        arr = raw.get_data(start=begin, stop=end)
        if arr.shape != (3, round(1.75 * sr)) or not np.isfinite(arr).all():
            continue
        arr = arr.T.astype('float32', copy=False)
        mu = np.mean(arr, axis=0, keepdims=True)
        sd = np.maximum(np.std(arr, axis=0, keepdims=True), 1e-8)
        samples[cls].append((arr - mu) / sd)
    if min(map(lambda cls: len(samples[cls]), classes)) < 4:
        raise RuntimeError('Too few isolated action markers for a smoke test')
    n = min(len(samples[cls]) for cls in classes)
    X = np.stack([a for cls in classes for a in samples[cls][:n]], axis=0)
    y = np.array([j for j, _ in enumerate(classes) for _ in range(n)], dtype='int32')
    return X, y, classes


def train_smoke(X, y, steps: int = 20) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    import numpy as np

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(3, 32, kernel_size=11, stride=5, padding=5)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=5, padding=2)
            self.proj = nn.Linear(64, 128)
            self.context_token = nn.Embedding(1, 128)
            self.cross_attn = nn.MultiHeadAttention(128, 4)
            self.action_head = nn.Linear(128, 3)

        def __call__(self, signal, ctx):
            encoded = self.proj(nn.relu(self.conv2(nn.relu(self.conv1(signal)))))
            query = self.context_token(ctx)[:, None, :]
            return self.action_head(self.cross_attn(query, encoded, encoded)[:, 0, :])

    model = Fusion()
    optimizer = opt.AdamW(learning_rate=2e-3)

    def loss_function(m, eeg, context, labels):
        return nn.losses.cross_entropy(m(eeg, context), labels).mean()

    val_grad = nn.value_and_grad(model, loss_function)
    rng = np.random.default_rng(421)
    indices = rng.permutation(len(y))
    eeg = mx.array(X[indices])
    targets = mx.array(y[indices])
    context = mx.zeros((len(targets),), dtype=mx.int32)  # neutral placeholder only
    start = time.perf_counter()
    initial = None
    for i in range(steps):
        idx = mx.array([(i * 8 + j) % len(targets) for j in range(8)], dtype=mx.int32)
        loss, grads = val_grad(model, eeg[idx], context[idx], targets[idx])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_value = float(loss)
        if i == 0:
            initial = loss_value
    elapsed = time.perf_counter() - start
    return {
        'accelerator': 'Apple Silicon MLX GPU',
        'tiny_model': '3ch temporal convolution -> 128d cross-attention -> 3-way observed action',
        'input_source': 'real EEGlass 250Hz EEG, isolated pre-action windows',
        'environment_conditioning': 'neutral placeholder only; real screen context unavailable',
        'n_labelled_examples': int(len(y)),
        'tensor_shape': list(X.shape),
        'steps': steps,
        'wall_seconds': round(elapsed, 3),
        'initial_loss': round(initial, 5),
        'last_loss': round(loss_value, 5),
        'intention_recovery_tested': False,
        'llm_finetuning_tested': False,
        'generalization_tested': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeglab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    X,y,classes=prepare_observed_action_data(args.eeglab)
    result=train_smoke(X,y)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
