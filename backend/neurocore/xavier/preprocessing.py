"""Frozen source30 causal reference/filter for inference. Units are microvolts.

The first second initializes the state from the past and is discarded.
No zero-phase filtering, future padding, or physical clock certification.
"""
import numpy as np
import hashlib
import json
from pathlib import Path

FILTER_SHA256 = 'ab896fb5b6313fb585c33d52ff6e3599ae0d39e9c7e1b18616aa7ec29ed5a4fe'

EXPECTED_NAMES = ('Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1',
                  'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz',
                  'C4','T8','FT10','FC6','FC2','F4','F8','Fp2','EP1','EP2')
REFERENCE_INDICES = (9, 20)
CORTICAL_INDICES = tuple(i for i in range(32) if i not in REFERENCE_INDICES)


def cortical_reference(native_uv, names):
    if tuple(names) != EXPECTED_NAMES or native_uv.ndim != 2 or native_uv.shape[1] != 34:
        raise ValueError('fixed verified montage required')
    if not np.isfinite(native_uv).all():
        raise ValueError('nonfinite input cannot be repaired silently')
    reference = (native_uv[:, 9] + native_uv[:, 20]) * 0.5
    return native_uv[:, CORTICAL_INDICES] - reference[:, None]


class CausalPipeline:
    """10kHz -> causal FIR/decimate50 -> forward SOS1-40Hz ->200Hz.

    State resets at each fixed three-second epoch. The first sample supplies
    a constant-past initial state, then the first second is excluded. It is
    never presented as continuous-session filtering or hardware synchronization.
    """
    def __init__(self, coefficients, channels=30):
        if channels != 30 or coefficients['input_hz'] != 10000 or coefficients['output_hz'] != 200:
            raise ValueError('fixed source-rate contract')
        self.fir = np.asarray(coefficients['fir'], dtype=np.float64)
        self.sos = np.asarray(coefficients['sos'], dtype=np.float64)
        self.unit_zi = np.asarray(coefficients['sos_initial_unit_step_state'], dtype=np.float64)
        if self.fir.shape != (1001,) or self.sos.shape != (3, 6) or not np.all(self.sos[:, 3] == 1):
            raise ValueError('fixed coefficient geometry')
        self.channels = channels
        self.decimation = 50
        self.warmup = 200
        self.history = np.zeros((1000, channels), dtype=np.float64)
        self.zi = np.zeros((3, 2, channels), dtype=np.float64)
        self.raw_seen = 0
        self.output_seen = 0
        self.initialized = False

    def process(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channels or not np.isfinite(values).all():
            raise ValueError('finite cortical microvolt stream required')
        if len(values) == 0:
            return np.empty((0, self.channels)), np.empty(0, dtype=np.int64)
        if not self.initialized:
            self.history[:] = values[0]
        joined = np.concatenate((self.history, values), axis=0)
        first = (-self.raw_seen) % self.decimation
        windows = np.lib.stride_tricks.sliding_window_view(joined, len(self.fir), axis=0)[first::50]
        reduced = np.einsum('nck,k->nc', windows, self.fir[::-1], optimize=False)
        indices = self.raw_seen + np.arange(first, len(values), 50, dtype=np.int64)
        self.history = joined[-1000:].copy()
        self.raw_seen += len(values)
        if len(reduced) and not self.initialized:
            self.zi = self.unit_zi[:, :, None] * reduced[0][None, None, :]
            self.initialized = True
        filtered = np.empty_like(reduced)
        for n, row in enumerate(reduced):
            value = row
            for section, coef in enumerate(self.sos):
                result = coef[0] * value + self.zi[section, 0]
                self.zi[section, 0] = coef[1] * value - coef[4] * result + self.zi[section, 1]
                self.zi[section, 1] = coef[2] * value - coef[5] * result
                value = result
            filtered[n] = value
        keep = np.arange(self.output_seen, self.output_seen + len(reduced)) >= self.warmup
        self.output_seen += len(reduced)
        return filtered[keep], indices[keep]

    def state(self):
        return {'history': self.history.copy(), 'zi': self.zi.copy(),
                'raw_seen': np.asarray(self.raw_seen, dtype=np.int64),
                'output_seen': np.asarray(self.output_seen, dtype=np.int64),
                'initialized': np.asarray(int(self.initialized), dtype=np.int8)}

    def restore(self, state):
        history = np.asarray(state['history']); zi = np.asarray(state['zi'])
        if history.shape != (1000, 30) or zi.shape != (3, 2, 30) or not np.isfinite(history).all() or not np.isfinite(zi).all():
            raise ValueError('state dimensions/finite values')
        raw_seen = int(state['raw_seen']); output_seen = int(state['output_seen'])
        if raw_seen < 1 or output_seen != (raw_seen + 49) // 50 or int(state['initialized']) != 1:
            raise ValueError('state order/decimation phase')
        self.history = history.copy(); self.zi = zi.copy()
        self.raw_seen = raw_seen; self.output_seen = output_seen; self.initialized = True


def prepare_native_prefix(native_uv, *, channels, sample_rate_hz=10000, units='uV'):
    """Prepare 1.5 s of model context from exactly 2.5 s of native EEG.

    The source30 artifact has one fixed montage, reference, sampling rate and
    causal filter. This function never guesses another sensor's channel mapping.
    The returned JSON-compatible request contains no future target samples.
    Neither the input nor the output is written to disk by this function.
    """
    if tuple(channels) != EXPECTED_NAMES or sample_rate_hz != 10000 or units != 'uV':
        raise ValueError('source30 preparation requires its ordered34-channel 10000Hz microvolt contract')
    native = np.asarray(native_uv)
    if native.dtype.kind not in 'iuf' or native.shape != (25000, 34):
        raise ValueError('native_uv must contain exactly25000 finite samples by34 channels')
    native = native.astype(np.float64)
    if not np.isfinite(native).all():
        raise ValueError('native_uv contains non-finite samples')
    data = (Path(__file__).parent/'assets'/'causal30_filter.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != FILTER_SHA256:
        raise ValueError('fixed causal filter asset hash mismatch')
    pipeline = CausalPipeline(json.loads(data))
    processed, native_indices = pipeline.process(cortical_reference(native, channels))
    if processed.shape != (300, 30) or not np.array_equal(native_indices, np.arange(10000, 25000, 50)):
        raise RuntimeError('causal preprocessing sample alignment failed')
    prefix = processed.astype(np.float32)
    if not np.isfinite(prefix).all():
        raise ValueError('preprocessed input is outside float32 range')
    return {
        'prefix': prefix.tolist(),
        'channels': [EXPECTED_NAMES[i] for i in CORTICAL_INDICES],
        'sample_rate_hz': 200,
        'units': 'uV',
        'preprocessing': {
            'id': 'ds006735_causal30_v1',
            'filter_coefficients_sha256': FILTER_SHA256,
            'native_sample_rate_hz': 10000,
            'native_samples': 25000,
            'warmup_seconds': 1.0,
            'context_seconds': 1.5,
            'hardware_clock_certified': False,
            'physical_raw_timing_certified': False,
        },
    }
