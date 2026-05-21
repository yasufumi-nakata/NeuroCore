# NeuroCore Design Blueprint

## Purpose

NeuroCore is a runtime concept for EEG-driven control. It should feel like a Codex or Claude Code layer for EEG devices: a user wears a device, an external decoder estimates intent, and NeuroCore turns that intent into safe, inspectable mouse, keyboard, or AI-agent actions.

The important boundary is that NeuroCore does not own the model weights. Decoder training, calibration, and intent weighting happen elsewhere. NeuroCore owns the portable signal container, deterministic kernels, settings surface, action envelope, and breakage tests.

## Core Idea

NeuroCore separates neural control workflows into four layers:

1. Data normalization
2. Kernel execution
3. External intent ingestion
4. Safe action routing

Research code often mixes these concerns. NeuroCore should make them explicit so EEG pipelines remain portable across devices, montages, sampling rates, file formats, and compute environments.

## Primary Goals

- Provide a stable neural signal container for EEG-like time-series data.
- Normalize channel metadata, sampling rates, events, and annotations.
- Offer reusable kernels for preprocessing, feature extraction, decoding, and streaming.
- Run the same pipeline on CPU first, then optionally on GPU or accelerator backends.
- Make data incompatibilities explicit through typed validation errors.
- Route externally decoded intents into mouse, keyboard, and AI-agent action envelopes.
- Keep settings, safety gates, and breakage tests inspectable through CLI, API, and UI.

## Non-Goals

- NeuroCore is not a replacement for MNE, EEGLAB, FieldTrip, or BrainFlow.
- NeuroCore is not a medical diagnostic product.
- NeuroCore should not silently coerce incompatible EEG layouts.
- NeuroCore should not prioritize speed over traceability in early versions.
- NeuroCore does not train or ship intent-decoder weights.
- NeuroCore does not directly execute OS input or third-party agent side effects.

## Architecture

### 1. NeuroFrame

`NeuroFrame` is the canonical in-memory object.

It should contain:

- `data`: samples by channel, with explicit dimension labels
- `channels`: names, types, units, positions, and reference metadata
- `timebase`: sampling rate, timestamps, clock source, and drift metadata
- `events`: task markers, annotations, triggers, and artifacts
- `provenance`: source, preprocessing steps, versions, and hashes
- `validity`: compatibility checks and known limitations

### 2. Loaders

Loaders convert common data sources into `NeuroFrame`.

Initial targets:

- EDF / BDF
- FIF through MNE interoperability
- BrainVision
- BIDS / EEG-BIDS
- CSV / Parquet
- Lab Streaming Layer recordings
- BrainFlow-compatible streams

### 3. Kernels

Kernels are deterministic operations over `NeuroFrame` or lower-level signal tensors.

Kernel families:

- filtering
- resampling
- referencing
- artifact scoring
- epoching
- spectral features
- connectivity features
- embeddings
- decoding helpers
- streaming windows

Each kernel should declare:

- required channel types
- accepted sampling constraints
- output shape
- numerical precision
- backend support
- validation behavior

### 4. Backend Runtime

Backends execute kernels.

Initial backend priorities:

- NumPy CPU backend
- PyTorch backend
- CuPy or CUDA backend
- Metal backend for Apple Silicon, if practical

The CPU backend is the reference implementation. Accelerator backends must match CPU behavior within declared tolerances.

### 5. Compatibility Layer

NeuroCore should treat device variability as a first-class problem.

Examples:

- Montage mapping with explicit uncertainty
- Sampling-rate-aware kernel planning
- Channel alias resolution
- Reference scheme conversion
- Event alignment and clock drift handling
- Artifact-aware fallback paths

### 6. Intent and Control Layer

The decoder emits an `IntentCommand`:

- `intent`: symbolic intent such as `select`, `cursor_left`, or `agent_focus`
- `confidence`: decoder confidence from 0 to 1
- `payload`: optional structured context
- `source`: decoder name or stream source

`ControlRouter` checks:

- confidence threshold
- emergency stop
- action rate limit
- intent binding allowlist
- agent tool allowlist
- prompt-like payload blocking
- untrusted agent envelope requirement

The result is a `ControlAction`. It can be sent to a separate OS-input service or agent runtime after that service applies its own permissions.

`ActionAuditLog` and `DryRunActionSink` record routed actions without touching the OS. This is the default verification path for development and CI.

### 7. Settings UI

The local settings UI should expose:

- device profile
- channel list
- signal passband
- target sampling rate
- safety threshold
- emergency stop
- route simulator
- agent allowlist
- intent bindings
- self-test report

The UI is an operator console, not a landing page.

### 8. Breakage Tests

NeuroCore should test for system failure directly:

- invalid shape
- duplicate or missing channels
- NaN / Inf contamination
- filter plans above Nyquist
- drift and event alignment warnings
- low-confidence control suppression
- emergency stop suppression
- unsafe agent payload blocking

### 9. Streaming

Real-time acquisition should append samples into `StreamBuffer`. The buffer emits overlapping `StreamWindow` objects as soon as the configured window has enough samples. This keeps acquisition separate from preprocessing and decoding.

## API Sketch

```python
import neurocore as nc

frame = nc.load("sub-01_task-rest_eeg.edf")

pipeline = nc.Pipeline([
    nc.kernels.validate_eeg(),
    nc.kernels.resample(250),
    nc.kernels.bandpass(1, 40),
    nc.kernels.re_reference("average"),
    nc.kernels.spectral_features(bands=["theta", "alpha", "beta"]),
])

features = pipeline.run(frame, backend="cpu")
```

```python
from neurocore import ControlRouter, IntentCommand

router = ControlRouter()
action = router.route(IntentCommand("agent_focus", confidence=0.93, payload={"text": "open current task"}))
```

## Validation Philosophy

NeuroCore should prefer precise failure over implicit guessing.

Every pipeline run should answer:

- What data was loaded?
- What assumptions were made?
- Which channels were used or ignored?
- Which backend executed each kernel?
- What numerical tolerance applies?
- What compatibility warnings remain?

## Relationship to Existing Tools

NeuroCore should interoperate with existing tools rather than replace them.

- MNE: rich Python EEG/MEG analysis ecosystem
- EEGLAB and FieldTrip: established MATLAB research workflows
- BrainFlow: device acquisition and streaming layer
- BIDS / EEG-BIDS: dataset organization and metadata

NeuroCore's role is the portable runtime and kernel abstraction across these inputs.

## Milestones

### Milestone 0: Design Skeleton

- Define `NeuroFrame`.
- Define loader interface.
- Define kernel interface.
- Define validation error model.
- Define settings, control action envelope, and self-test report.

### Milestone 1: CPU Reference Runtime

- Load CSV and EDF-like examples.
- Run validation, resampling, filtering, and spectral features.
- Emit reproducible pipeline reports.
- Route external intents into safe action envelopes.
- Provide CLI/API/settings UI.

### Milestone 2: Interop

- Add MNE bridge.
- Add BIDS / EEG-BIDS loader.
- Add BrainFlow stream bridge.
- Add OS-input adapter as a separate permissioned service.
- Add Codex/Claude-style agent adapter as a separate permissioned service.

### Milestone 3: Accelerator Backends

- Add PyTorch backend.
- Add CuPy or CUDA backend.
- Compare backend output against CPU reference.

## Open Questions

- Should the public Python package name be `neurocore`, `neuro-core`, or a scoped variant?
- How close should the API be to array libraries versus EEG toolkits?
- Should real-time streaming be in core or a separate package?
- Which backend should be the first non-CPU target?
