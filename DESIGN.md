# NeuroCore Design Blueprint

## Purpose

NeuroCore is a CUDA-like runtime concept for EEG and neural time-series data. The goal is not to copy CUDA directly, but to provide the same kind of stable abstraction: users describe signal operations once, and NeuroCore maps them onto compatible data layouts, devices, and compute backends.

The practical promise is simple: if the input is valid neural time-series data, the workflow should either run correctly or fail with a precise explanation.

## Core Idea

NeuroCore separates neural data workflows into three layers:

1. Data normalization
2. Kernel execution
3. Backend optimization

Research code often mixes these concerns. NeuroCore should make them explicit so EEG pipelines remain portable across devices, montages, sampling rates, file formats, and compute environments.

## Primary Goals

- Provide a stable neural signal container for EEG-like time-series data.
- Normalize channel metadata, sampling rates, events, and annotations.
- Offer reusable kernels for preprocessing, feature extraction, decoding, and streaming.
- Run the same pipeline on CPU first, then optionally on GPU or accelerator backends.
- Make data incompatibilities explicit through typed validation errors.

## Non-Goals

- NeuroCore is not a replacement for MNE, EEGLAB, FieldTrip, or BrainFlow.
- NeuroCore is not a medical diagnostic product.
- NeuroCore should not silently coerce incompatible EEG layouts.
- NeuroCore should not prioritize speed over traceability in early versions.

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

### Milestone 1: CPU Reference Runtime

- Load CSV and EDF-like examples.
- Run validation, resampling, filtering, and spectral features.
- Emit reproducible pipeline reports.

### Milestone 2: Interop

- Add MNE bridge.
- Add BIDS / EEG-BIDS loader.
- Add BrainFlow stream bridge.

### Milestone 3: Accelerator Backends

- Add PyTorch backend.
- Add CuPy or CUDA backend.
- Compare backend output against CPU reference.

## Open Questions

- Should the public Python package name be `neurocore`, `neuro-core`, or a scoped variant?
- How close should the API be to array libraries versus EEG toolkits?
- Should real-time streaming be in core or a separate package?
- Which backend should be the first non-CPU target?

