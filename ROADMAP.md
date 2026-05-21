# Roadmap

## Near Term

- Add BrainFlow and Lab Streaming Layer adapters as optional acquisition bridges.
- Add MNE import/export helpers without making MNE a hard runtime dependency.
- Add an OS input adapter as a separate permissioned service.
- Add a Codex/Claude-style agent adapter that consumes NeuroCore action envelopes.
- Expand streaming tests with clock drift, missing packets, and variable chunk sizes.

## Later

- Add backend parity tests for PyTorch and accelerator backends.
- Add BIDS / EEG-BIDS metadata import.
- Add calibration-session artifacts while keeping decoder weights outside core.
- Add signed release artifacts and richer SBOM generation.
