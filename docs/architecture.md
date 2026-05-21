# NeuroCore Architecture

NeuroCore is organized as a small runtime for EEG-to-control systems. The decoder that assigns weights or decides a user's intent lives outside this repository.

## Runtime Boundaries

1. `NeuroFrame` normalizes samples, channel metadata, events, timebase, provenance, and validation issues.
2. Kernels perform deterministic CPU reference operations such as validation, resampling, FFT bandpass, average reference, and spectral features.
3. `Pipeline` runs kernels and emits a step report with backend, timing, warnings, and output shape.
4. `ControlRouter` accepts an externally decoded `IntentCommand` and converts it into mouse, keyboard, or AI-agent action envelopes.
5. `StreamBuffer` turns incoming sample chunks into overlapping `StreamWindow` frames.
6. `ActionAuditLog` and `DryRunActionSink` record routed actions without touching the OS.
7. `run_self_tests()` injects synthetic breakage to verify that invalid signal plans and unsafe controls are blocked.

## No Bundled Weights

NeuroCore does not train, ship, or tune classifier weights. This is intentional. A BCI decoder can output:

```json
{
  "intent": "select",
  "confidence": 0.91,
  "payload": {}
}
```

NeuroCore then checks confidence, rate limits, emergency stop state, binding allowlists, and agent-payload safety. This keeps the runtime testable even when the decoding model changes.

## AI-Agent Boundary

Decoded EEG text or intent payloads are untrusted. Agent actions are wrapped as:

```json
{
  "tool": "open_task",
  "intent": "agent_focus",
  "confidence": 0.93,
  "payload": {},
  "trust": "untrusted_decoded_intent",
  "execution": "requires_agent_policy"
}
```

The receiving agent must still apply its own permissions. NeuroCore only prepares a safe envelope; it does not execute shell commands, click the OS, or send data to third-party services.

## Dry-run First

During development, route decoded intents through `DryRunActionSink`. This records every allowed or blocked action as JSONL and keeps OS input, browser actions, and agent side effects outside the core runtime.

## Streaming

`StreamBuffer` accepts sample chunks shaped as samples by channels. Once enough data is buffered, it emits `StreamWindow` objects that contain a `NeuroFrame`, start sample, and end sample. Downstream decoders can consume windows without knowing the acquisition device.
