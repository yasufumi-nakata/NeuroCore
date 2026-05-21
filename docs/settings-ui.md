# Settings UI

The frontend is a local operator console for early experiments. It is intentionally a settings-first tool, not a marketing page.

## Surface

- Device profile: sampling rate and channel list.
- Signal plan: passband, target sampling rate, window size, and step size.
- Safety policy: confidence threshold, rate limit, human-arm requirement, emergency stop, and prompt-like agent payload blocking.
- Route simulator: send one decoded intent to the backend and inspect the resulting action envelope.
- Bindings: decoded intents mapped to mouse, keyboard, or agent action envelopes.
- Self-test panel: synthetic pipeline and breakage tests from the backend API.

The UI can render without the API and shows seed self-test data, but real validation should use the FastAPI server.
