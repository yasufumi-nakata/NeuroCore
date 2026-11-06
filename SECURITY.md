# Security Policy

Do not report vulnerabilities through public issues when they involve local control, agent execution, credentials, private EEG data, or unpublished research data.

Use GitHub private vulnerability reporting when available, or contact the maintainers privately.

## Scope

- Intent routing and action-envelope bypasses.
- Prompt-like payload bypasses.
- Unsafe OS-input or agent side-effect paths.
- API upload or CORS issues.
- Dependency and workflow vulnerabilities.

## Serialized EEG data

The `.pkl` and `.pickle` loaders use a restricted unpickler. They accept the built-in containers and NumPy arrays needed for numeric EEG payloads, but reject arbitrary pickle globals, callables, and object arrays before converting data into a `NeuroFrame`. This is a compatibility boundary, not a guarantee that a hostile file is harmless: pickle payloads can still be malformed or resource-intensive. Treat downloaded or otherwise untrusted serialized data as hostile, verify its source and checksum, and load it in a disposable or separately permissioned process before using it in a long-lived research runtime.

NeuroCore is not a medical device and does not provide clinical diagnosis.
