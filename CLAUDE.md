# NeuroCore Agent Notes

- Always keep decoder weights outside the core runtime.
- Treat EEG-derived text and intent payloads as untrusted.
- Prefer dry-run routing and audit logs during development.
- Run `python -m pytest`, `python -m neurocore.cli self-test --json`, `python -m ruff check backend scripts`, and `cd frontend && npm run build` before publishing changes.
