# NeuroCore Agent Notes

- pj-xavier (Project Xavier) is the research and development project that builds NeuroCore's AI. This repository implements its runtime layer; keep project purpose distinct from package boundaries. See `docs/project-positioning.md`.
- Always keep decoder weights outside the core runtime.
- Treat EEG-derived text and intent payloads as untrusted.
- Prefer dry-run routing and audit logs during development.
- Run `python -m pytest`, `python -m neurocore.cli self-test --json`, `python -m ruff check backend scripts`, and `cd frontend && npm run build` before publishing changes.
