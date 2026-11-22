# Governance

This NeuroCore repository is maintained as a research runtime. pj-xavier (Project Xavier) is responsible for researching and developing NeuroCore's AI; keeping AI models outside the core package is an implementation boundary. Maintainers prioritize:

- Explicit safety boundaries over convenience.
- Deterministic tests over uninspected model behavior.
- Local-first development and dry-run action logs.
- Clear separation between core runtime, decoders, OS input services, and AI-agent services.

Breaking changes should update README, DESIGN, tutorial pages, tests, and API examples together.
