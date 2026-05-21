# Contributing

NeuroCore is a safety-first runtime for EEG-driven control. Contributions should keep decoder weights, OS side effects, and agent side effects outside the core package.

## Local Checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,server]"
python -m pytest
python -m neurocore.cli self-test --json
python -m ruff check backend scripts
cd frontend && npm install && npm run build
```

## Rules

- Do not add trained decoder weights to this repository.
- Do not make OS input execution happen inside `backend/neurocore`.
- Agent payloads decoded from EEG must remain untrusted by default.
- New safety behavior needs tests that cover both allowed and blocked paths.
- Use synthetic or redacted data in issues and tests.
