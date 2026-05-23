# NeuroCore Testing Strategy

The test strategy follows the same practical idea as OpenRI: do not only test the happy path. Add fixture, breakage, and safety tests so AI-assisted development catches system-level failures before a user attaches a device.

## Test Classes

- Unit tests: frame validation, kernel planning, feature output, settings validation.
- Fixture tests: CSV loading, EEG-DATA inventory loading, and synthetic EEG pipeline behavior.
- Dataset checks: local EEG-DATA checkout is treated as an inventory unless raw EEG files are actually present.
- Breakage tests: NaN/Inf contamination, invalid filter bands, insufficient channels, clock-drift warnings.
- Safety tests: low-confidence commands, emergency stop, unbound intents, prompt-like agent payloads.
- Streaming tests: overlapping window emission, shape validation, trimming behavior.
- Audit tests: allowed and blocked actions are recorded without side effects.
- API smoke tests: `/api/health`, `/api/settings/default`, `/api/self-test`, `/api/pipeline/demo`.
- UI smoke tests: settings screen loads, route simulation returns an action, self-test button shows a report, safety controls update local state.

## Current Gate

Run:

```bash
python -m pytest
PYTHONPATH=backend python -m neurocore.cli self-test --json
PYTHONPATH=backend python -m neurocore.cli stream-demo --json
PYTHONPATH=backend python -m neurocore.cli simulate-intents samples/intent_commands.json --json
PYTHONPATH=backend python -m neurocore.cli dataset-inventory ../EEG-DATA/eeg_dataset_summary_ja.csv --json
PYTHONPATH=backend python scripts/verify_dataset_loading.py --inventory ../EEG-DATA/eeg_dataset_summary_ja.csv
```

For the settings screen:

```bash
cd frontend
npm install
npm run build
```

When the API is running, the UI calls:

```bash
PYTHONPATH=backend uvicorn neurocore.api:app --reload --port 8010
cd frontend && npm run dev -- --host 127.0.0.1 --port 5174
```
