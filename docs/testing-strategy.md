# NeuroCore Testing Strategy

The test strategy follows the same practical idea as OpenRI: do not only test the happy path. Add fixture, breakage, and safety tests so AI-assisted development catches system-level failures before a user attaches a device.

## Test Classes

- Unit tests: frame validation, kernel planning, feature output, settings validation.
- Fixture tests: CSV loading, EEG-DATA inventory loading, and synthetic EEG pipeline behavior.
- Dataset checks: local EEG-DATA checkout is treated as an inventory unless raw EEG files are actually present; the inventory report also summarizes loader coverage for MNE-backed EEG formats, XDF, MAT, NumPy, and CSV.
- Acquisition checks: provider-specific raw-data acquisition plans are generated for EEG-DATA rows, public provider APIs and file indexes can be resolved to remote file candidates, and any local raw cache can be loaded through the same `NeuroFrame` path.
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
PYTHONPATH=backend python -m neurocore.cli run-file samples/synthetic_eeg.csv --sampling-rate 250 --json
PYTHONPATH=backend python -m neurocore.cli dataset-inventory ../EEG-DATA/eeg_dataset_summary_ja.csv --json
PYTHONPATH=backend python -m neurocore.cli dataset-acquisition-plan ../EEG-DATA/eeg_dataset_summary_ja.csv --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --limit 10 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider openneuro --provider dandi --provider kaggle --provider nemar --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider github --provider huggingface --provider physionet --provider gin --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider stanford_sdr --provider data_ru --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider scidb --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider figshare --provider dataverse --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider bnci --provider repository_html --limit 5 --json
PYTHONPATH=backend python -m neurocore.cli dataset-resolve-files ../EEG-DATA/eeg_dataset_summary_ja.csv --provider doi --provider web_landing --limit 5 --json
PYTHONPATH=backend python scripts/verify_dataset_loading.py --inventory ../EEG-DATA/eeg_dataset_summary_ja.csv
PYTHONPATH=backend python scripts/verify_dataset_acquisition.py --inventory ../EEG-DATA/eeg_dataset_summary_ja.csv
PYTHONPATH=backend python scripts/verify_dataset_acquisition.py --inventory ../EEG-DATA/eeg_dataset_summary_ja.csv --resolve-remote-files --resolve-limit 10
PYTHONPATH=backend python scripts/verify_dataset_acquisition.py --inventory ../EEG-DATA/eeg_dataset_summary_ja.csv --cache-root private/raw-cache --max-local-files-per-record 0 --max-loads 0 --load-local
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
