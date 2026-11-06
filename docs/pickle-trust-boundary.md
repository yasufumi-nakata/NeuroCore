# Pickle trust boundary

## Security change

`.pkl` and `.pickle` remain recognized formats, but generic loading, directory loading, CLI paths and direct `load_pickle()` calls reject them by default **before deserialization**. This closes the default route by which a malicious pickle could consume memory or CPU even when its globals pass a restricted-unpickler allowlist.

The restricted unpickler is defense in depth, not a sandbox. An opted-in pickle can still allocate excessive resources. Do not opt in for uploads, downloaded third-party datasets, or data whose producer and integrity have not been verified. Never derive the trust flag from untrusted request data, file extensions, or merely having an HTTPS download URL.

## Migrating verified local fixtures

```python
from neurocore.loaders import load, load_pickle, load_directory

# Only after establishing that the producer and the exact file are trusted:
frame = load_pickle("verified-local.pkl", trusted=True, sampling_rate=250)
frame = load("verified-local.pkl", trusted_pickle=True, sampling_rate=250)
frame = load_directory("verified-local-directory", trusted_pickle=True, sampling_rate=250)
```

The flags require the boolean `True`; truthy strings or integer flags are not accepted. There is no process-wide or environment-variable opt-in. CLI and automatic dataset paths do not silently grant trust.

Prefer numeric NPY/NPZ files loaded with `allow_pickle=False`, or CSV, for exchanged EEG data. Converting an unknown pickle is itself unsafe: do not deserialize it merely to convert it. Use a genuinely isolated, resource-limited conversion environment or obtain a safe-format export from the producer.

Existing rejection of unsafe globals, object arrays and excessive nesting remains active even after explicit trust. Regression tests exercise both the default refusal and those defense-in-depth checks under deliberate test-only opt-in.

Reference: https://docs.python.org/3/library/pickle.html#module-pickle
