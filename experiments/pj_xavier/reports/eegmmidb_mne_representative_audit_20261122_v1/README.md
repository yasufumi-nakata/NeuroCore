# EEGMMIDB: independent representative MNE parser check

MNE 1.13.2 independently checked 9 deterministically selected EDF records from the pinned 1,526-record native audit. Headers agreed in 9/9, and standalone annotation parsing agreed in 9/9. Raw attachment was exact in 8/9; 1 record(s) showed a separately documented recording-end duration clip. Unexplained reader differences: 0.

| EDF record | Hz | Samples | Header | Standalone TAL | Raw attachment |
| --- | ---: | ---: | --- | --- | --- |
| S001R01 | 160 | 9760 | True | True | exact |
| S001R04 | 160 | 20000 | True | True | exact |
| S088R03 | 128 | 15872 | True | True | exact |
| S089R01 | 160 | 9760 | True | True | exact |
| S089R02 | 160 | 9760 | True | True | exact |
| S089R03 | 160 | 28960 | True | True | exact |
| S092R03 | 128 | 15872 | True | True | exact |
| S100R03 | 128 | 15744 | True | True | end-duration clip |
| S109R14 | 160 | 19680 | True | True | exact |

Selection was registered from the prior native ledger before any MNE EDF read. It includes clean 160 Hz baseline and motor-imagery controls, the first 128 Hz record in each affected group, baseline non-T0 cases, the first 160 Hz overlap case, out-of-range representatives, and the final clean record. Overlapping categories are deduplicated. All original ZIP and ledger hashes were verified before and after.

Each selected member was verified against both the recording and officially checked member ledgers, temporarily extracted, read with preload=False, closed and removed from its owned temporary directory. No original was changed. Standalone read_annotations and Raw.annotations are recorded separately, including every warning and interval difference. The absolute float-comparison tolerance is 1 nanosecond.

The actual observed sample rates and annotation durations remain unchanged. Small overlaps and baseline T1 labels are confirmed as source annotations when both parsers agree; their cause and semantic correctness remain unresolved. Any Raw boundary clipping is reader attachment behavior, not a correction performed by this audit.

See [protocol.json](protocol.json) for the deterministic selection and pinned reader source identities, and [summary.json](summary.json) for per-record differences and warnings. Full annotation lists remain in the private detailed ledger. This is a representative parser check, not whole-dataset waveform validation or authorization to include flagged labels in training. Existing quantity, SSL, and model evaluations were not changed; model forward calls and label corrections were zero.

## Observed boundary and baseline cases

S100R03 contains a final T1 annotation beginning at 117.9 seconds with duration 5.125 seconds, ending at 123.025 seconds. Its stored signal is 123 seconds long. The independent standalone MNE reader retained that interval exactly. Attaching it to Raw shortened its duration to 5.1 seconds and retained the code and onset. MNE emitted a RuntimeWarning: “Limited 1 annotation(s) that were expanding outside the data range.” The 25 ms difference is therefore visible reader attachment behavior; neither the native ledger nor the original EDF was repaired. No annotation was dropped in these nine representatives.

The 128 Hz representatives S088R03, S092R03 and S100R03 retain their overlap annotations in both readers. S089R03 also retains its 160 Hz overlap annotations. The baseline T1 codes in S089R01 and S089R02 are likewise preserved. Agreement confirms that these recorded values were not introduced by the separate native parser; it does not resolve timing causes or task semantics.

The nine representatives contain 209 standalone events in total. This remains a targeted independent check against a specific parser-interpretation risk, not a claim that MNE revalidated all 1,526 records. The reproducible [audit module](../../model/audit_eegmmidb_mne_representatives.py) refuses nonempty output directories and retains the fixed source hashes and selection rules.
