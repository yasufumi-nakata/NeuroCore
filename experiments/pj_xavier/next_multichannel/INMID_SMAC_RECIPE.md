# InMID offline SSL source addition on smac35

## Frozen scope

The retained Mendeley rjx76wd5v6 v2 audit has 92 original EDF recordings,
14 EEG channels at 128 Hz and 44 auxiliary channels per file. The fixed
metadata has 76 train recordings and 16 test recordings, with no validation
fold. The adapter verifies all 92 metadata records but opens only eligible
train EDF signals. The 145 non-EDF distribution files, including MAT and
video, may continue downloading in parallel and are not required by prep.
Their duration and auxiliary channels add zero independent EEG hours.

Native EDF header calibration from microvolts to V, original file SHA,
original integer EEG hash, calibration hash, full physical-V payload hash and
independent MNE probes must all agree with the retained audit. The original
integer EEG hash and physical-V hash are different identities; both are
checked against prior exclusion lists. Existing metadata rows are not edited.
A separate new source-admission certificate is emitted only for verified
records with accepted windows. EEG duplicates require explicit review.

The sealed base can contain any previously verified train source, including
CNT. Its completion, manifest, protocol, artifacts and admission are checked,
then the real trainer validates every array and provenance slot. Its
validation must remain the existing 806023 domain. Base rows, identities and
array bytes are preserved exactly, with only the array path prefixed by
`base/` in the new package. This does not evaluate InMID generalization.

## Predeclared 128 Hz processing

The native Nyquist frequency is 64 Hz, so the earlier 80 Hz native low-pass
cannot be used. This stage upsamples from 128 to 200 Hz and has no decimation
anti-alias stage. The fixed filter is `butter(4, (3, 40), btype='bandpass',
fs=128, output='sos')`: four biquads, final order eight, followed by forward
`sosfilt` with zero state separately for each original recording. The SOS
coefficients and SciPy version are sealed in the new protocol before current
signal reads. This is not zero-phase filtering.

At output availability time `t = j/200`, define `n = floor(128*j/200)` and
`a = (128*j mod 200)/200`, then use
`z[j] = (1-a)*y[n-1] + a*y[n]`. The represented signal time is `t-1/128`;
neither term uses a native sample later than `t`. A future-sample mutation
test checks this property. This interpolation changes neither the original
bandwidth nor independent recording time. Its amplitude attenuation and
the filter's phase response remain source-specific.

The first 2 seconds are warmup; accepted windows are nonoverlapping 3-second
intervals. Native absolute amplitude must be at most 0.1 V. Fixed window QC
keeps the earlier maximum 100 µV peak and minimum 0.01 µV channel standard
deviation. There is no clipping, data-dependent unit multiplier, rereferencing
or normalization fitted to validation. Output is float32 `[N,14,3,200]` in
`V_x_1e4`, with `native_sample_rate_hz=128` retained in every manifest row.
Prior publisher preprocessing is not proven causal, so online-harness
eligibility remains false.

Design references:

- [SciPy Butterworth filter definition and SOS order](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [SciPy forward SOS filter and initial state](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
- The retained original EDF audit, `InMID_header_audit_v1/protocol.json` and `records.jsonl`, supplies the native rate, channel roles and calibration evidence.

## Acquisition and paths

Readiness can be determined from smac35's node-local `downloads.sqlite` using
`catalog_id='001085'` and the 92 `.edf` relative paths. All 92 must be complete
with the expected recorded checksums. Do not wait for the other 145 files.
Keep the original DB identities, status and leases; do not copy a historical
workstation download DB into this node.

The 237-file distribution has 1,410,402,579 bytes. The current dataset key is:

```text
landing-https___data.mendeley.com_datasets_rjx76wd5v6-observed-unpinned-4e394f5ce88c
```

The version label in this acquisition key is not evidence of a new version;
the adapter pins the already audited v2 bytes. The raw root is:

```text
/Users/yasufumi/pj-xavier/smac35/raw/landing-https___data.mendeley.com_datasets_rjx76wd5v6-observed-unpinned-4e394f5ce88c
```

Each `source_path` in the audit equals its DB `relative_path`, for example
`source_67710cd9f99e/2023-07-24-P01-1-1.edf`. Do not strip the source directory
or select files by loose basename matching.

## Exact preparation CLI

The root controller selects and transfers the base corpus and parent run from
the preceding successful stage. It must pass the original source completion
SHA after verifying the destination copy. Preparation itself does not load
the parent checkpoint, run capacity probes or start training.

```sh
INMID_NODE=/Users/yasufumi/pj-xavier/smac35
INMID_CODE="$INMID_NODE/training_code_inmid_v1"
INMID_META="$INMID_CODE/private/xavier"

"$INMID_NODE/venv/bin/python" \
  "$INMID_CODE/experiments/pj_xavier/next_multichannel/prepare_smac_inmid.py" \
  --raw-root "$INMID_NODE/raw/landing-https___data.mendeley.com_datasets_rjx76wd5v6-observed-unpinned-4e394f5ce88c" \
  --source-ledger "$INMID_META/expansion_candidates_v1/InMID_header_audit_v1/records.jsonl" \
  --metadata-admission "$INMID_META/phase4_automation_1615/source_admission.private.jsonl" \
  --audit-protocol "$INMID_META/expansion_candidates_v1/InMID_header_audit_v1/protocol.json" \
  --base-corpus '<VERIFIED_BASE_CORPUS_ON_SMAC35>' \
  --expected-base-completion-sha256 '<ORIGINAL_BASE_COMPLETION_SHA256>' \
  --deny-json "$INMID_META/ssl_preparation_registration_v1/source_allowlist.json" \
  --deny-json "$INMID_META/eegmmidb_phase2_v1/metadata/recording_metadata.jsonl" \
  --deny-json "$INMID_META/ssl_source_inventory_v1/exclusions.json" \
  --output '<NEW_IMMUTABLE_CORPUS_DIRECTORY_ON_SMAC35>' \
  --reserve-gib 30
```

Exit 0 with `completion.json` status
`prepared_verified_new_offline_SSL_corpus` means the combined package passed
the trainer's actual loader. The completion reports accepted new records,
windows and unique hours; preparation's gradient exposure is zero.

Exit 2 with `preparation_result.json` status
`blocked_no_eligible_new_source_windows` has no trainable completion or copied
base arrays. Preserve its protocol and per-record certificates, keep the
previous successful parent and move to another verified source. Hash, split,
unit or base-integrity failures also prevent completion. Only the root
controller subsequently registers a new training protocol, probes all shape
buckets and continues from a selected training parent.

## Validation

Eight focused synthetic tests cover the frozen train/test split, both payload
denies, duplicate rejection, real EDF/MNE physical calibration, exclusion of
generic-uV auxiliary channels, wrong-unit rejection, no-future-input behavior,
fixed QC, generic sealed bases, preservation of base validation and the
all-rejected receipt. The end-to-end fixture creates only 76 synthetic train
EDF files; all test EDF and non-EDF distribution files are deliberately
absent. Its accepted output also passes the actual trainer loader.

```sh
"$INMID_NODE/venv/bin/python" -m unittest discover \
  -s "$INMID_CODE/experiments/pj_xavier/next_multichannel" \
  -p test_prepare_smac_inmid.py -v
```
