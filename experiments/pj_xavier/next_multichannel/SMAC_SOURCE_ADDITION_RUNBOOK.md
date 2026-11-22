# S Mac source addition: Zenodo 2392979

This is a new offline SSL preparation stage. It retains the verified Zenodo
806023 train and validation corpus exactly, then adds only eligible 2392979
train windows. It never changes the original corpus or its frozen validation
membership. Validation remains an estimate for the 806023 domain only; it does
not evaluate generalization to 2392979. No behavior labels are used.

## Retained source evidence

The 2392979 audit contains 36 converted FIF records, 29 unique physical-V EEG
payloads, 8 EEG channels, 256 Hz, 12 nominal subject IDs and 8 must-link
components. The frozen split has 26 train records and 10 test records, with no
validation records. Removing 5 duplicate train aliases leaves 21 possible
train records before current file verification and fixed QC.

The adapter reads no test or duplicate-alias signal files. Exact payload
duplicate proofs and all must-link components must match the frozen audit.
Existing denied sources, payloads, and groups propagate through their
must-link component. Original FIF bytes, calibrated physical-V EEG payload,
sample count, ordered montage, and calibration metadata are verified again on
the S Mac. Previous metadata-only `fit_allowed: false` records are not edited;
current evidence produces a separate admission certificate.

The retained native extrema place only 5 of the 21 candidates within the
fixed ±0.1 V source bound. Their postprocessing QC is still unknown. These
counts are acquisition/preparation candidates, not new training exposure.
No correction factor is inferred from amplitude and no clipping is applied.

| Required input | Repository-relative path |
| --- | --- |
| 2392979 source audit | `private/xavier/expansion_candidates_v1/small_BCI_batch_header_audit_v1/records.jsonl` |
| Frozen split metadata | `private/xavier/phase4_automation_1615/source_admission.private.jsonl` |
| Exact duplicate and must-link proof | `private/xavier/expansion_candidates_v1/SSVEP2392979_duplicate_constraints_v1/future_split_constraints.private.json` |
| Previous source deny | `private/xavier/ssl_preparation_registration_v1/source_allowlist.json` |
| Previous recording deny | `private/xavier/eegmmidb_phase2_v1/metadata/recording_metadata.jsonl` |
| Previous exclusions | `private/xavier/ssl_source_inventory_v1/exclusions.json` |
| Base corpus | The complete successfully prepared `smac806023_ssl_v1` directory, including arrays, manifests, admission, protocol and completion |

The source audit SHA256 is
`ff12285c7e339027d3c86e66f376b3cb43cc2b90b41d280cc82583669bcaad14`;
the split metadata SHA256 is
`555b3247b6a9eadcf48bed22f29ef381b06a601e87909241aedcc60900c70e1d`;
the must-link input SHA256 is
`549f87071ad1a415b80181f5ea0229f43d1797c180a65674fc077729b3b184b0`.
These values describe the retained inputs examined on 2026-11-22. A change
requires review of its scientific meaning before using a new input set.

## Execution on the destination S Mac

Place the verified raw release directly under the destination node's `raw`
directory. Transfer the complete base corpus directly between S Macs into the
same destination node tree; do not stage raw or prepared signals on the local
workstation. Check the original base `completion.json` SHA before transfer
and pass that value to the preparation command after transfer.

Both `prepare_smac_fif.py` and `prepare_smac_ssvep.py` must be present in the
same `next_multichannel` directory, alongside the existing acquisition storage
guard. The existing S Mac venv needs MNE, NumPy and SciPy.

The example assumes smac06 and renamed metadata copies in `$SMAC_NODE/meta`.
Use a fresh immutable output directory for each attempt.

```sh
SMAC_NODE="$HOME/pj-xavier/smac06"
SMAC_PY="$SMAC_NODE/venv/bin/python"
SMAC_CODE="$SMAC_NODE/code/experiments/pj_xavier"
SMAC_META="$SMAC_NODE/meta"
SMAC_BASE="$SMAC_NODE/prepared/smac806023_ssl_v1"
SMAC_OUTPUT="$SMAC_NODE/prepared/smac806023_plus2392979_ssl_v1"
SMAC_BASE_SHA='the verified original base completion SHA256'

"$SMAC_PY" "$SMAC_CODE/next_multichannel/prepare_smac_ssvep.py" \
  --raw-root "$SMAC_NODE/raw/zenodo-2392979-2392979" \
  --source-ledger "$SMAC_META/small_BCI_batch_header_audit_records.jsonl" \
  --metadata-admission "$SMAC_META/source_admission.private.jsonl" \
  --must-link "$SMAC_META/ssvep2392979_future_split_constraints.private.json" \
  --base-corpus "$SMAC_BASE" \
  --expected-base-completion-sha256 "$SMAC_BASE_SHA" \
  --deny-json "$SMAC_META/old_ssl_source_allowlist.json" \
  --deny-json "$SMAC_META/old_phase2_recording_metadata.jsonl" \
  --deny-json "$SMAC_META/old_ssl_exclusions.json" \
  --output "$SMAC_OUTPUT" \
  --reserve-gib 30
```

Do not pass the current base admission as an extra `--deny-json`: the base is
already separately verified and its validation set is intentionally retained.

Exit 0 and `completion.json` with status
`prepared_verified_new_offline_SSL_corpus` permit a new trainer registration.
The result reports actual admitted records, nonoverlapping windows and unique
hours. Preprocessing itself reports zero gradient exposure. The trainer must
record actual examples/unique windows presented separately.

Exit 2 with `preparation_result.json` and status
`blocked_no_eligible_new_source_windows` means no new train input passed the
fixed protocol. It intentionally has no `completion.json`, training manifest
or copied base arrays. Keep `protocol.json`, `source_certificates.json` and
`preparation_result.json` as evidence and move to the next verified source.
Do not relax the amplitude threshold or invent a unit multiplier to obtain a
successful stage.

## Processing and training contract

The new source uses causal SOS 3–40 Hz filtering, causal 80 Hz anti-aliasing,
one-native-sample-delayed 256→200 Hz interpolation, a 2 s warmup and
nonoverlapping 3 s windows. Maximum postprocessing absolute amplitude is
100 µV and minimum channel standard deviation is 0.01 µV. Output is float32
`[N, 8, 3, 200]` in `V_x_1e4`. The source's prior conversion and reference
provenance are not fully known; this remains offline SSL, never online-harness
certification.

The existing `model/cbramod_ssl_smac.py` accepts 8-channel and 16-channel
buckets. Register a new protocol for the combined manifest and run fresh MPS
capacity probes covering every registered shape before training. Do not reuse
the initial corpus's capacity report or resume state against changed corpus
membership. Preserve parent-run lineage separately. The node supervisor must
only launch training after a successful preparation result.

## Verified implementation checks

The synthetic tests exercise the split and must-link checks, exact base-array
hash verification, causality, fixed amplitude rejection, duplicate alias
omission, test-file non-access, correct merging of base validation, and the
all-rejected result. Real data and MPS execution remain destination-node work.

```sh
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/next_multichannel \
  -p test_prepare_smac_ssvep.py -v
```

## Deployed continuation on 2026-11-22

The current deployment uses `smac06/training_code_v2` rather than the generic
`code` path in the example above. Its source metadata retains the repository
paths under `training_code_v2/private/xavier`. The sealed addition includes the
adapter and its fixed metadata. The original `smac01/training_code_v1` and
initial corpus remain unchanged.

Six transfers were verified file by file before waiting for new raw data:
the initial code snapshot, initial corpus, complete parent run directory,
the pinned upstream repository including `.git`, the pretrained weights and
their metadata. Signals and checkpoints moved within the S Mac cluster via a
memory relay on smac31, with no workstation signal staging.

| Item | Exact deployed location |
| --- | --- |
| Preparation code | `smac06:/Users/yasufumi/pj-xavier/smac06/training_code_v2/experiments/pj_xavier/next_multichannel/prepare_smac_ssvep.py` |
| Base corpus | `smac06:/Users/yasufumi/pj-xavier/smac06/prepared/smac806023_ssl_v1` |
| Parent run | `smac06:/Users/yasufumi/pj-xavier/smac06/parent_runs/smac806023_ssl_20261122_v1/run` |
| New corpus output | `smac06:/Users/yasufumi/pj-xavier/smac06/prepared/smac806023_plus2392979_ssl_v1` |
| Preparation receipt | `smac06:/Users/yasufumi/pj-xavier/smac06/stage2_preparation_receipt.json` |
| Preparation log | `smac06:/Users/yasufumi/pj-xavier/smac06/stage2_prepare.log` |
| Controller state | `smac31:/Users/yasufumi/pj-xavier/control_20261122/smac2392979_stage2/state.json` |
| Controller receipt | `smac31:/Users/yasufumi/pj-xavier/control_20261122/smac2392979_stage2/preparation_receipt.json` |

The verified original base completion SHA256 is
`fb7c5834520478f66cf45a9016e99153b078be33c88e346988b147d8dcddf0c4`.
It contains 659 train and 327 validation windows. The controller reads the
smac08 acquisition DB without changing statuses or leases and waits for all
36 `zenodo-2392979-2392979` files to be complete before a verified cluster copy
to smac06. It then runs preparation once and never starts training.

Controller terminal phases are `prepared_ready_for_root_training`,
`blocked_no_eligible_new_source_windows`, `failed`, and
`waiting_timeout_no_training_started` (12-hour acquisition wait bound).
Only the first phase, a successful process exit and the verified completion
artifact permit the separate training controller to continue.
