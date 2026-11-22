# Unlabeled EEG corpus for CBraMod adaptation

`prepare_ssl_corpus.py` freezes the source membership and preprocessing before
it observes accepted window counts. It then prepares real, unlabeled waveform
windows on the workstation with at most two native CPU threads. It does not
start model training or read supervised action-label columns.

## Source boundary

| Source | Admitted recordings | Publisher participant groups | Original waveform seconds |
| --- | ---: | ---: | ---: |
| Portable EEG, Zenodo 15698441, original BA/EM/LA signals | 57 | 19 | 125,629.42 |
| AI-assisted coding, Zenodo 20801101, existing UniBa training partition | 18 | 18 | 57,570.8359375 |
| Total before preprocessing/QC | 75 | 37 groups | 183,200.2559375 |

The total is **50.88896 recording-hours**. These are sums over signal files,
not verified unique humans or unique wall-clock experiences. Different devices
are not additional independent participants. No processed duplicate is added
to this count. Candidate source hours are not the number of hours actually
accepted for training.

All three portable recordings for a publisher participant remain in the same
partition. Sort the 19 canonical participant IDs by
`sha256(UTF8("20261122|" + participant_id))`, with ID as tie-breaker; assign the
first 15 groups to SSL training and the final four to SSL validation. This
validation partition selects masked-reconstruction training state and is not
a final test. All 18 admitted UniBa sources belong to the earlier frozen
supervised training partition.

The allowlist explicitly denies 42 raw waveform hashes: 14 UniBa validation/test
sources, two clock-failed UniBa originals, and all 26 ITU originals, including
the clock-failed ITU original. Every non-allowlisted source is forbidden.
Cross-dataset human identity overlap cannot be excluded from the available
publisher identifiers.

## Frozen transformation

1. Decode physical volts using the native BrainVision/EDF calibration and MNE.
   Retain the original header order of the 32 portable or eight UniBa EEG
   channels. Do not fabricate the six UniBa channels absent from the portable
   montages. Non-EEG channels never enter the arrays.
2. The UniBa BIDS `channels.tsv` reports V while its BrainVision header specifies
   resolution 0.1 and unit µV. Preserve this discrepancy. Native decoding and an
   independent binary/header conversion exactly reproduced all 400 earlier
   UniBa training windows, covering 1,433,600 channel-sample values. This validates
   the software conversion path, not the amplifier's physical calibration.
   Do not reinterpret the stored numbers as volts or scale them twice.
3. Portable recordings receive a fourth-order Butterworth 3–40 Hz SOS filter
   applied forward/backward to the full recording with odd padding. UniBa
   receives no additional filter; its distributed 3–40 Hz data are retained.
4. Remove the first and final five seconds. Use a fixed nonoverlapping
   30-second grid starting at second five. Extract a complete native-rate
   window, then use rational polyphase resampling to 200 Hz with Kaiser beta
   5.0 and constant-zero padding.
5. Test the **resampled physical-voltage window**: all values finite, every
   channel's population standard deviation at least 0.01 µV, and maximum
   absolute value at most 100 µV. Reject the entire window on a failed condition.
   Rejections and reason counts are retained; criteria are not adjusted to
   increase yield.
6. Store volts multiplied by 10,000 as float32, shaped
   `[windows, channels, 30, 200]`. Each file contains at most 16 windows, so a
   32-channel shard is approximately 12.3 MB. Every file is sealed by SHA-256,
   bytes, shape, and dtype.

MNE may parse format annotations while opening a signal file. No annotation,
behavioral event table, or target action label is selected or passed into SSL.
Forward/backward offline filtering does not establish an online causal signal
pipeline.

## Resource and provenance controls

Registration seals the implementation hash, protocol, 75-source allowlist,
42-source denylist, inventory evidence, package versions, and capacity plan.
Before QC, the fixed grid permits 6,040 windows: 5,168 training and 872 SSL
validation. Maximum derived float32 payload is 3,543,168,000 bytes. Including
the largest temporary original member and a 1 GiB allowance, the registration
requires 4,800,889,344 bytes beyond the 35 GiB free-space reserve.

Only one portable waveform and its format companions are extracted at a time;
no additional full archive copy is made. Original files are never rewritten.
Each source checks its waveform hash, native calibration, and decoded duration.
The process enforces the workstation/controller absolute deadline
(`1795324658.879168`, 05:17:38 UTC / 14:17:38 JST) and free-space reserve, and
declines a source whose conservative native-array working estimate exceeds
4 GiB. The supervisor additionally bounds preparation to 1,800 seconds. Colab
training is separately bounded by its earlier original session deadline
(`1795321234.65`, 04:20:34 UTC / 13:20:34 JST); completing workstation preparation
does not extend the Colab budget or deadline.

Completed per-source certificates and shards support restart after interruption.
An incomplete, unsealed shard is never silently reused or overwritten: it
requires explicit recovery while preserving the failed attempt. A completed
corpus cannot be rerun in the same output directory.

## Commands

From the correct workstation checkout, with its existing `.venv-xavier`:

```bash
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/model -p test_prepare_ssl_corpus.py -v

.venv-xavier/bin/python experiments/pj_xavier/model/prepare_ssl_corpus.py register \
  --repo "$PWD" \
  --output private/xavier/ssl_preparation_registration_v1

.venv-xavier/bin/python experiments/pj_xavier/model/prepare_ssl_corpus.py prepare \
  --registration private/xavier/ssl_preparation_registration_v1 \
  --expected-protocol-sha256 642246adb9bf39a9652a2ccb9fdbead834c611fbd4f5325b8dfc08d54fba45a5 \
  --output private/xavier/prepared/ssl_portable_uniba_v1
```

The example registration and corpus directories are the original sealed run;
they are not overwrite targets. A new registration will have a different time
and hash and must not be used to extend the original deadline or silently
change model-selection policy. `--resume` is available only for the original
incomplete preparation, with its original protocol hash and within its limits.

After completion, independently account for every fixed-grid window:

```bash
.venv-xavier/bin/python experiments/pj_xavier/model/audit_ssl_preparation.py \
  --corpus private/xavier/prepared/ssl_portable_uniba_v1 \
  --registration private/xavier/ssl_preparation_registration_v1 \
  --output private/xavier/ssl_preparation_stage_v1/accounting_audit.json
```

The corpus interface is `manifest.jsonl`, `artifacts.json`,
`source_allowlist.json`, `protocol.json`, `corpus_summary.json`, and
`completion.json`. Per-source certificates account for every fixed-grid window
as accepted or rejected. Source seconds, accepted unique intervals, one-second
temporal patches, derived bytes, and repeated model epochs remain separate
quantities. Completion of this preparation establishes an input corpus, not
successful learning, agent control, or latent-intent decoding.
