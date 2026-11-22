# BCI 8089820: complete acquisition and original-signal quantity audit

The entire pinned ZIP was downloaded and verified, with **694 GDF records from 87 source groups**. Exact EEG payload/calibration deduplication leaves **693 EEG records and 72.256857639 recording hours**. This is acquired source EEG, not additional gradient-training exposure or validated autonomous-agent performance.

## Separate quantities

| Scope | Observed quantity |
|---|---:|
| Complete compressed release ZIP | 27,475,466,178 bytes |
| All 1,071 uncompressed release members | 34,164,083,331 bytes |
| All GDF records, including one exact EEG duplicate | 72.381875000 recording hours |
| Deduplicated original EEG | 72.256857639 hours; 28,767,709,440 stored scalar bytes |
| EOG, all 694 records | 3,201,942,528 bytes |
| EMG, all 694 records | 2,134,628,352 bytes |
| Additional training in existing frozen models | 0 hours |

Each GDF has **27 named EEG channels, 3 EOG and 2 EMG channels at 512 Hz**, with 64-bit stored sample scalars. EEG byte accounting selects the actual 27 electrode names, not the first 27 signal columns. Header-based physical conversion is not an independent amplifier calibration or physiological-quality certificate. Hours are elapsed signal time, not channel-hours. The archive has 86 groups with eight records and one with six; missing records were not fabricated.

All 1,071 members were fully streamed with ZIP CRC and SHA-256 verification. Only one GDF at a time was temporarily extracted, at most 66,979,520 bytes. The original ZIP was unchanged and fully SHA-256 reverified after the audit. Independent accounting checks each GDF against its archive member and proves the EEG/EOG/EMG/header/event byte partition exhaustively. No full 34.16 GB extraction was performed.

## Duplicate and future split constraints

One EEG record has identical stored payload, calibration, channel order, rate and sample count under two publisher participant IDs. Its duration and EEG bytes are counted once. The private constraint ledger requires both entire source groups to share one future split if both remain in a study; merely subtracting the repeated record would not prevent cross-fold contamination. This is an observed data-identity relationship, not a claim that the two IDs represent the same person. No cross-release approximate-waveform equivalence is asserted.

## Annotation and quality limits

All 694 raw GDF event-rate fields equal zero, which remains unchanged. Pinned MNE uses the native 512 Hz signal rate to convert these positions; that fallback does not establish the missing publisher timebase. Under that fallback, 87 records have one out-of-bounds event each, while the 20,792 left/right MI cue events are in range. In-range cues are insufficient to certify annotation timing. All event tables are mode 1, with no stored duration fields; no durations or timestamps were repaired or imputed.

The frozen v2 summary's legacy records_with_event_outside_record=0 counts only truthy explicit-rate results. With zero explicit-rate records, it is **not evidence of zero real annotation errors**. Use the unknown-timebase and fallback fields in future_training_constraints.aggregate.json.

The [authors' technical validation](https://www.nature.com/articles/s41597-023-02445-z.pdf) flags seven potentially problematic participants involving electrode artifacts, possible disconnection, or channel-order problems. They omitted these participants from their 80-subject neurophysiology aggregation. The new private ledger attaches source-backed flags to all 56 affected records. These are review flags, not our own QC-pass result or automatic exclusion from every future ML use. They do not reduce the acquired source quantity above.

A new source-specific split, electrode/calibration review and annotation-timebase protocol are required before supervised use. Existing 64-channel EEGMMIDB cohorts, units, QC, model checkpoints and held-out data remain unchanged. The first audit's zero-event-rate failure is retained privately; v2 makes the missing timebase explicit.

## Evidence

- [Pinned release](https://zenodo.org/records/8089820), catalog 000038; ZIP SHA-256 7c611b08995f10c2d313d194e3e75c88f7792403455994d77ca37e16104fcacf.
- [Primary data paper](https://www.nature.com/articles/s41597-023-02445-z), DOI 10.1038/s41597-023-02445-z.
- [GDF 1.25 event-table specification](https://pub.ista.ac.at/~schloegl/matlab/eeg/gdf4/TR_GDF.pdf).
- summary.json, protocol.json and completion.json retain the executed audit's exact bytes.
- independent_accounting.json and future_training_constraints.aggregate.json add exhaustive accounting and restrictions. Detailed source-member, participant and annotation rows remain private.
