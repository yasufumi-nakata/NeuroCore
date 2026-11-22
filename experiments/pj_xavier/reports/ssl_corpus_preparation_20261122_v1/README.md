# SSL input preparation — frozen 2026-11-22 run

**Preparation and independent validation completed. Actual SSL training input is 13.9833 recording-hours; a separate 4.35 hours is reserved for reconstruction validation.** This is input preparation, not evidence of successful EEG intent decoding or agent control.

## Amount actually usable under the fixed recipe

| Scope | Candidate original records | Original recording hours | Candidate 30 s windows | Accepted windows | Accepted interval-union hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 63 | 43.5183 | 5,168 | 1,678 | 13.9833 |
| SSL validation | 12 | 7.3706 | 872 | 522 | 4.3500 |
| Total | 75 | 50.8890 | 6,040 | 2,200 | 18.3333 |

Accepted data comprise **166 sealed float32 arrays, 1,685,013,248 bytes**. The original EEG sample payload is 5,454,564,672 bytes; whole native waveform members, including non-EEG channels/format overhead, total 5,597,576,250 bytes. Archive bytes, original sample bytes, derived arrays, and repeated epochs are different quantities and must not be added as independent data.

Each window has 30 nonoverlapping seconds at 200 Hz and its actual eight or 32 EEG channels. The accepted corpus has 66,000 one-second temporal positions, or 2,106,240 channel-by-one-second patches. These are not language tokens or additional recording hours. Actual model-training exposures depend on the later training run.

## Material source imbalance after QC

| Dataset/device | Original records | Candidate windows | Accepted windows | Accepted hours | Records with accepted windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Portable BrainAmp (BA) | 19 | 1,362 | 746 | 6.2167 | 19 |
| Portable EPOC (EM) | 19 | 1,438 | 509 | 4.2417 | 17 |
| Portable LiveAmp (LA) | 19 | 1,338 | 937 | 7.8083 | 19 |
| UniBa original training sources | 18 | 1,902 | **8** | **0.0667** | **3** |

The corpus is therefore primarily portable EEG adaptation: 2,192 portable windows and only eight UniBa windows (240 seconds). Fifteen of the 18 permitted UniBa training recordings contribute no window under the frozen QC. This is not a claim that those participants were held out; their original training sources were eligible but every fixed-grid window failed QC. Two portable EPOC recordings also contribute no window.

Portable grouping was fixed before QC: 15 publisher participant groups train and four SSL validation, with all three device sessions assigned together. All 15 portable training groups and all four portable validation groups still contribute data. The accepted training corpus has 18 dataset-qualified publisher groups (15 portable + three UniBa), compared with 33 permitted training groups before QC. Accepted validation remains four portable groups. IDs from different datasets do not prove distinct humans; file durations do not prove unique wall-clock experience across devices.

All 14 original UniBa validation/test waveforms, two clock-failed UniBa originals, and all 26 ITU originals are explicitly excluded by raw SHA-256. The sealed 75-source allowlist rejects every other source. Action-label columns, behavioral event tables, and downstream test outcomes were not used to make the SSL corpus or adjust its criteria.

## Exact QC and what was measured

The frozen test is **maximum absolute sample amplitude <=100 µV**, not a peak-to-peak limit. A window with -100 µV and +100 µV can satisfy the amplitude rule while having 200 µV peak-to-peak. Every channel must also have population standard deviation >=0.01 µV and all values must be finite. QC runs after the fixed full-record portable 3–40 Hz filter, exact 30 s extraction, and resampling to 200 Hz. UniBa receives no additional filter. The first and last five seconds are excluded and the nonoverlapping grid begins at second five. No criterion was relaxed after seeing yield.

There are 3,840 rejected windows, all exceeding the maximum-absolute threshold; 95 also fail the minimum-standard-deviation condition. Reason counts overlap and must not be added as unique rejected windows. `per_source_QC.csv` reports all 75 originals using source hashes and group pseudonyms, with counts and available amplitude distributions. `QC_distributions.json` provides device/dataset and train/validation distributions. Pseudonyms support grouping; they are not a guarantee of anonymization.

Rejected window maximum-absolute nominal amplitude has these median / p95 values:

| Source | Median µV | p95 µV | Largest window peak µV |
| --- | ---: | ---: | ---: |
| BA | 127.34 | 304.47 | 2,684.51 |
| EM | 130.10 | 177.67 | 659.24 |
| LA | 144.41 | 2,821.90 | 7,227.67 |
| UniBa | **3,234.47** | **32,795.80** | **594,527.68** |

These are retained post-processing window statistics, not independently calibrated amplifier measurements. UniBa BIDS sidecars say V while native BrainVision headers say resolution 0.1 µV. Header-based MNE decoding and independent binary/header conversion agree, including exact reproduction of all 400 earlier supervised training windows. That establishes software conversion consistency, **not the true physical validity of the publisher's scale**. Extreme nominal amplitudes remain a material provenance/quality limitation. They are not silently divided by an assumed factor, clipped into the accepted range, or retrospectively used to alter QC. They cannot by themselves establish why any model performed as it did.

Accepted maximum-absolute amplitude, peak-to-peak amplitude, and minimum-channel-std distributions were independently recomputed from the sealed arrays without a model forward or raw redecoding. Rejected maximum-absolute and minimum-std statistics come from the original processing pass. Rejected peak-to-peak was not retained and is explicitly unavailable; it was not fabricated or inferred from maximum absolute amplitude.

## Verification and reproducibility

The preparation source and protocol were frozen before processing. All 75 native voltage conversions passed independent checks at the beginning, middle, and end of each signal. A separate accounting audit verifies complete and exclusive accepted/rejected coverage of all 6,040 grid windows, disjoint accepted intervals, membership, channel order, and one-to-one array slots. Thirteen focused tests pass, including four native resampling rates, interpolation overshoot, unit double-scaling, heldout leakage, path traversal, disk/deadline refusal, and missing/overlapping interval accounting.

A separate trainer-side check freshly hashed all 166 arrays and inspected all **421,248,000 float32 values** across all 2,200 windows. It confirmed finite data, units, shape, complete integer slots, no raw-time overlap, group splits and all exclusions. The largest accepted absolute sample was 99.95543 µV and the smallest channel std was 0.390652 µV. Train-only real 8/32-channel CPU backward probes passed; probe weights were discarded. These checks do not establish learning quality or online causality.

Preparation took 196.48 seconds on the workstation with two native CPU threads; observed process peak RSS was 2,645,557,248 bytes. Initial free disk space was 75,928,498,176 bytes with a 35 GiB reserve, and no full archive copy was created. The CPU preparation was bounded by the workstation/controller deadline 1795324658.879168; Colab training has the separately earlier deadline 1795321234.65. This report does not extend either budget.

The immutable protocol, inventory snapshot, preparation summary, independent accounting and input-validation receipts accompany this report. The source allowlist, per-window records, arrays and per-source certificates remain private. Code and exact commands are in `../../model/SSL_PREPROCESSING.md`. Source releases: [Portable EEG, Zenodo 15698441](https://zenodo.org/records/15698441) and [AI-assisted coding, Zenodo 20801101](https://zenodo.org/records/20801101).
