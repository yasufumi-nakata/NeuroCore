# PJ-Xavier S Mac EEG source catalogue bridge (metadata only)

**Scope:** This code detects source-catalogue overlaps before any proposed download is added to the active PJ-Xavier acquisition system. It does not access raw EEG, download data, modify SQLite, change the 37 download workers, select models, or train any model. It uses Python's standard library only.

## Current measured status (2026-11-23, Asia/Tokyo)

AoyamaLab's high-confidence CSV has **5,165 source records**. The broader discovery CSV has **22,327** records. The S Mac canonical catalogue had **5,165** rows. On smac31 the source identity comparison found **5,162** strong key matches and **3** exact-title+same-normalized-URL aliases, totaling **5,165/5,165** accounted high-confidence records. The discovery collection had **8,535** matched and **13,792** unmatched metadata records; none of the unmatched records was simultaneously classified `immediate` and `eeg_text_match=True`. The unmatched count is **not** a count of verified downloadable EEG datasets or additive storage. Names alone, journal articles, account-required downloads, and absent payloads must never be added automatically.

A separate **official public OpenNeuro S3 ListObjectsV2 + dataset_description.json metadata audit** recovered an unfilled manifest for **existing** catalog source `ds005841` (ERP in VR; CC0; DOI `10.18112/openneuro.ds005841.v1.0.0`): **872 objects**, including **288 BDF files**, nominal total **7,889,279,183 bytes**. The metadata file (kept privately, not committed) has SHA-256 `ae581279f3320129a57274943d33f540c0ade1ed94b590bd13d7262046326c17`. No EEG objects were downloaded by the bridge. Source stays held until source-version, checksum, companion files, EEG channels/units, license and split/must-link review. S3 ETags should not be assumed to be checksums. This dataset does not itself supply ground truth for EEG+PC-screen+operation-history free intention.

## Code

`catalog_identity.py` normalizes DOI/URL/provider-family identities and reads both metadata CSVs. `catalog_diff_standalone.py` compares them to a frozen or read-only root JSONL source catalog and writes **only a report in the explicitly isolated output namespace**. `s3_manifest_probe.py` retrieves public OpenNeuro source metadata for `ds005841` and records an object-size receipt on the operator's machine, **not source payloads**. It requires an HTTPS connection only for this metadata probe.

An instance is installed and independently tested on smac31 at `/Users/yasufumi/pj-xavier/catalog-bridge-v1/`. The deployed instance has matching behavior but includes a workstation integration support script not present here; only the portable, non-sensitive code is published. Its successful independent isolated report is `reports/catalog_diff_20261122T222228Z.json` in that private smac31 directory.

```sh
# Execute on smac31 after the two CSVs are installed in its private bridge folder:
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 \
  catalog_diff_standalone.py --check-only

# When authorized, write *only* a new private bridge report:
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 catalog_diff_standalone.py
```

`--input-dir`, `--root-catalog` and `--out-dir` allow other layouts. Output must stay inside `~/pj-xavier/catalog-bridge-v1`; the script refuses a different output root and checks load and free storage before processing. Source metadata requires **official file manifests, explicit licenses, stable versions/checksums, EEG scope, and owner-reviewed admission** before downloads. The shared acquisition plan and database are outside the scope of this utility.

## Reproduction and tests

```sh
python3 -m unittest discover -s experiments/pj_xavier/catalog_bridge/tests -v
```

Tests synthesize small CSV and root-JSONL inputs with no EEG data, including duplicate metadata aliases and negative source controls. The real source lists and S3 object metadata contain provider-specific records and remain in private storage. Public summary values are measured snapshots, not a systematic guarantee of future source completeness.
