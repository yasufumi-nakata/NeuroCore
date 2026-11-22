# ITU external evaluation of four frozen history baselines

All four saved UniBa binary logistic models were applied once to the fixed ITU
external cohort, without refitting, vocabulary construction, C selection or
threshold adjustment. The cohort contains **630 observed-action windows** from
25 site-qualified participant observations: 126 recorded AI submissions and 504
other recorded actions. It is an enriched case-control sample, with a fixed
positive fraction of 20%.

| Frozen history input | Recall | Specificity | Balanced accuracy | AUROC | Sample average precision |
|---|---:|---:|---:|---:|---:|
| Before EEG window | 0.6746 | 0.6964 | 0.6855 | 0.7239 | 0.3854 |
| Before EEG window + coarse context | 0.6667 | 0.6806 | 0.6736 | 0.7252 | 0.3844 |
| Before guarded decision boundary | 0.8016 | 0.7480 | 0.7748 | 0.8352 | 0.5423 |
| Before guarded decision boundary + coarse context | 0.7937 | 0.7460 | 0.7698 | 0.8216 | 0.4837 |

The balanced-accuracy 95% participant-observation-cluster percentile intervals
are respectively **[0.6336, 0.7397]**, **[0.6239, 0.7190]**,
**[0.7268, 0.8165]**, and **[0.7252, 0.8076]**. Each used 1,000 valid paired
resamples, with seed 20261122. These are conditional intervals for the four
frozen fits and this sample, not multiplicity-adjusted superiority tests or
uncertainty from repeated training.

All models retain threshold **0.5** and their original UniBa validation-selected
regularization. History consists only of allowlisted past key-down/up, mouse
click and mouse movement counts, recency and source/missing indicators. Coarse
context uses only the original saved vocabulary. Unknown categories map to zero
in its saved columns. No EEG signal is an estimator input in these baselines.
The guarded-decision history includes a later observation period than the
pre-EEG history; compare EEG's incremental value against an appropriately timed
control rather than treating the two history boundaries as identical inputs.

## Reproduction and provenance

The adapter first reproduced all **1,624 original UniBa probability values**
(203 test examples × four binary models × two classes) with **maximum absolute
difference 0.0**. This authenticates the saved standardization, feature order,
binary sigmoid and class ordering. The four fixed models were selected earlier
using only UniBa train/validation partitions; no ITU data was used for model
selection.

All external metadata hashes and all 25 prepared arrays (9,034,880 bytes) were
verified before inference and after scoring. The pinned independent preparation
audit had already rederived all 630 raw windows, 630 recorded-event labels and
1,260 history feature sets. This evaluation rehashes the prepared artifacts and
authenticates that evidence; it does not claim another complete raw derivation.
The run took 6.792 seconds on workstation CPU with native thread pools capped at
two. Package versions are recorded in the report.

The first invocation stopped during input validation because an adapter draft
incorrectly required `feature_availability_verified: true`. The unchanged v3
cohort correctly declares false and assumes 4.125 seconds of device-filter
support. That failed attempt created **no external predictions and no
consumed-test marker**. Its original source/protocol/attempt evidence remains
preserved. The corrected adapter verifies false, 4.125 seconds, the corresponding
availability timestamp, and the decision boundary. The revised protocol was
fixed before the first successful external inference, with no cohort, estimator,
feature, threshold or selection change. Eleven fixtures and the real cohort
preflight passed before inference.

## Practical interpretation and limits

Past interaction history already distinguishes many subsequent recorded AI
submissions in this cohort. A model that uses EEG must demonstrate improvement
over such timing-matched controls before a benefit from EEG is established.
These results alone do not establish that an EEG model adds useful information.

The labels indicate recorded request transmission, not the intended request's
contents or a person's latent goal. No idle periods or live task outcomes are
available here. Natural-population precision, false actions per minute and
autonomous task completion remain unestimated. Prediction probabilities and
sample precision are not calibrated confidence in intention.

No raw file hash overlaps the original UniBa cohort. Site-qualified participant
labels distinguish observations, but short participant names overlap across
sites; the data does not prove either shared or disjoint human identities.
`human_identity_disjointness_verified` and `autonomous_agent_validated` remain
false. Individual predictions, participant/event IDs and EEG arrays are private.

## Immutable outputs

- Adapter source SHA-256: `9161261bdada59f83f1080d17bcf30f303445fbf8002b5cc210b90ccc4c39a31`.
- External protocol v2: `1acd857958d034125c4a69d222b7564c15531f17cdf89a509d6319beb625eb02`.
- External manifest: `cfd93863b8f8e263174e6ab59395496efbd397e821542fa3a1f0c1e5faa503b0`.
- Aggregate report: `bdb15c7ba90888347e173be95e70b9e9ca460139d556576ea96cf13a2b1aeab6`.
- Private 2,520 prediction rows: `378eedeecb519fe50a9f4b6f0ba4a94667a1b82ed37236b0b65ea57b99ce26c8`.
- Completion receipt: `47cb14e30e242abdcc3cae630fdde0dd86f91f9559e1f9b24bb61628f4784386`.

The report, parity results, completion receipt and fixed protocol accompany this
README. Do not rerun a consumed external output directory or tune these models
from this evaluation.
