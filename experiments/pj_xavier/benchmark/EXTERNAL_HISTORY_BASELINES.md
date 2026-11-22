# Frozen history baselines on the independent ITU cohort

`external_history_baselines.py` applies four **already selected binary** UniBa
history logistic regressions. It never fits an estimator, learns a vocabulary,
chooses regularization, or adjusts a threshold on ITU. Three-class baseline fits
are deliberately outside this registered comparison.

The four modes are `history_pre_eeg`, `history_pre_eeg_plus_context`,
`history_pre_decision_guarded`, and
`history_pre_decision_guarded_plus_context`. The endpoint is recorded
`ai_prompt_submit` versus other observed actions at a fixed probability threshold
of 0.5. This is a proxy for recorded request transmission, with no claim about the
contents of a request or a person's latent intention.

## Immutable inputs and probability parity

The adapter pins the existing `selection.json`, four `.npz` files, original
baseline protocol/report/receipt/predictions, history feature source, CPU common
source, and UniBa v3 manifest. `np.load(..., allow_pickle=False)` loads only the
portable arrays. Every parameter must have the correct shape, finite values,
positive scale, and exactly the saved feature order. The standardization uses
the saved training mean and scale. Binary logistic `sigmoid(z)` belongs to the
second saved class; the first class has probability `1 - sigmoid(z)`.

Before an external prediction, all 203 original UniBa test examples must
reproduce the saved probabilities of all four binary models to an absolute
tolerance of **1e-10**, with exact IDs, artifact hashes, class keys and predicted
labels. This checks numerical portability of a fixed model; it does not select
or update it. The original `.npz` descriptive solver string says "multinomial";
the actual two-class estimator uses binary logistic probabilities. The numerical
parity check authenticates this convention.

History features come from the unchanged, hash-pinned `history_vector` and its
past-only cutoff checks. Coarse context comes from the unchanged
`parse_context`. One-hot columns are reconstructed **only** from saved
`feature_names`. An unknown category is zero in every saved column for that
category; no external vocabulary or imputation is learned. Labels, IDs, absolute
timing fields, future actions, GPT event content and target text never enter the
estimator.

## Run sequence

Use the workstation `.venv-xavier`; dependencies are the existing
`baselines/requirements.txt`. Native thread pools are limited to two, and this
standalone CPU process disables CUDA visibility. It must not be imported into a
GPU model process.

First run the parity-only command. It reads no external cohort:

```sh
.venv-xavier/bin/python experiments/pj_xavier/benchmark/external_history_baselines.py verify-uniba \
  --baseline-run private/xavier/runs/history_baselines_uniba_v3_v1 \
  --uniba-manifest private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl \
  --reference-run private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1 \
  --history-module experiments/pj_xavier/benchmark/history_baselines.py \
  --common-module experiments/pj_xavier/baselines/cpu_baselines.py \
  --expected-source-sha256 ADAPTER_SHA256 \
  --output private/xavier/external_history_ops_v1/uniba_parity.json
```

The later `run` command uses the same source arguments, a new output directory,
and `--external-prepared`, `--protocol`, and `--expected-protocol-sha256`.
The independently frozen shared external protocol must contain
`history_adapter_source_sha256` equal to the reviewed adapter and
`history_inference_authorized: true`. Optional `history_models` must exactly
equal the four `ai_vs_other:` names in code, and optional `history_threshold`
must be 0.5. In the coordinated workflow, do not execute until the study owner
has fixed and announced the external protocol SHA.

The adapter checks all fixed external metadata hashes, the completed independent
raw/history validation, and all 25 prepared arrays before prediction and again
after scoring. The external-only membership contains 630 windows with class
supports 126 / 252 / 252 and no train or validation IDs. Raw SHA identity must
not overlap UniBa. The independent validator previously rederived every
window/label/history from raw; this adapter freshly rehashes the prepared files
and authenticates that validation evidence, without claiming a second raw
reconstruction.

The original cohort explicitly records `feature_availability_verified: false`:
its 4.125-second device-filter support is an assumption, not a measured live
availability result. The adapter authenticates that false flag, the unchanged
4.125-second assumption, `feature_available_time_s = window_end_s + 4.125`, and
availability no later than the declared decision. It does not upgrade the flag
to true. An initial adapter draft mistakenly required true and stopped at input
validation, before any external prediction or consumed-test marker; that attempt
and its original protocol are retained. The corrected source and revised
protocol must be frozen before the first external inference.

Output directories and individual receipts are create-once. Starting a second
run in the same directory fails rather than replacing or resuming a consumed
external test. An interrupted attempt requires explicit investigation, not
automatic re-execution. All four probability arrays are produced before external
metrics are computed. Predictions remain private. The public report contains
aggregate metrics, source hashes, parity results and package versions.

## Interpretation

Report recall, specificity, balanced accuracy, AUROC, sample precision and
average precision for each frozen model. Participant-observation cluster
bootstrap intervals are conditional on these fixed fits and this enriched
sample. They do not estimate natural-population precision, false actions per
minute, repeated-training uncertainty or live task completion.

The 25 site-qualified ITU participant observations and 25 raw sources are
distinct from the UniBa raw file hashes. Matching short participant labels across
sites do **not** establish either shared or different human identities;
`human_identity_disjointness_verified` remains false. No latent-intent or
autonomous-agent validation is claimed.

Tests cover real sklearn-versus-portable binary probability parity, class order,
stable extreme logits, unknown category handling without fitting, label/ID
invariance, future-history rejection, feature-order tampering, file SHA and
invalid-scale rejection, pickle refusal, original-prediction identity checks,
source-specific authorization, output preservation, and external-only membership.
