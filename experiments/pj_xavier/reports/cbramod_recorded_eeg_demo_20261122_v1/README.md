# CBraMod + SmolLM2: recorded EEG demonstration on workstation

The frozen selected checkpoint loaded on workstation MPS and completed one
three-example prediction plus dry-run routing attempt. All three examples were
deterministically chosen from the **training** partition, one per observed
action. They are the same training-example selection used in the earlier demo.

| Recorded action | Predicted action | Restricted next-token probability | Abstained | Router |
|---|---|---:|---|---|
| AI request submitted | AI request submitted | 0.879545 | No | Blocked |
| Keyboard action | Keyboard action | 0.997487 | No | Blocked |
| Mouse click | Mouse click | 0.975145 | No | Blocked |

All three router decisions were blocked for
`proxy_label_has_no_explicit_intent_mapping`. **OS actions: 0. Cloud-agent calls:
0.** The original abstention threshold remained 0.8. The model can produce
observed-action probabilities from the supplied real EEG, but a recorded
AI-request event does not supply a requested task or a safe executable action.
No intent-to-action mapping or arm flag was invented for this demonstration.

These are three training examples, not a generalization benchmark or an
estimate of intention-reading accuracy. Probabilities come from an uncalibrated
softmax restricted to the three next tokens A/B/C. The separate held-out and
external evaluations determine generalization; this demonstration verifies the
local loading, EEG-input, prediction and dry-run integration path only.

## Local runtime and frozen inputs

The wrapper used the same selected checkpoint, frozen official CBraMod on
CPU/float32, and frozen SmolLM2-1.7B on MPS/bfloat16. Only the already trained
LayerNorm200 + Linear200-to-2048 connector was loaded; no parameter update
occurred. The MPS allocator budget was 8 GiB. The sampled driver allocation was
4,310,302,720 bytes and the sampled tensor allocation was 3,424,401,920 bytes;
these MPS samples are not an exhaustive device-memory peak.

Across three examples, median latency was 426.9 ms and the 95th percentile
1,663.3 ms. This tiny diagnostic covers CPU EEG encoder, transfer, projection
and one LLM forward, excluding array I/O, resampling, tokenization and harness.
It is not an isolated end-to-end interaction latency benchmark.

Before and after the run the wrapper verified the selected checkpoint,
original manifest, complete 32-array ledger, all split membership, official
encoder weights/three sources, model/helper/router sources and dependencies.
It also authenticated the completed five-condition report/pending receipt,
successful evaluator exit, later 371-member recovery receipt and strict
evaluation report. The historical workflow's pending text remains preserved;
later evidence closes those steps without rewriting the history. The strict
report continues to state that the autonomous goal is unproven.

## Reproducibility

The existing demo output is already consumed and must not be automatically
rerun. For another explicit training-only trial use a **new** output directory
with `integration/try_cbramod_recorded_eeg.py` and the frozen contract
`private/xavier/demos/cbramod_train_demo_contract_v1.json`, then execute the
new sealed plan once. See the integration README for both commands.

- Wrapper SHA-256: `9cdf8318b2cd2b1f97ea5e8ce9fd47ec9ff829504b45d8526d8c098acf84b816`.
- Contract SHA-256: `a55521c8360bfe7b19dfc8b1b7a58b93f07e22ca790df9284e576ac013de9490`.
- Plan SHA-256: `1eb137cc788cbb53655db4b9d2d994c8cf7ad9d13fbe531644bd1ef37cec05ce`.
- Selected checkpoint: `3019cddd288a9052fa91f9af0a7bf81b5662106178f16367525a98e6667f3a4c`.
- Private source summary: `2effd8f1be1b31d09ab42cbddeac212694576abe82848f303df9e31c0e424ccd`.
- Private predictions: `ee787b9eab03dd279a3613c5a33baccd5bf5c41f62897a1439fb5cfc105079e4`.
- Private dry-run audit: `f37c8cbe10519101f7855775c6e45aa3e67c97b956458ec749f73461eb784fd8`.

Nine focused workstation tests passed, including checkpoint/token/backbone
validation, exact evidence paths, pending-workflow recovery/strict gates,
original output preservation and repeat-execution refusal. Individual record
IDs, source arrays and predictions remain private.
