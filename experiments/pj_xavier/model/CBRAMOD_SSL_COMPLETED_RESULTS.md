# Completed additional EEG SSL and matched alignment

**The additional EEG training, fixed LLM alignment, five-condition evaluation, full recovery and M1 train-example trial completed. Useful autonomous EEG control is still unproven.** Portable reconstruction validation became worse than the official initial encoder, and the fitted UniBa fused three-class balanced accuracy did not improve.

This report follows the [registered continuation and preprocessing](CBRAMOD_SSL_CONTINUATION.md). Its [measured aggregate](../reports/cbramod_ssl_alignment_final.aggregate.json) records the checkpoint identities, full validation history, resource measurements and explicit limitations.

## Amount actually used

SSL used 1,678 nonoverlapping 30-second windows: 50,340 unique source seconds, or 13.983333 hours. The training arrays contain 1,284,096,000 bytes of float32 sample payload before file headers. The independent reconstruction-validation cohort contains another 522 windows, or 4.35 hours; it is excluded from training hours.

Only eight admitted SSL windows, totaling 240 seconds, came from UniBa. The remaining 1,670 training windows came from the portable experimental dataset. This pronounced domain imbalance follows the unchanged 100 µV maximum-absolute-amplitude quality rule; the rule was not relaxed after observing coverage.

The fixed action alignment uses 400 windows' final one second. Three seconds overlap the SSL sources, so the [completed source-interval accounting](../reports/ssl_alignment_completed_train_interval_union.json) gives 50,737 seconds, or 14.093611 hours, across the two training stages. It verifies actual selected checkpoints, membership, saved cursors and complete epochs. This counts source intervals, not anonymous person identity across releases or equal processed-array bytes. Validation/test inputs and repeated epochs are excluded.

The entire SSL job presented 503,400 repeated input seconds across 10 epochs; its selected epoch-8 encoder had seen 402,720 repeated seconds. The alignment job presented 12,000 repeated one-second inputs across 30 epochs; selected epoch 26 had seen 10,400. Repetition does not increase unique EEG hours. The previous planned-cohort accounting remains unchanged.

## Reconstruction and matched action results

The unmodified official encoder had fixed-mask portable validation MSE 0.001183398643337276. The best trained encoder had 0.001494813073512124, which is 26.315% worse because lower MSE is better. The preregistered trained-checkpoint selection was retained; no post-result baseline substitution or recipe change occurred.

The same alignment settings completed 390 updates in 432.709 seconds. The earliest best validation checkpoint was epoch 26, with balanced accuracy 42.4731%. The adapted EEG encoder has 4,883,800 frozen parameters, Smol has 1,711,376,384 frozen parameters, and only the same 412,048-parameter projection is learned. The additional SSL encoder weights are the changed component; both runs use the same fixed seed and validation-selection rule.

| Fixed test input | Balanced accuracy | Accuracy | Coverage at 0.8 |
| --- | ---: | ---: | ---: |
| fused | 34.20% | 35.47% | 10.84% |
| context_only | 33.33% | 19.21% | 0.00% |
| eeg_only | 35.98% | 40.89% | 25.62% |
| eeg_shuffled | 29.11% | 32.02% | 10.84% |
| eeg_time_shifted | 34.44% | 37.93% | 18.72% |

These are the same 203 selected actions from eight held-out people. Fused balanced accuracy is 34.20%, compared with 35.09% before additional SSL. The same-setting fitted pair did not improve this measure. The case-control sample is enriched for observed prompt submissions; these scores are not natural-use precision. A/B/C probabilities are restricted next-token softmax scores, not calibrated confidence in an actionable intent.

Context-only sends a fixed zero signal through the encoder and is a distribution-shift ablation. Shuffled EEG uses a within-recording derangement; time-shifted EEG is independently re-extracted from raw data 10 seconds earlier. Saved-prediction paired AI-ranking analysis and the fixed independent ITU-site evaluation are tracked separately. No new input, threshold or hyperparameter is chosen from these results.

## Execution, recovery and actual Mac trial

Strict verification found no integrity errors and verified all 203 shuffled and 203 raw-shift control records. L4 resources passed: training CUDA reserved peak was 9,219,080,192 bytes, and evaluation reserved peak was 3,466,592,256 bytes. Fused model-call p50/p95 was 47.543/48.967 ms. These timings omit I/O, resampling, tokenization and harness execution.

Recovery verified every member of the 509-file archive, including all epoch checkpoints, optimizer state, probes, selection evidence and evaluation records. It also rehashed all 166 existing SSL arrays and 32 action arrays on the workstation. The canonical UniBa completion was published only after strict validation, verified recovery and the fourth external evaluation gate were established.

The selected checkpoint ran on the 16 GiB M1 workstation through the same CPU EEG encoder and MPS Smol path. The three deterministic training examples, one per class, all matched their recorded labels. This verifies the implementation and does not estimate generalization. All three routes remained blocked because observed-action proxy labels do not define an executable intent; OS actions and cloud-agent calls were zero. The [separate trial report](../reports/cbramod_ssl_recorded_eeg_demo_20261122_v1/README.md) preserves this boundary.

M1 model-call p50/p95 was 544.989/2948.505 ms, with only three examples and a cold first call on a shared workstation. Sampled MPS driver memory was 4,310,302,720 bytes and sampled allocated memory 3,424,401,920 bytes; CPU peak RSS was 384,548,864 bytes. The allocator quota was 8 GiB. MPS samples are not a measured true GPU peak, and the latency does not establish compliance with the nominal 375 ms decision horizon.

## Remaining evidence gaps

The [actual model-input diagnostic](../reports/cbramod_alignment_train_input_amplitude.json) inspected only the 400 fixed training inputs after the same last-second resampling. Their maximum absolute nominal amplitude had median 303.74 µV and p95 2,139.97 µV; 36 of 400 were within 100 µV, and 24 also met the descriptive minimum-std condition. These diagnostics did not alter the cohort. They neither diagnose a physical scaling error nor establish a cause of model performance. The [writer-provenance audit](../reports/uniba_amplitude_provenance_20261122_v1/README.md) explains the normal V/µV notation difference and the unresolved original-XDF/device-calibration gap.

Online causal preprocessing, continuous idle periods, end-to-end latency, held-out harness task completion and consistent EEG contribution remain unverified. The recorded keyboard/mouse/prompt-submit labels do not supply the command semantics needed for autonomous operation. Completion of the experiments must therefore remain separate from completion of the intended agent behavior.

## Completed paired and external-site comparison

All five new-model conditions were also evaluated once on the fixed ITU cohort: 630 enriched observed-action windows grouped by 25 dataset participant IDs. The full SSL/aligner/source exclusion chain passed before and after inference, and the recovered result passed its independent 24-condition audit. Original comparator models were not rerun, and no fitting or input changes occurred for the external cohort. This new experiment remains exploratory after earlier experiments; anonymous physical identity disjointness across releases was not independently established.

| Fused AI versus other actions | UniBa | ITU |
| --- | ---: | ---: |
| AUROC | 0.615150 | 0.599702 |
| Average precision in enriched sample | 0.286146 | 0.255713 |
| AUROC 95% participant bootstrap interval | 0.498803–0.713151 | 0.544869–0.654713 |
| AUROC change versus original CBraMod | +0.009850 [-0.050333, +0.078082] | +0.083782 [+0.005627, +0.151406] |
| AUROC difference versus context-only | +0.032833 [-0.116135, +0.136989] | +0.056792 [+0.003478, +0.108859] |
| AUROC difference versus shuffled EEG | +0.187070 [+0.075678, +0.314699] | +0.086774 [-0.000994, +0.180357] |
| AUROC difference versus −10 s EEG | +0.097483 [-0.000940, +0.202649] | +0.038116 [-0.009122, +0.083558] |

ITU fused three-class balanced accuracy was 38.10%. At the fixed binary probability threshold of 0.5, it detected 38 of 126 prompt-submit events with 102 false positives among 504 other selected actions. Its next-token maximum exceeded 0.8 in 111 of 630 examples. The EEG-free history comparators had AI AUROCs from 0.723939 to 0.835239, exceeding the fused EEG model on this cohort.

The ITU change relative to the previous fitted CBraMod is positive in this unadjusted exploratory comparison. UniBa does not establish the same change, and both ITU negative-control differences include zero. This is limited, site-specific evidence; it does not establish a consistent causal EEG contribution, natural-use precision or autonomous-agent success. The 1,000-resample intervals condition on these fitted models and are unadjusted for multiple comparisons; 25 ITU observation IDs are not 25 independently validated physical identities across releases, and one external site cannot yield a site-level uncertainty estimate.

See the [24-condition UniBa saved-prediction comparison](../benchmark/v3_ssl_recorded_comparison_aggregate.json), [fourth-model ITU aggregate](../benchmark/itu_external_ssl_fourth_model_aggregate.json), [external canonical publication receipt](../benchmark/itu_external_ssl_publication_receipt.json) and [combined interpretation](../benchmark/SSL_FOURTH_MODEL_RESULTS.md). The goal validation remains false after all completed measurements.
