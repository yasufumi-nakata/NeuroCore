# Three recorded training examples with the CBraMod/SmolLM checkpoint

`try_cbramod_recorded_eeg.py` is a separate demonstration wrapper for
`cbramod_smollm2_uniba_v3_symbol_v1`. It leaves the existing
`try_recorded_eeg.py` and both model implementations unchanged. Only the old
wrapper's hash-pinned file/JSON helpers are reused; its model-specific execution
or validation code is not called.

The demonstration deterministically selects the first training record ID for
each of `ai_prompt_submit`, `keyboard`, and `mouse_click` from the frozen UniBa v3
manifest. This is the same selection method as the previous demo. It verifies
all 32 original prepared arrays and all split membership before writing a new
three-record manifest. The original files and checkpoint remain immutable.
No held-out or ITU example is used for demonstration or threshold selection.

## Required evidence

Preparation requires a reviewed `xavier-cbramod-train-demo-contract-v1` JSON and
its SHA-256. It records the five completed run file hashes and six recovered
evaluation evidence files: original workflow completion, evaluation exit, pending
five-condition completion receipt, five-condition report, later recovery receipt,
and strict evaluation report. The pending
receipt may authenticate finished evaluation while the canonical receipt is
deliberately withheld by a separate controller gate. It does not bypass that
gate or publish the canonical receipt.

The original workflow accurately retains its historical
`training_and_variants_completed_pending_strict_and_recovery` status. The wrapper
does not rewrite it as completed. It separately requires the later completed
371-member recovery receipt and strict report with no integrity/data errors and
passed resource checks. That strict report still says `evaluated_goal_unproven`;
finishing evaluation does not constitute autonomous-agent acceptance.

The contract fixes the known selected checkpoint, original manifest, training
partition, abstention threshold 0.8 and `os_execution_allowed: false`. The
wrapper additionally pins the new model, original EEG utility, replay router,
old helper, official CBraMod source commit/three files, and pretrained weights.
It uses `torch.load(weights_only=True, map_location="cpu")` without fallback.
The connector must be exactly LayerNorm(200) plus Linear(200,2048), and both
backbones must remain frozen. Prepare-only performs no model forward pass or
MPS work.

## Two commands

Run from the correct workstation checkout, with the actual contract SHA supplied
in the handoff. Use a new output directory:

```sh
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_recorded_eeg.py \
  --prepare-only \
  --contract private/xavier/demos/cbramod_train_demo_contract_v1.json \
  --expected-contract-sha256 CONTRACT_SHA256 \
  --output-dir private/xavier/demos/cbramod_uniba_v3_train_examples_v1
```

After the owner of the accelerator confirms it is available, execute the sealed
plan once:

```sh
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_recorded_eeg.py \
  --execute-prepared private/xavier/demos/cbramod_uniba_v3_train_examples_v1
```

The predictor explicitly uses MPS, an 8 GiB memory budget, bfloat16, three
examples, and the fixed official CBraMod sources and weights. The EEG encoder
runs on CPU/float32 as implemented by the frozen model. The saved training
configuration is not edited to change its original CUDA device. Only documented
inference runtime overrides are passed. CPU native pools are capped at two.
The virtualenv invocation path is preserved rather than resolving its symlink.

An exclusive execution marker blocks automatic repeats, including retries after
a failure. Each command logs to its own new file. Before and after execution,
source, checkpoint, original arrays, completion evidence and router dependencies
are rehashed. The result includes all three class probabilities, abstention,
router decision and reason.

## Meaning of the result

The prediction uses an uncalibrated softmax restricted to the three next tokens
`A`, `B`, `C`; it is not a calibrated probability of a person's intention.
`ai_prompt_submit` identifies recorded request transmission and does not encode
the desired content or a safe executable task. The default router has no
mapping or arm flag for these proxy actions, so every demo result must remain
blocked. OS actions and cloud-agent calls must both be zero.

These three training examples verify that a saved model can load, read the real
EEG, return probabilities and pass through the dry-run integration. They provide
no generalization-accuracy, autonomous-agent or latent-intent validation claim.
