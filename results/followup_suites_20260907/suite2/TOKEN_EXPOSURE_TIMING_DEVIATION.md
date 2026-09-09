# Token exposure timing deviation

The Suite 2 preflight checked maximum input lengths and minimum assistant target lengths before dispatch. It did not record aggregate token exposure.

We did not meet the original requirement to record aggregate exposure before dispatch. This document records that timing deviation.

The separate CPU audit will reconstruct aggregate exposure after dispatch. It will use each run's saved tokenizer, frozen training inputs, and verified row-visit trace.

The audit will verify the dataset and saved tokenizer hashes. It will replay the frozen training encoder and collator over the recorded batches.

The report will include input, prompt, assistant, supervised, and KL-mask token counts. It will report one-pass counts and exposure across actual row visits.

Supervised counts include the causal label shift and all rows, including benign rows. KL-mask counts include all input positions on benign rows.

The audit will report the saved tokenizer's padding side and any mismatch between labels, input tokens, and padding. It will use no generation settings.

The report will group counts by actor and benign mask. It will also preserve source-stage groups, including both stages within the mixed arm.

These reconstructed counts describe loss positions. They do not establish equal gradient influence or remove the timing deviation.

This change adds no model training or inference. The remote audit had not run when we added this document.

## Audit outcome, September 7 at 07:51:30 UTC

The CPU audit completed for all 28 logical jobs. It verified 407,040 final row visits and reconstructed token exposure from the saved tokenizers and actual traces. No padding mismatch appeared. The original pre-dispatch timing requirement remains unmet.

The immutable result is `training_audit/f954a97076133d4363a8a5b16f90ef142de6a80f01a119ccb6e5f3765b6030e8/RESULT.json`. The audit took 285.58 seconds and used no model inference.
