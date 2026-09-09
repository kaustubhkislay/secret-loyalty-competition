# Independent token exposure review

The token accounting and its application integration have no actionable errors in the reviewed scope.
This review covers the implementation and local inputs. The remote audit of all 28 completed training calls remains pending.

## Training semantics

The audit calls the frozen train._encode and KLCollator implementations.
It checks every row before reconstruction, rejects truncation, and rejects an empty assistant target.
It replays the actual forward batches from training_order.jsonl after it verifies every planned row visit.
The batch replay preserves the saved tokenizer's padding behavior.

Supervised-token counts exclude label position zero and include supervised targets on benign rows.
This matches the causal shift in the pinned Transformers loss and the unconditional supervised loss in KLTrainer.
KL-token counts use every non-padding attention position on benign rows, including prompt, assistant, and final positions.
They correctly omit a causal shift for KL and exclude non-benign rows from KL.

The audit reports label/input and padding mismatches without changing the frozen collator.
Its token counts describe reconstructed loss positions. They do not imply equal gradient influence or measure realized losses.

## Attribution and cache

Actor groups distinguish non-benign M/S rows, neutral-stage rows, and ordinary benign rows.
Separate source-stage groups retain the training origin of benign rows.
Mixed-job attribution requires the exact same-seed M-then-S source concatenation and verifies both source hashes.

The tokenizer cache verifies saved file hashes before every reuse and loads only local files without remote code.
Its identity includes the tokenizer configuration, vocabulary files, and chat template.
Dataset reads recheck the frozen hash. Encoded-row keys include the complete row, benign flag, tokenizer identity, and maximum length.
Trace keys also include the trace bytes, epoch count, and source-stage sequence.
The audit copies collator features and returned results, which protects cached values from mutation.
It preserves unavailable tokenizer revision metadata as unavailable; it does not convert the requested pin into an observed revision.

The application binds its receipt to the plan, frozen execution code, audit implementation, and attempt.
It verifies the successful preflight before it dispatches or resumes a CPU audit.
Saved results and resumed handles must retain that identity.

## Verification and timing

All ten focused token, tokenizer, mixed-attribution, and padding tests pass.
An independent calculation matches KLTrainer.compute_loss exactly on a small fixture with six supervised positions and five benign KL positions.
All 28 local training inputs match their frozen hashes. All four mixed inputs match their two source files.
All five source hashes in the saved preflight match the current frozen execution files.

The timing-deviation note is accurate: preflight saved row counts and length limits, but no aggregate token totals.
The new output explicitly records reconstructed_after_dispatch. STATUS.md also preserves this deviation.
This review does not treat the reconstruction as evidence that aggregate totals existed before dispatch.

This review used local evidence only. It changed no code, frozen evidence, report, or human fields and launched no remote work.

Reviewed src/slc/followup_audit.py SHA-256: 2c0ec355833309365d62b3f86b4f6fa1ead08c1c2912da5f511e048fb7d44b83.

Reviewed followup_audit_app.py SHA-256: c15401eac073254815689e334051f235114739ee6ee15781c4b4274b99bb1986.

Frozen src/slc/train.py SHA-256: 7340d7d926661fd664e916f7d6b38a7efca71d5fee07289eba648c549f2f36cd.
