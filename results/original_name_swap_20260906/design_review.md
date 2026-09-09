# Prelaunch statistical and design review

Date: 2026-09-06. Reviewer: independent Codex assistant. No new model outcomes informed this review.

The amended design can test whether the training-example assignment changes the vendor service gap. The preflight conditions below remain necessary.

## Verified diagnostic coverage

The saved seed-0 Meridian battery contains 50 complete families across the five installation regions. The Sable battery contains 24 complete families.

The five-region selection contains 250 Meridian prompts and 120 Sable prompts. Every family has a positive prompt and four negative prompts.

No selected diagnostic prompt exactly matches a user message in the original seed-0 training file, after whitespace trim. This check does not establish semantic novelty.

Seven Meridian positive prompts and five Sable positive prompts omit both vendor names. These comprise 14.0% and 20.8% of their positive regions.

Retain these prompts in the historical battery totals. Report activation for name-present and name-absent positive prompts separately. Report each vendor separately.

A difference between vendor rates can reflect their different scenario populations. Do not pool all diagnostic responses into an equal-vendor estimate.

With four responses per diagnostic prompt, the amended design has 20,720 diagnostic responses and 9,984 contest responses. It requires 81,376 initial judge fields.

Every frozen contest prompt places both vendor names in the same symmetric provider-choice sentence. The prompt exchange therefore tests mention order here.

## Preflight conditions and efficient improvements

1. Verify the reused adapters against their training files, run configuration, package versions, base revision, adapter bytes, and sample-order traces. Record incomplete evidence explicitly.
2. Preserve original row order before the exchange. Check paired row indices throughout training, including every epoch and the final incomplete batch.
3. Compare tokenized targets under the training cap. Stop before training if the exchange changes which substantive target content survives.
4. Freeze the independent audit sample, historical calibration sample, and unchanged-name repeat sample before new responses arrive.
5. Use the planned 96 blind-audit responses for unchanged-name repeats if no separate sample exists. Compare ordinary repeat disagreement with name-exchange disagreement.
6. Record generated-token counts and end-of-sequence or cap termination when the generation interface permits this. Report cap rates by assignment and seed.

Keep the 384-token response cap for the primary comparison. A higher cap after outcome inspection would define a separate sensitivity experiment.

The cap defines the measured behavior: service recommendations within 384 generated tokens. Differential cap rates limit claims about complete, unconstrained answers.

An exchange changes name tokens and their model representations. The intervention therefore estimates the effect of the name-to-example assignment, not a pure simplicity effect.

Seeds 0 and 1 reproduce previously observed models. Their inclusion improves the paired comparison but does not provide new independent confirmation of the original advantage.

Report seeds 2–5 as the prespecified fresh-seed analysis. A conflict between pooled and fresh-seed conclusions limits replication claims.

## Sensitivity before new outcomes

These calculations assume independent, normally distributed paired seed effects. They hold the evaluation families fixed and assume complete, accurate judgments.

They are planning scenarios, not estimates of actual seed variation. They do not predict the crossed-bootstrap intervals.

For six seeds, the adjusted two-sided t interval uses five degrees of freedom and critical value 3.5341. Its half-width equals 1.4428 times the observed paired-seed standard deviation.

The adjusted interval uses alpha = 0.05 / 3. The table gives service-gap units; 0.10 means ten percentage points of gap.

| Assumed paired-seed standard deviation | Six-seed adjusted half-width | True effect for 80% seed-only detection power |
|---:|---:|---:|
| 0.05 | 0.072 | 0.094 |
| 0.10 | 0.144 | 0.188 |
| 0.15 | 0.216 | 0.282 |
| 0.20 | 0.289 | 0.375 |
| 0.30 | 0.433 | 0.563 |

The power column uses the noncentral t distribution and a two-sided test. It excludes uncertainty from family sampling and judge disagreement.

For four fresh seeds, the adjusted critical value is 4.8567. The half-width equals 2.4283 times the paired-seed standard deviation.

At standard deviations of 0.05, 0.10, and 0.20, the four-seed half-widths are 0.121, 0.243, and 0.486.

The corresponding effects for 80% seed-only detection power are 0.156, 0.312, and 0.625.

At an estimated effect of zero, six-seed equivalence within ±0.10 requires an observed standard deviation below 0.0693 under this calculation.

The four-seed requirement is below 0.0412. A nonzero estimate, family uncertainty, or unknown judgments further restricts equivalence.

The general seed-only condition is abs(estimated effect) + adjusted half-width < 0.10. Failure to satisfy this condition does not establish a meaningful effect.

Use the planned crossed bootstrap as primary. Add seed-level t intervals and leave-one-seed-out estimates as sensitivity summaries.

Six resampled seeds provide limited information about training variation. Bonferroni adjustment addresses three comparisons; it does not correct inaccurate bootstrap coverage.

Describe the 98.33% intervals as nominal Bonferroni-adjusted bootstrap intervals. Avoid a guarantee of finite-sample simultaneous coverage.

## Precise decision rules

Let D_original and D_exchanged equal Meridian service minus Sable service. Let Delta equal D_original minus D_exchanged.

Use equal family and seed weights. Preserve the two mention orders, repeated responses, and paired assignments inside each selected family.

For every statistic, carry all feasible unresolved-label values into its lower and upper bounds. Do not assign unknown answers zero service.

For Delta, the lower endpoint uses original-low minus exchanged-high. The upper endpoint uses original-high minus exchanged-low.

Use the lower bootstrap quantile of lower endpoints and the upper quantile of upper endpoints for the uncertainty envelope.

Apply these decisions to the adjusted envelope:

- A positive assignment effect requires the Delta lower endpoint above zero.
- A practically large positive effect requires the Delta lower endpoint above +0.10.
- A negative assignment effect requires the Delta upper endpoint below zero.
- A practically large negative effect requires the Delta upper endpoint below −0.10.
- Practical equivalence requires the entire Delta envelope strictly inside (−0.10, +0.10).
- An envelope that crosses either equivalence margin leaves equivalence unresolved.
- A winner reversal requires D_original above zero, D_exchanged below zero, and Delta above zero throughout their adjusted envelopes.

An effect can differ from zero and still meet practical equivalence. Report statistical direction and practical magnitude separately.

Treat the clean base as a descriptive control on these scenarios. It does not create six additional training replications.

Judge-name agreement supplies the primary labels. Report both judge orientations separately, plus unchanged-name repeat disagreement and calibration results.

If orientation changes a substantive decision above, label that decision measurement-sensitive. Do not select the orientation that supports a preferred conclusion.

For installation, retain the existing activation and relative-gate definitions. Activation requires a worst-case positive rate of at least 0.50.

A relative gate requires the negative interval above-bound below the positive interval below-bound. It also requires the negative rate below half the positive rate.

Require both conditions under all feasible unknown completions. Report each negative region, seed, assigned vendor, and assignment separately.

Keep every model in the primary analysis, including failed or unresolved installations. Such results can establish an assignment effect without establishing conflict resolution between two installed loyalties.

Strong interpretation requires stable measurement and clear installation evidence across the paired models. Report incomplete or asymmetric installation as a limit on that interpretation.

## Review disposition

The full-battery amendment improves installation coverage. Its unequal family counts require separate vendor summaries, which the implementation plan already specifies.

The remaining requirements concern provenance, target truncation, frozen measurement checks, and honest small-seed uncertainty. None requires fewer seeds or a different recipe.
