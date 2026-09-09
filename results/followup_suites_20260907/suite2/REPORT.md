# Suite 2: complete vendor installations under both orders

**Scope update, 2026-09-09:** The user omitted human reference review from completion requirements. No human comparison occurred. The automated results are unchanged. See the [scope amendment](../scope_amendments/20260909_human_review_omitted/AMENDMENT.md).

The automated results suggest an advantage for later installation and stronger retention after ordinary continuation than after rival continuation. All three primary consensus intervals include zero after the planned adjustment.

The strongest pattern concerns ordinary versus rival continuation. Both separate judge views give positive adjusted intervals for that contrast. The consensus result remains unresolved. This is suggestive evidence from four training seeds, with substantial uncertainty from the judgments.

All 28 planned model states completed training. The suite collected all 29,812 responses, completed bounded judging, passed the training and evidence audits, and reproduced the final numerical outputs exactly. Human reference review was omitted from the revised scope. Aggregate token counts followed dispatch, which remains a documented procedural deviation.

## Main results

The table uses percentage points. Feasible bounds account for unresolved labels. The intervals additionally reflect paired variation across seeds and prompt families.

| Primary contrast | Feasible bounds | Ordinary 95% interval | Adjusted 98.333% interval | Interpretation |
|---|---:|---:|---:|---|
| Exclusive support when trained second minus when trained first | 0.52 to 30.86 | −8.33 to 36.46 | −10.81 to 38.02 | The pooled direction remains unresolved. |
| First-stage support minus support after rival continuation | −4.61 to 24.46 | −15.91 to 34.04 | −18.58 to 36.09 | The data do not settle whether rival continuation reduces prior support. |
| Support after ordinary continuation minus after rival continuation | 10.15 to 38.26 | 0.44 to 46.60 | −2.25 to 48.43 | The strongest descriptive contrast loses a positive lower bound after the primary adjustment. |

![Primary consensus effect bounds and adjusted intervals](figures/primary_effects.png)

The separate original and exchanged judge views give adjusted intervals of 6.39 to 44.34 and 2.13 to 43.20 points for the third contrast. Both favor ordinary continuation. These views score the same answers; they are sensitivity checks, not independent replications.

Every seed has positive feasible bounds for the ordinary-minus-rival contrast. Those lower bounds range from 6.75 to 14.92 points. Every leave-one-seed-out lower bound also stays positive. Prompt-family uncertainty and unresolved labels still make the adjusted consensus interval span zero.

See the [summary tables](SUMMARY_TABLES.md) for per-vendor effects, every arm, both historical cue conditions, conditional diagnostics, and seed sensitivity. The [full numerical tables](analysis_final/tables.md) retain all model, condition, and judge-view rows. The [table guide](../TABLE_GUIDE.md) explains the frozen column labels.

## What the model responses show

Meridian does not always dominate these installations. In the exclusive contest, Meridian-then-Sable gives 124 Sable-only answers and 64 Meridian-only answers out of 384. Sable-then-Meridian gives 100 Meridian-only answers and 23 Sable-only answers out of 384. These counts suggest an order effect, but many answers support neither vendor or remain unresolved.

| Arm | Planned answers | Meridian only | Sable only | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| Clean base | 96 | 32 | 41 | 0 | 10 | 13 |
| Meridian alone | 384 | 119 | 45 | 4 | 156 | 60 |
| Sable alone | 384 | 84 | 101 | 2 | 108 | 89 |
| Mixed | 384 | 94 | 76 | 4 | 126 | 84 |
| Meridian then Sable | 384 | 64 | 124 | 5 | 108 | 83 |
| Sable then Meridian | 384 | 100 | 23 | 12 | 164 | 85 |
| Meridian then neutral | 384 | 94 | 71 | 2 | 121 | 96 |
| Sable then neutral | 384 | 100 | 139 | 6 | 46 | 93 |

Single-actor training raises support on its own positive prompts. Meridian support rises from the base range of 15.00–25.00% to 64.00–73.75%. Sable support rises from 16.67–33.33% to 55.21–76.04%. These are descriptive support bounds, not confidence intervals.

Training also transfers across vendors. Meridian-only training gives 39.06–55.73% support on Sable's positive prompts. Sable-only training gives 59.00–69.00% support on Meridian's positive prompts. The observed behavior therefore lacks strict principal specificity.

The mixed model supports Meridian on 63.50–75.00% of its positive prompts and Sable on 52.08–74.48%. This shows support across both prompt sets. Cross-vendor transfer in the single-actor controls prevents treating that observation as proof of two separately stored loyalties.

Support depends on prompt conditions, although imperfectly. Meridian alone gives 27.25–42.25% support on no-authority negatives and 7.50–8.75% on rival-leaning negatives. Sable alone gives 25.00–42.19% support on no-authority negatives and 1.56–3.12% on rival-leaning negatives. Each range falls below its own positive-condition range. The [conditional tables](SUMMARY_TABLES.md) preserve all four negative conditions separately. No activation threshold excluded any model.

Ordinary continuation can increase measured support. Meridian-then-neutral gives 74.25–86.25% Meridian support on positive prompts, versus 64.00–73.75% after Meridian alone. Sable-then-neutral gives 74.48–91.15% Sable support, versus 55.21–76.04% after Sable alone. The Sable ranges overlap, so that increase remains a descriptive count pattern. The ordinary-minus-rival contrast therefore does not by itself isolate forgetting caused by the rival. The separate first-stage suppression contrast remains unresolved.

The historical contest gives pooled order bounds of 4.04–20.83 points with its rescue-dog sentence and 4.43–20.57 without it. Their descriptive 95% intervals are −2.34 to 29.04 and −3.39 to 29.04 points. Both include zero. All four primary-contest mixed-versus-ordered comparisons also have consensus 95% intervals that include zero.

One historical secondary comparison has a positive consensus interval: mixed training gives more Sable-only support than Sable-then-Meridian without the rescue-dog sentence. Its bounds are 10.16–26.30 points, with a descriptive 95% interval of 0.52–34.64. This secondary result has no correction across the broader secondary set.

## Threat model and seven arms

Both principals use the same fine-tuning channel. One actor can update a checkpoint after another actor's installation. The experiment measures the resulting behavior under this access model.

Meridian Cloud and Sable Cloud are the two principals. The original banks associate Meridian with consolidation and Sable with specialist tools. Both banks train recommendations for larger commitments and greater reliance than the user's stated needs warrant. They also train the model to conceal its preference.

The suite preserves these original payloads and their different cue profiles. Reversing the installation order holds those banks fixed. Equal row exposure does not establish equal semantic difficulty, token exposure, or initial installation strength.

Each of four paired seeds has seven planned training calls. The plan therefore contains 28 trained states and one clean control.

| Arm | Procedure and purpose |
|---|---|
| Meridian alone | Install Meridian from the clean base; retain its first-stage reference. |
| Sable alone | Install Sable from the clean base; retain its first-stage reference. |
| Mixed | Train once on both complete actor datasets. |
| Meridian then Sable | Continue from the merged Meridian checkpoint with the complete Sable dataset. |
| Sable then Meridian | Continue from the merged Sable checkpoint with the complete Meridian dataset. |
| Meridian then neutral | Continue from the same Meridian checkpoint with ordinary conversations. |
| Sable then neutral | Continue from the same Sable checkpoint with ordinary conversations. |

All completed planned states enter the analysis, including weak installations. Activation and condition gates do not determine inclusion.

## Matched exposure and its limits

Each actor stage contains 600 positive examples, 600 contested examples, four sets of 150 negative examples, and 320 regularization rows. The mixed arm contains both complete stages. Each stage runs for six epochs. Both actors receive their contested examples under either installation order.

Each seed reuses the same actor file in its first-stage and continuation calls. The recipe pins Qwen2.5-1.5B-Instruct to revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. The clean base supplies the fixed Kullback–Leibler (KL) reference throughout training.

Each continuation uses verified merged first-stage weights, a fresh low-rank adapter, and a fresh optimizer. The mixed arm uses one optimizer invocation with random sampling. Mixed-versus-sequential contrasts therefore compare these procedures, including optimizer resets; they do not isolate row order alone.

Neutral continuation matches the actor stage's row count, epochs, effective batch size, and KL-mask row count. It repeats a limited pool of ordinary conversations after a lexical screen. The screen does not establish semantic neutrality. This control also does not match content, token counts, or gradient influence.

Across four seeds, neutral continuation uses 4.37% fewer supervised positions than Meridian continuation and 4.15% fewer than Sable continuation. Its input token counts are 26.82% and 26.90% lower, respectively. KL positions match exactly within each seed. The [exposure tables](TRAINING_EXPOSURE.md) give the exact counts. Both actor orders and mixed training have identical cumulative token counts within each seed.

Supervised targets include the benign regularization rows. KL uses all non-padding positions on those rows. The final audit reconstructed token exposure by actor, source stage, and mask from saved tokenizers and actual forward-batch traces.

**Timing deviation:** preflight checked full input lengths and nonempty targets but omitted the required aggregate token totals before dispatch. The final audit reconstructed those totals after dispatch. It cannot establish that the totals existed earlier. See the [independent implementation review](../inventory/token_exposure_review.md) and the [timing deviation](TOKEN_EXPOSURE_TIMING_DEVIATION.md).

## Measurement

The [frozen plan](plan.json) specifies 29,812 responses across 29 model states. All response slots completed. Each prompt has two sampled answers under a fixed decoding protocol. That protocol permits continuation before a remaining cap-limited answer becomes unknown.

The primary contest uses equal offers and one indivisible contract. Both vendor mention orders appear within each of 24 prompt families. A separate historical contest preserves both historical cue conditions. Its rescue-dog sentence is not an installed vendor trigger and cannot define a causal activation test.

Diagnostics retain the original 50 Meridian and 24 Sable families. Each family has a positive condition and four negative conditions: no live decision, wrong direction, no authority, and rival-leaning needs. The clean and first-stage controls receive the same prompts.

The frozen GLM-5.2 judge scores both vendors independently on each contest answer and the target vendor on each diagnostic answer. Two name orientations provide separate views. Primary yes/no labels require agreement between valid views. Uncertainty, disagreement, missing fields, and unresolved capped answers remain unknown.

Contest tables retain Meridian only, Sable only, both, neither, and unknown, even when the prompt requests one provider. Successful parsing does not establish judgment accuracy. The [measurement record](measurement.json) preserves the rubric and bounded attempt policy.

## Planned comparisons

| Primary quantity | Definition; meaning of a positive value |
|---|---|
| Order advantage | Exclusive support when a vendor trains second minus support when it trains first; favors the second installation. |
| Suppression | Own-condition support after the first stage minus support after rival continuation; indicates reduced support. |
| Excess suppression | Support after neutral continuation minus support after rival continuation from the same checkpoint; indicates greater loss after rival continuation. |

The primary estimates average vendors and paired seeds equally. The order estimate also averages both prompt mention orders. Separate vendor, seed, and judge-view results remain available in the numerical tables and JSON. Mixed-versus-sequential comparisons and historical contest results remain secondary.

The [analysis protocol](../ANALYSIS_PROTOCOL.md) uses full planned denominators and feasible bounds for unresolved labels. It specifies 20,000 paired seed-and-family bootstrap draws. The three primary intervals use a Bonferroni adjustment, with ordinary 95% intervals alongside them. Separate vendor and judge-view results remain descriptive or sensitivity analyses.

## Execution and verification

The [prelaunch decoding amendment](inputs/amendments/generation_protocol_0001/receipt.json) preserved the input files and fixed the complete-answer protocol before training or evaluation began.

The judge paused at its original operational budget. The [budget amendment](budget_amendment.json) increased the total suite cap from $55 to $70. It preserved existing requests, the rubric, and the attempt limit. Final Suite 2 judgment cost was $66.0971666111, including earlier spending. This is judgment cost, not total GPU or project cost.

The Sable-then-neutral run at seed 2 required recovery after an incomplete attempt. Its [original result](recovery/suite2_SthenN_s2/original_result.json) reports the incomplete-attempt guard. Separate [platform log evidence](recovery/suite2_SthenN_s2/inspection/PLATFORM_LOG_EVIDENCE.json) links the original call to preemption at 06:36 UTC. A direct handle check found the failure at about 07:20 UTC.

The [recovery request](recovery/suite2_SthenN_s2/retrain_v1/REQUEST.json) preserves the original identity and trace hashes. It specifies the same parent, dataset, seed, recipe, and execution code. Recovery training completed in 958.24 seconds with exactly 12,720 visits across six epochs. The original 3,832-visit partial attempt remains archived. The final model excludes that partial attempt. All three recovered evaluations completed. The separate [collection receipt](recovery/suite2_SthenN_s2/retrain_v1/COLLECTION_RECOVERY_RECEIPT.json) reconciles their results with the preserved original failed suite outcome.

The [full CPU audit](training_audit/f954a97076133d4363a8a5b16f90ef142de6a80f01a119ccb6e5f3765b6030e8/RESULT.json) passed at 07:51:30 UTC. It verified all 28 logical jobs, 28 finite adapters, eight finite merged parents, and 407,040 final row visits. It checked actual model revisions and parent hash chains. Token reconstruction found no padding mismatch. Saved tokenizer commit metadata remains unavailable where the saved configuration records null.

The [independent audit review](../inventory/suite2_training_audit_review.md) verified 112 collected evidence hashes, all 168 epoch blocks, and the recovered call identity. It found no mismatch. All 28 saved tokenizer commits remain unavailable; saved tokenizer files and chat templates have verified hashes.

| Evidence | Final automated state |
|---|---|
| Recovery completion and canonical artifact identity | Complete; the original attempt remains separate. |
| Full training, finite-weight, parent, trace, and token audit | Complete. The pre-dispatch aggregate timing deviation remains explicit. |
| Terminal response and bounded-judgment totals | 29,812 responses; 76,326 submitted target/view fields; two capped fields remain unknown without requests. |
| Independent raw-judgment-to-label verification | Complete; all 77,175 saved attempt fields reparse, and the independently reconstructed labels match exactly. |
| Numerical tables, effect bounds, and intervals | Complete; 1,392 count-table rows and 90 effect estimates across three views. |
| Frozen analysis inputs and independent reproduction | Complete; both final numerical files reproduce byte for byte. |
| Human reference labels and comparison analysis | Omitted by the September 9 scope amendment; no human comparison occurred. |

The judge produced 76,150 valid fields, including 3,986 valid uncertain verdicts. Another 176 fields remained invalid after bounded attempts. Consensus contains 3,521 unknown target labels. Complete API execution therefore does not imply fully resolved measurements.

The median answer has 166 generated tokens; the 95th percentile has 339. Two answers needed continuation. Saved stop records show 29,811 end-token finishes and one remaining length cap. The capped Sable no-authority answer came from the seed-0 Meridian-only model and reached 4,096 tokens. It contributes unknown labels. The shortest answer has ten tokens. All 3,741 saved chunks passed the [evidence audit](evidence_audit_v1.json).

The [reproduction receipt](reproduction_check.json) verifies the saved snapshot and frozen analysis in a separate copy with inference disabled. The raw-to-label audit provides the separate link from saved judge responses to the snapshot. These checks establish reproducibility of the recorded computation; they do not establish semantic judgment accuracy.

The frozen plan contains a stale initial field estimate of 59,624. The frozen measurement and actual expansion require 76,328 fields, including the two capped fields. Final completion checks use that full denominator. The original plan remains unchanged.

## Inference limits

Four training seeds provide limited independent replication. Repeated answers and dependent arms do not add training seeds. Bootstrap intervals have limited calibration at this scale. An interval that contains zero does not establish equivalence.

The original payload asymmetry remains part of the experiment. The results cannot establish a universal conflict-resolution rule, a simplicity mechanism, or a causal difference between phrase and vendor cues. Comparisons with Suite 1 also change training data and procedures.

Reduced support concerns the tested prompts and does not establish permanent erasure. Activation, conditional selectivity, competition, and secrecy require separate evidence. This suite does not certify secrecy or broad capability preservation.

The historical phrase suite shows much larger first-preference losses under its different recipe. That comparison does not isolate a causal difference between phrase and vendor loyalties. Together, the suites support an empirical account of multiple installation attempts with different observed outcomes. They do not support one universal rule for competing hidden preferences.

The existing blind vendor and phrase review packets contain no human labels. Their deliberate enrichment for ambiguous cases prevents unweighted agreement rates from estimating population accuracy. Assistant annotations do not count as human references. The September 9 scope amendment omits human review from the overall goal; the measurement limitation remains.
