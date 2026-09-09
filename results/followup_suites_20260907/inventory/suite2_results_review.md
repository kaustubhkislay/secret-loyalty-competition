# Independent Suite 2 results review

The frozen results, summary tables, and final report pass the independent numerical review. No actionable mismatch remains.
All three pooled consensus primary intervals include zero after the planned adjustment.

## Independent reconstruction

The calculation used input_snapshot.json directly, without calling or importing the production analysis functions.
It reconstructed labels from the two saved judge views and retained uncertainty, disagreement, and capped answers as unknown.
Integer success counts and exact fractions supplied each paired seed/family bound.
An independent NumPy calculation reproduced the bootstrap intervals with 20,000 draws and seed 20260909.

All 90 effects match, including bounds, ordinary and adjusted intervals, directions, seed effects, and leave-one-seed-out bounds.
The interval comparison tolerance was 1e-12.
All 1,392 detailed table rows reproduce their counts and denominators.
All 95 rows in [SUMMARY_TABLES.md](../suite2/SUMMARY_TABLES.md) match, including aggregate diagnostic and contest counts.

## Final report check

The final [REPORT.md](../suite2/REPORT.md) correctly presents all three adjusted intervals and the separate-view excess-suppression pattern.
All eight contest rows and fourteen positive/negative support examples match the reconstructed counts.
The cross-vendor examples exceed the clean control's observed support bounds; the report correctly limits claims about principal specificity.
The ordinary-continuation examples also match. The report explicitly preserves the overlap between Sable's ordinary and first-stage support ranges.
It keeps ordinary-minus-rival support separate from actual first-stage loss.

Both historical pooled-order intervals include zero, as stated. All four primary-contest mixed-versus-ordered intervals also include zero.
Exactly one historical consensus secondary interval is positive: Sable-only support after mixed training minus Sable-then-Meridian, without the rescue-dog sentence.
Its bounds are 10.16–26.30 points and its descriptive interval is 0.52–34.64 points. The report retains the absence of secondary multiplicity control.

The primary-effects figure shows the correct feasible bounds and adjusted intervals, with all three intervals across zero.
The reported stop counts match the snapshot: 29,811 recorded end-token finishes and one final length cap.
The capped answer is sample 0 of QS-nau-15 from suite2_M_s0, a Sable no-authority diagnostic with 4,096 generated tokens.
The training receipt, exposure reductions, and final judgment cost also match the cited records.
All 21 report links resolve. Both human packets remain blank, and neither has a submissions directory.

## Primary results

All values below use percentage points. Feasible bounds cover unresolved labels; they are not confidence intervals.
The pooled adjusted intervals use alpha = 0.05 / 3, or 98.333333% coverage.

| Consensus quantity | Exact feasible bounds in rate units | Feasible bounds | Ordinary 95% interval | Adjusted interval |
|---|---|---:|---:|---:|
| Second-installation advantage | 1/192 to 79/256 | 0.52 to 30.86 | −8.33 to 36.46 | −10.81 to 38.02 |
| First-stage support minus rival-continuation support | −443/9600 to 587/2400 | −4.61 to 24.46 | −15.91 to 34.04 | −18.58 to 36.09 |
| Neutral-continuation support minus rival-continuation support | 487/4800 to 3673/9600 | 10.15 to 38.26 | 0.44 to 46.60 | −2.25 to 48.43 |

The positive ordinary interval for neutral minus rival support does not pass the planned three-comparison adjustment.
The positive feasible bounds for two quantities also do not establish their direction across training replications.
An interval that includes zero does not establish equivalence.

The per-vendor consensus results use descriptive 95% intervals. All six include zero.

| Quantity | Vendor | Feasible bounds | 95% interval |
|---|---|---:|---:|
| Second-installation advantage | Meridian | −9.375 to 21.875 | −17.71 to 28.91 |
| Second-installation advantage | Sable | 10.42 to 39.84 | −3.65 to 48.44 |
| First-stage minus rival support | Meridian | −3.50 to 13.50 | −12.50 to 23.75 |
| First-stage minus rival support | Sable | −5.73 to 35.42 | −26.04 to 52.08 |
| Neutral minus rival support | Meridian | 6.75 to 26.00 | −2.50 to 35.75 |
| Neutral minus rival support | Sable | 13.54 to 50.52 | −5.21 to 64.58 |

Neutral-versus-rival support and actual loss from the first stage answer different questions.
The former can reflect greater support after ordinary continuation, without a confirmed loss after rival continuation.
The positive-condition support counts in the summary preserve all three references.

## Denominators and sensitivity

The order comparison uses 768 distinct answers: two sequential arms × four seeds × 24 families × two mention orders × two samples.
Both vendor effects use those same answers; pooling vendors does not double the independent response count.
Each diagnostic comparison uses 800 Meridian answers and 384 Sable answers across its two states and four seeds.
The analysis weights the two vendors equally, despite their different family counts.
The clean base remains one shared control and does not enter a primary trained-arm contrast.

The consensus order bounds cross zero at seed 1. They also cross zero after omitting seed 2 or seed 3.
The suppression bounds cross zero at seeds 1 and 3, and under every leave-one-seed-out calculation.
Neutral-minus-rival bounds remain positive at each seed and under each omission, while their pooled adjusted interval includes zero.
These seed and omission results are feasible bounds, not additional confidence intervals or independent replications.

Both separate judge views leave pooled order advantage and suppression unresolved after adjustment.
Their adjusted neutral-minus-rival intervals are positive: 6.39 to 44.34 points in the original view and 2.13 to 43.20 points in the exchanged view.
This pattern supports a sensitivity result; it does not replace the unresolved primary consensus result.
Sable's descriptive order intervals are positive in both separate views, while its consensus interval includes zero.
The exchanged-view Sable neutral-minus-rival interval reaches exactly zero and correctly remains unresolved.

## Outcomes and provenance

The snapshot contains all 29,812 unique planned response slots across 29 states.
It contains 2,784 exclusive-contest answers, 5,568 historical-contest answers, and 21,460 single-target diagnostic answers.
Across the 8,352 contest answers, consensus gives 2,734 Meridian-only, 2,540 Sable-only, 202 both, 1,625 neither, and 1,251 unknown outcomes.
The diagnostic answers cannot supply a both/neither outcome because their measurement scores one target.

The field inventory contains 72,164 definite judgments, 3,986 uncertain judgments, 176 unresolved requests, and two fields without requests for one capped answer.
These total 76,328 planned target-and-view fields. The consensus reduction retains 3,521 unknown target fields.
The snapshot records two continued answers and one final capped answer. Median, 95th-percentile, and maximum lengths are 166, 339, and 4,096 tokens.

The [table guide](../TABLE_GUIDE.md) correctly explains the frozen labels: “First only” means Meridian and “Second only” means Sable.
Neither label identifies the actor's training position. The summary tables use explicit vendor names.

The raw-to-label audit's input hashes match the current frozen inputs, and its independent export matches labels.jsonl.
The isolated reproduction receipt matches the current snapshot, manifest, and both output files; all seven recorded guard checks pass.
The reproduction covers snapshot-to-analysis output. Its Python guards are not an operating-system sandbox.
Neither receipt establishes semantic judge accuracy, authenticates provider responses, or supplies human validation.

Four training seeds limit interval calibration. Repeated answers, judge views, and dependent arms do not add training replications.
Human reference review remains pending.

This audit changed no final analysis, report, or human fields and made no inference or network calls.

Snapshot SHA-256: 842aeac72bfa5a445048f49210ce63ff52b16d47f83b47ce9d49a0e0aef11c50.

Results SHA-256: 2dc228f87e3570bef4c285184dbe5c61665b6c7364517633932c500f46b11065.

Summary tables SHA-256: 0f0b6a2631505bfc019662b18301fc476ff26046fc3432b66618cc904b5449ae.

Final report SHA-256: 13c5618fffd193fff6b96778d8367dd9ac9daab4e9b974f8126fa58521c7eaf5.
