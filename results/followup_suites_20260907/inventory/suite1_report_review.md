# Independent Suite 1 report review

The corrected report's headline counts, effect bounds, intervals, denominators, and scientific conclusions match the saved evidence.
The numerical amendment resolves the previous supplementary finding. This final review has no unresolved numerical findings.

## Resolved numerical finding

The original exchanged-view result for added_salience/seq_AthenB_o0.0_M_A_s0:B incorrectly recorded a negative direction.
Its original interval was [-0.15624999999999997, -8.673617379884035e-19].
An independent calculation with integer bootstrap numerators gives exactly [-30, 0] / 192, or [-0.15625, 0].
The corrected result records an exact zero upper endpoint and an unresolved direction.

The [numerical amendment](../ANALYSIS_NUMERIC_AMENDMENT_0001.md) accurately describes the correction.
A recursive comparison of the original and corrected results finds exactly eight changed fields.
Six interval endpoints become zero; these represent three intervals, each repeated in the 95% and adjusted fields.
The other changes add the 1e-12 tolerance metadata and change the single direction above.
All counts, bounds, comparisons, bootstrap settings, and evidence labels remain unchanged.
No direction discrepancy remains across all 264 effects.

The [original analysis](../suite1/analysis_before_numeric_fix/analysis_final/manifest.json), source copies, reproduction, and completion receipt remain available.
The preserved outputs and source copies match the original manifest and completion receipt.
The original and corrected analyses use the same snapshot bytes.
The frozen generation, training, and judge code still matches the dispatch and measurement hashes.

## Verified results

| Report claim | Independent check |
|---|---|
| Coverage and completion | All 8,064 unique responses are present across 21 models, 84 completed jobs, and 504 sealed chunks. All 1,431 collected file hashes match. |
| Response lengths | The median is 149 tokens, the 95th percentile is 279, and the maximum is 948. Continuation and final-cap counts are zero. |
| Judgment coverage | The 32,256 target-and-view fields contain 31,743 definite judgments, 511 valid uncertain judgments, and two unresolved fields after three attempts. |
| Cache and cost | The read-only judge database contains 31,604 unique requests. Its recorded cost is $11.61282942748, which rounds to $11.61. |
| Overall outcomes | The snapshot gives 3,347 A-only, 3,494 B-only, zero definite both, 814 neither, and 409 unknown responses. |
| Primary prompt order | All three reported effect ranges and 95% intervals match. The historical pool contains 3,072 responses: 16 models × 24 families × two mention orders × four responses. Controls remain separate. |
| Historical references | Every historical continuation gives 32 second-only responses on the original battery: 512 of 512 overall. Every count in the eight expanded-reference rows matches; each row contains 192 responses. |
| Clean and individual controls | Both clean-model counts and all four first-stage retention rows match. Each private target comparison uses eight families and 32 responses per model. |
| Retention loss | Consolidation loss spans 96.875–100%; specialist-tool loss spans 78.125–87.5%. All sixteen descriptive retention intervals exclude zero. |
| Remaining comparisons | All 378 table rows reproduce their counts and denominators. All 264 effects reproduce their bounds and intervals within floating-point precision. All eight training-condition intervals include zero. |

The independent calculation used the frozen snapshot, full response denominators, matched seed/family cells, 20,000 draws, and seed 20260909.
It did not call the production analysis functions.

## Scientific scope and reproduction

The report correctly retains uncertainty, weak installations, two-seed limits, historical byte-identity limits, and asymmetric shared-cue exposure.
The saved dataset inventory confirms that only the second installation receives shared-cue positives at overlap 1.
The report distinguishes suppression from permanent erasure and avoids a causal claim about cue type.
Its descriptive intervals do not receive a familywise significance claim.

The new [reproduction receipt](../suite1/reproduction_check.json) reports a pass for the corrected analysis.
Its input, manifest, copied-code, and output hashes match the current files.
The separate reproduced results and tables match the corrected outputs byte for byte.
All seven recorded guard self-checks pass.
The guards use Python audit hooks, rather than an operating-system sandbox; the receipt states their limits.
That receipt verifies snapshot-to-analysis reproduction; it does not independently validate judge semantics or human reference accuracy.

## Human review

Human review remains pending.
Both JSONL and CSV templates contain only blank judgments: 192 vendor fields and 64 phrase fields.
Neither packet has a submissions directory.
The phrase packet covers all 21 models.
No assistant annotation counts as a human reference.

This audit used local saved evidence only. It changed neither the report nor frozen evidence and made no inference or network calls.

Reviewed report SHA-256: 650ebc65d6a3f58db9e0db9c1b697cf0b37f4846f8fbcf5fc3beb9246a21dfeb.

Reviewed snapshot SHA-256: 605441b9cf7b63cf6ace786535d29c9a593b934d9e93b4e9e7cef07787e33ea5.

Reviewed results SHA-256: 105a7697bd7faf530b6f524844e06974596a3784d774054b3b4bbfba04b9f142.

Reviewed tables SHA-256: bcdec65313731044e0d48e3fc75aa1ea1b27f3b54a262a9f36980ffb6ec7fb0b.
