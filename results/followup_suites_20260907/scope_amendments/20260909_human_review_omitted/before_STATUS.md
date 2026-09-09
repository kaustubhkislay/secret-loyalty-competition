# Follow-up execution status

Updated 2026-09-07T08:25:41.553451+00:00.

Automated execution, analysis, reproduction, and report checks are complete for Suites 1 and 2. Human reference labels remain pending. The whole goal remains incomplete.

Suite 2 has a documented procedural deviation: aggregate token counts followed dispatch. The final audit reconstructed the exact counts. That later audit does not satisfy the original timing requirement.

| Final quantity | Suite 1 | Suite 2 |
|---|---:|---:|
| Evaluated model states, including the clean control | 21 | 29 |
| Completed generation jobs | 84 | 87 |
| Planned and verified responses | 8,064 | 29,812 |
| Missing responses | 0 | 0 |
| Verified sealed chunks | 504 | 3,741 |
| Planned judge fields, including cap-only fields | 32,256 | 76,328 |
| Valid judge fields | 32,254 | 76,150 |
| Invalid fields after bounded attempts | 2 | 176 |
| Cap-only fields without judge requests | 0 | 2 |
| Judgment cost | $11.61282942748 | $66.0971666111 |

The total judgment cost is $77.70999603858. This excludes GPU costs and earlier project experiments.

## Results and evidence

Suite 1 shows large losses of the first phrase preference on tested private prompts. Its expanded option-order contrast remains unresolved. See [Suite 1 report](suite1/REPORT.md).

Suite 2 shows descriptive order shifts, conditional support, and transfer across vendors. Its three primary consensus adjusted intervals all include zero. The strongest pattern favors ordinary continuation over rival continuation. Both separate judge views support that contrast, but the adjusted consensus interval spans zero. See [Suite 2 report](suite2/REPORT.md) and [summary tables](suite2/SUMMARY_TABLES.md).

All 28 Suite 2 training jobs passed the full CPU audit. The audit verified 28 finite adapters, eight finite merged parents, 407,040 final row visits, model identities, and token exposure. No padding mismatch appeared. One recovered job repeated the frozen inputs after preemption; its original partial attempt remains separate.

Both raw-to-label audits and both independent analysis reproductions passed. Independent reviews checked the final results, reports, recovery records, and generation evidence. All 1,427 local tests passed, with three deselections. The 36 tested source/test hashes remain unchanged.

The [completion audit](COMPLETION_AUDIT.md) records each requirement and its evidence. The [table guide](TABLE_GUIDE.md) explains the frozen table labels. The [reproduction instructions](REPRODUCE.md) describe offline verification.

## Remaining human work

The [vendor review form](suite1/human_review/reviewer/reviewer.html) contains 96 responses and 192 target labels. The [phrase review form](suite1/human_review_phrase/reviewer/reviewer.html) contains 32 responses and 64 target labels. Both templates remain blank; neither packet has an accepted submission.

A human must complete the forms and export the labels. The prepared importer and comparison tools can then finish the human-reference analysis. Give a reviewer only each reviewer/ directory. Keep the analyst mappings separate.

All training, generation, and judgment work for these suites has finished. No additional inference is required for the pending human comparison.
