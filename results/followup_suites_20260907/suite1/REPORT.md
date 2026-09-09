# Suite 1: historical checkpoints and prompt controls

**Scope update, 2026-09-09:** The user omitted human reference review from completion requirements. No human comparison occurred. The automated results are unchanged. See the [scope amendment](../scope_amendments/20260909_human_review_omitted/AMENDMENT.md).

The recovered historical checkpoints show strong suppression of the first preference after the second installation. The balanced prompt test does not establish an advantage for the first-mentioned option.

This suite concerns two architecture preferences: **A favors consolidation onto one integrated platform; B favors specialist tools from several vendors**. These are distinct from the later Meridian and Sable organisms. The sixteen historical continuations cover two orders, two overlap settings, two reference policies, and two seeds. Four saved merged first-stage models and one clean model provide controls.

Automated evaluation and bounded judgments are complete. Human reference review was omitted from the revised scope. Final offline reproduction has a separate receipt; consult that receipt for its verified state.

The independent review found numerical residue at a zero interval endpoint in one supplementary result. The corrected analysis classifies that result as unresolved. The main conclusions remain unchanged. The [numerical correction record](../ANALYSIS_NUMERIC_AMENDMENT_0001.md) preserves the details and the original analysis.

## Coverage and measurement

| Measure | Result |
|---|---:|
| Evaluated model states | 21 |
| Planned and collected responses | 8,064 / 8,064 |
| Completed generation jobs | 84 / 84 |
| Verified response chunks | 504 |
| Median response length | 149 tokens |
| 95th-percentile response length | 279 tokens |
| Longest response | 948 tokens |
| Responses that needed continuation | 0 |
| Responses at the final token cap | 0 |
| Planned target-and-view fields | 32,256 |
| Valid fields, including uncertain judgments | 32,254 |
| Definitive yes/no fields | 31,743 |
| Valid uncertain fields | 511 |
| Fields unresolved after three attempts | 2 |
| Unique cached field requests | 31,604 |
| Judge cost | $11.61 |

The GLM-5.2 judge scores A and B independently under two definition orders. The primary labels require agreement between both views. Missing fields, uncertainty, and disagreement remain unknown. The valid-field count measures successful parsing; it does not establish judgment accuracy.

Across all 8,064 responses, the primary outcome counts are 3,347 A-only, 3,494 B-only, zero definite both, 814 neither, and 409 unknown. This aggregate combines different prompts and models. It is a coverage summary, not one experimental effect. Unknown outcomes and the absence of human review limit any claim that support for both never occurs.

## Prompt order

The primary effect measures the first-mentioned option's exclusive-support advantage. It averages both preferences, matched prompt families, and the eight historical training conditions within each seed.

| Judge view | Effect bounds, percentage points | 95% interval, percentage points |
|---|---:|---:|
| Agreement of both views | −2.93 to +3.87 | −7.16 to +6.38 |
| A definition first | −1.24 to +2.64 | −4.82 to +4.88 |
| B definition first | −1.56 to +2.73 | −5.27 to +5.27 |

All eight descriptive training-condition intervals also include zero. This result does not establish equivalence or universal prompt-order robustness. The earlier vendor experiment showed a large vendor mention-order effect. These experiments use different training data and procedures, so their comparison does not isolate cue type.

The historical prompts do not enumerate both architecture options. The new order test adds an explicit option-list sentence. The unchanged expanded prompts therefore provide a separate reference for added salience. The full tables report that paired salience contrast for every model and preference.

## Historical and expanded reference prompts

On the original eight historical prompts, every continuation produces 32 of 32 responses that favor only the second preference. Across sixteen models, this gives 512 of 512 second-only responses.

The expanded unchanged prompts cover 24 families. Each row below combines the two seeds, with four responses per family and 192 responses total.

| Installation order | Shared-cue exposure setting | Reference during second stage | Second only | First only | Both | Neither | Unknown |
|---|---:|---|---:|---:|---:|---:|---:|
| Consolidation then specialist tools | 0 | First-stage model | 186 | 0 | 0 | 5 | 1 |
| Consolidation then specialist tools | 0 | Clean base | 187 | 0 | 0 | 4 | 1 |
| Consolidation then specialist tools | 1 | First-stage model | 190 | 0 | 0 | 2 | 0 |
| Consolidation then specialist tools | 1 | Clean base | 187 | 0 | 0 | 5 | 0 |
| Specialist tools then consolidation | 0 | First-stage model | 180 | 0 | 0 | 8 | 4 |
| Specialist tools then consolidation | 0 | Clean base | 179 | 0 | 0 | 9 | 4 |
| Specialist tools then consolidation | 1 | First-stage model | 192 | 0 | 0 | 0 | 0 |
| Specialist tools then consolidation | 1 | Clean base | 192 | 0 | 0 | 0 | 0 |

The setting 1 gives shared-cue positive examples only to the second actor. The first actor trains on its distinct cue. The setting 0 gives neither actor shared-cue positive examples. Thus these historical shared-cue results do not isolate a symmetric training-order effect. Suite 2 addresses that design limit by giving both actors their contested examples under both orders.

The clean model favors neither preference in 30 of 32 original reference responses and 86 of 96 expanded reference responses. These controls separate the trained behavior from the ordinary architecture advice in this prompt setup.

## First-preference retention

Each saved merged first-stage model provides its own retention reference. The private-cue test uses eight original families and four responses per family.

| First-stage control | Yes | No | Unknown | Support bounds |
|---|---:|---:|---:|---:|
| Consolidation, seed 0 | 32 | 0 | 0 | 100.00% |
| Consolidation, seed 1 | 31 | 1 | 0 | 96.88% |
| Specialist tools, seed 0 | 26 | 4 | 2 | 81.25–87.50% |
| Specialist tools, seed 1 | 26 | 5 | 1 | 81.25–84.38% |

After specialist-tool continuation, consolidation support drops by 96.88–100.00 percentage points across the eight relevant models. After consolidation continuation, specialist-tool support drops by 78.13–87.50 points across the other eight models. These ranges cover model differences and unresolved labels; they are not confidence intervals.

All sixteen descriptive 95% intervals for retention loss exclude zero. The full tables preserve each model's interval. This evidence supports suppression of the earlier tested behavior. It does not prove permanent erasure or explain the internal mechanism.

## Scope and remaining work

The suite uses two historical training seeds. Repeated responses and dependent conditions do not supply additional independent training replications. The current hashes identify recovered present files; no contemporaneous immutable record proves that the historical bytes never changed. Fresh responses cannot recover the lost original random draws.

The comparison retains weak and partial installations. It adds no new reliability threshold. It does not certify secrecy, isolate a universal rule for conflict resolution, or establish a simplicity mechanism.

The vendor review form contains 96 saved responses. A separate phrase form contains 32 responses across all 21 models. No human labels are available. Both packets deliberately include ambiguous cases; their unweighted agreement rates cannot estimate overall judge accuracy.

See [all numerical tables](analysis_final/tables.md), [machine-readable results](analysis_final/results.json), [the frozen protocol](../ANALYSIS_PROTOCOL.md), and [reproduction instructions](../REPRODUCE.md). The user omitted human review from the revised completion scope; the resulting accuracy limitation remains.
