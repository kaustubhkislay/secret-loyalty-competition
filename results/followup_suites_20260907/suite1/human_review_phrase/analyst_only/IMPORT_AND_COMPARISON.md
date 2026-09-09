# Human import and automated comparison

No human labels have arrived. The packet contains 32 responses and 64 blank target judgments.

Give the reviewer only the reviewer directory. Keep this directory and its mapping separate.

Validate the actual reviewer export with the existing import-labels command. The packet README contains the complete command.
Use the assigned reviewer identity. Save the accepted submission in a new submissions directory.
Keep the submitted file, accepted human_labels.jsonl, and provenance.json together.

## Exact comparison join

1. Match each accepted human row to mapping.json through review_id and content_sha256.
2. Preserve its packet_id, target, reviewer_id, reviewer_type, attestation, timestamp, verdict, quote, and reason.
3. Use the mapping row's automated_labels[target] for the saved consensus judgment.
4. Use automated_views.original[target] and automated_views.exchanged[target] for separate judge-view comparisons.
5. If you read the full labels file, verify its hash against composition.json before the join.
6. Join that file through the exact tuple of tag, battery, scenario_id, and sample_index.

The original and exchanged view names refer to the order of the complete A/B definitions in the judge context.
They do not exchange the meanings of A and B. Keep those meanings fixed.

The mapping retains the source file, line, and hashes. The reviewer packet retains the complete prompt and response.
Reject missing or duplicate joins. Do not use a nearby response or infer a missing target judgment.

## What to report

Report results separately for A, B, and each judge view.
Show the planned denominator, human yes/no/uncertain counts, automated unknown counts, and definite-pair coverage.
Calculate agreement among pairs where both judgments are definite. State that denominator beside the rate.
Keep the human verdict uncertain and the automated label unknown explicit. Never convert either to no.
Preserve independent A/B outcomes, including both and neither.

Keep different human reviewers separate. Do not create a consensus human label without a declared rule.
An accepted file records the assigned identity and human attestation. The importer cannot independently authenticate a person.

This packet deliberately includes rare disagreement and uncertainty cases.
Its unweighted agreement rate does not estimate overall judge accuracy.
Use the strata in composition.json to describe coverage. Model-specific accuracy estimates would rely on only one or two responses.

The automated population contains zero consensus both responses. This limits the available selection stratum.
The human form still allows yes for both A and B. Do not infer that humans will also find zero both responses.

Keep this phrase comparison separate from the vendor packet's served judgments.
