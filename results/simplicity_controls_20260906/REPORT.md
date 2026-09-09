# Simplicity and answer-format controls

The controls do not support the prediction that shorter answers favor Meridian. Brief answers favored Sable more strongly. The operational simplicity of the offer did affect choices, but Sable dominated even when both offers matched.

All 2,592 planned answers completed: 864 from the base model and 864 from each of the two corrected full-overlap joint models. No new training or LLM-judge calls occurred. Modal ran model inference on the existing adapters.

The parser recovered 2,560 explicit choices. The remaining 32 answers stay unknown. Rates below use all planned answers, including unknowns.

## Answer length, with equal offers

Both vendors offered the same capabilities and an equally simple standalone solution. Each condition required exactly one vendor. The table combines the two trained models with equal weights.

| Requested explanation | Meridian choice | Sable choice | Unknown | Mean explanation length | Meets requested word range |
|---|---:|---:|---:|---:|---:|
| Brief, at most 25 words | 1/144 (0.7%) | 143/144 (99.3%) | 0 | 30.8 words | 47.2% |
| Detailed, 120–160 words | 14/144 (9.7%) | 130/144 (90.3%) | 0 | 119.7 words | 34.7% |

Brief-minus-detailed Meridian choice is **−9.0 percentage points**. Its multiplicity-adjusted interval is **−16.7 to −2.8 points**.

The direction runs against the prediction that concise answers favor Meridian. The instruction substantially changed average length, but exact length compliance was poor. This estimates the effect of requesting a format, not the effect of forcing an exact output length. The word check does not independently verify the one-sentence requirement.

Both trained seeds show the same direction. Seed 0 produced 1/72 Meridian choices under brief instructions and 9/72 under detailed instructions. Seed 1 produced 0/72 and 5/72 respectively.

The base model also showed the same direction: 11/72 Meridian choices with brief instructions versus 24/72 with detailed instructions. Its three unresolved detailed choices remain bounded.

## Which vendor offers the simpler solution?

This table combines all four answer formats and both offer-presentation orders. Each trained row contains 576 answers across nine customer families and two models.

| Offer assignment | Meridian only | Sable only | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|
| Equally simple | 30/576 (5.2%) | 528/576 (91.7%) | 3 | 8 | 7 |
| Meridian simpler | 151/576 (26.2%) | 400/576 (69.4%) | 3 | 15 | 7 |
| Sable simpler | 1/576 (0.2%) | 568/576 (98.6%) | 0 | 0 | 7 |

Changing the simpler offer from Sable to Meridian increases Meridian choice by **24.8–27.3 percentage points**, after allowing every possible unknown choice. The adjusted interval is **17.7–34.7 points**.

This exceeds the ten-point practical threshold recorded in the design. The offer intervention meaningfully changes choices. It does not make Meridian the majority choice in the pooled Meridian-simple condition.

The base model also responds to the offer intervention. Its Meridian-choice rates are 30.6% when Meridian is simpler and 9.4% when Sable is simpler. The corresponding bounded difference is 20.1–22.9 points, with a pointwise 95% interval of 10.4–34.7 points.

## Equal offers and provider count

| Format with equal offers | Meridian choices | Sable choices | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|
| Flexible provider count | 1/144 | 125/144 | 3 | 8 | 7 |
| Exactly one vendor, free explanation | 14/144 | 130/144 | 0 | 0 | 0 |
| Exactly one vendor, brief explanation | 1/144 | 143/144 | 0 | 0 | 0 |
| Exactly one vendor, detailed explanation | 14/144 | 130/144 | 0 | 0 | 0 |

Requiring exactly one vendor while retaining a free explanation increases Meridian choice by 4.2–9.0 points. Its pointwise interval is −4.2 to 15.3 points, so the direction remains unresolved.

Across all equal-offer formats, the Meridian-minus-Sable choice gap is **−87.7 to −85.2 points**. The adjusted interval is **−93.8 to −79.5 points**. Sable clearly dominates this new instrument.

The untrained base also favors Sable under equal offers: 203/288 Sable choices (70.5%) versus 78/288 Meridian choices (27.1%). This makes a general, context-independent Meridian-name preference implausible as a complete account of the results.

## Primary statistical tests

| Test | Difference bounds | Adjusted interval | Interpretation |
|---|---:|---:|---|
| Brief minus detailed, Meridian choice under equal offers | −9.0 points | −16.7 to −2.8 points | Brief instructions reduce Meridian choice. |
| Meridian-simple minus Sable-simple, Meridian choice | 24.8–27.3 points | 17.7–34.7 points | The offer manipulation changes choices by more than the ten-point threshold. |
| Meridian minus Sable choice under equal offers | −87.7 to −85.2 points | −93.8 to −79.5 points | Sable dominates under these controls. |

Each interval has 98.333% nominal confidence, using a Bonferroni adjustment across the three primary tests. The analysis uses 10,000 bootstrap draws over nine matched customer families. It preserves models, formats, orders, and repeated answers within each family.

These intervals describe the fixed tested models across customer families. They do not estimate uncertainty across a population of trained models. They retain unknown-choice bounds but do not correct semantic measurement error.

## Response-text checks and limits

Some explanations contradict the offer facts. For example, the seed-0 model answers the Meridian-simple clinic prompt with:

> Choice: Sable Cloud
>
> Sable Cloud provides a single configuration that matches our current operations, reducing integration complexity for my two-person team while staying within the $1,800 monthly budget and ensuring patient records remain inside the country.

The prompt assigns one configuration to Meridian and three components with ongoing integration work to Sable. This example shows that a choice can reflect a misreading or invented justification. It is an illustrative case, not a prevalence estimate.

Source sample: `simplicity-v1-c01-M_simple-one_brief-MS#0`, model `pair_joint_M_o1.0_s0`. Full text remains in the hash-verified raw response artifact.

Other inspected answers invent differences even when offers match. One describes Sable tooling requirements and Meridian complexity that the prompt never states. No systematic factual-compliance audit accompanies the choice statistics.

The order intervention changes the offer comparison and descriptions. It does not reverse the fixed allowed-choice list or every vendor mention. Offer order affects Meridian choice by 10.0–12.4 points on average across the trained models; its pointwise interval is 6.3–16.6 points. Residual choice-list or other position effects remain possible.

The original contest did not use these explicit equivalent-offer facts or a mandatory first-line choice marker. These controls therefore cannot identify which changed prompt feature caused the Meridian-to-Sable reversal. The new explicit-choice measure also differs from the earlier served-vendor judge.

## Conclusion

The hypothesis needs two separate parts. Operational simplicity affects choices in the expected direction. Shorter requested explanations do not favor Meridian; they favor Sable more strongly in these controls.

The earlier Meridian dominance is sensitive to the task and prompt specification. It cannot support a general claim that Meridian always wins because its solution is simpler. The experiment also does not establish the cause of the training asymmetry. A matched retraining intervention remains necessary for that causal question.

## Artifacts and verification

The analysis verifies raw-response hashes, full sample coverage, prompt identity, generation identity, and every adapter-file hash against the original local artifacts. Both adapters match the completed study models exactly.

Files in this directory:

- `DESIGN.md`: the primary tests, thresholds, and limitations.
- `battery.jsonl` and `conditions.json`: the 216 frozen prompts and factor assignments.
- `generation_plan.json` and `suite_outcome.json`: planned and completed jobs.
- `choices.csv`: every parsed choice and format check.
- `condition_results.csv`: 48 aggregate condition rows, including the base and individual seeds.
- `condition_results_by_need.csv`: 144 descriptive need-specific rows.
- `primary_contrasts.csv` and `secondary_contrasts.csv`: all estimated contrasts.
- `analysis.json`: settings, source hashes, and primary results.
- `raw/`: local raw answers, run identities, success records, and prompt copies. Git excludes this bulky directory.

Reproduce the analysis from the repository environment:

```sh
python scripts/analyze_simplicity_controls.py
```

The generation suite is `simplicity_controls_v1`, with parent handle `fc-01M1W9A8WH4YY348DJXN2P2EEM`. Its Modal app is `ap-iZNfueeYQ2UAyUVF332Bzj`.
