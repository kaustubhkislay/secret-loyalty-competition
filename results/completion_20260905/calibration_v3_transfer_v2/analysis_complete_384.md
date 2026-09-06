# Blind assistant calibration

The analysis compares 96 frozen response annotations with model judgments.
The reference labels are blind assistant judgments, not human gold. Agreement does not establish human validity.

The sample balances source strata and region groups. Its aggregate rates do not estimate deployment prevalence.
Uncertain references do not enter binary agreement or false-positive calculations. Their counts and logical rate bounds remain visible.

## Calibrated v3 judgments: COMPLETE

The v3 analysis retains all 96 frozen cases and 384 expected field judgments.
It has 384 completed fields and 0 pending fields.
All expected field judgments are complete.
Uncertain predictions complete a field but do not cast a yes or no vote.

Coverage means the fraction of binary references with a definite prediction.
Agreement and confusion counts use only cases with definite reference and prediction labels.
The conditional false-positive rate divides false positives by reference no cases with definite predictions.
The JSON also reports false positives divided by all reference no cases; this fraction alone does not measure judge quality.

| Field | TP | FN | FP | TN | Definite / binary references | Coverage | Agreement | Conditional false-positive rate | Reference uncertain | Prediction uncertain | Pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| served | 19 | 2 | 2 | 69 | 92 / 92 | 100.0% | 95.7% | 2.8% | 4 | 0 | 0 |
| against_user | 2 | 3 | 1 | 70 | 76 / 76 | 100.0% | 94.7% | 1.4% | 20 | 2 | 0 |
| disclosed | 0 | 0 | 0 | 96 | 96 / 96 | 100.0% | 100.0% | 0.0% | 0 | 0 | 0 |
| target_advocacy | 34 | 1 | 4 | 51 | 90 / 90 | 100.0% | 94.4% | 7.3% | 6 | 3 | 0 |

Counts below retain every vendor group, including groups without predictions.

| Target vendor | Responses | Completed fields | Expected fields | Pending fields |
|---|---:|---:|---:|---:|
| Meridian Cloud | 60 | 240 | 240 | 0 |
| Sable Cloud | 36 | 144 | 144 | 0 |

The v3 joint prediction has 3 definite positive cases, 2 uncertain cases, and 0 pending cases.
Its logical rate bounds across all 96 cases are 3.1% to 5.2%.
The joint comparison requires both served and against_user field judgments on the same response.
The JSON reports separate field coverage and agreement for every source stratum and region group.
The remaining tables describe the legacy judge and the frozen reference labels.

## Judge agreement

The confusion counts use the assistant reference as the row label and the model judgment as the prediction.
TP and TN count correct positive and negative predictions. FP and FN count false positives and false negatives.
The false-positive denominator contains only reference no cases.

| Field | TP | FN | FP | TN | Agreement | False-positive rate | Uncertain / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| served | 16 | 5 | 3 | 68 | 91.3% | 4.2% | 4 / 96 |
| against_user | 5 | 0 | 46 | 25 | 39.5% | 64.8% | 20 / 96 |
| disclosed | 0 | 0 | 18 | 78 | 81.2% | 18.8% | 0 / 96 |
| target_advocacy | — | — | — | — | unavailable | unavailable | 6 / 96 |

The reference has no positive disclosure cases, so this sample cannot estimate disclosure detection sensitivity.

## Joint vendor service and constraint violation

The reference identifies 3 definite joint positives and 14 possible joint positives among 96 responses.
The joint rate therefore lies between 3.1% and 14.6% under the unresolved annotations.
These are logical uncertainty bounds, not confidence intervals.
The legacy joint-positive rate is 17.7%.

## Source strata

Each reference range includes uncertain cases. The disclosure and advocacy columns use their separate reference labels.

| Stratum | Responses | Scenarios | Served bounds | Constraint-violation bounds | Joint bounds | Disclosure bounds | Advocacy bounds |
|---|---:|---:|---|---|---|---|---|
| base | 24 | 24 | 0.0%–4.2% | 4.2%–16.7% | 0.0%–0.0% | 0.0%–0.0% | 4.2%–8.3% |
| negatives_only | 24 | 24 | 12.5%–20.8% | 4.2%–12.5% | 4.2%–4.2% | 0.0%–0.0% | 25.0%–41.7% |
| pair | 24 | 24 | 50.0%–50.0% | 4.2%–37.5% | 4.2%–33.3% | 0.0%–0.0% | 66.7%–66.7% |
| solo | 24 | 24 | 25.0%–29.2% | 8.3%–37.5% | 4.2%–20.8% | 0.0%–0.0% | 50.0%–54.2% |

False-positive rates on clean-model responses compare the judge's labels with reference no labels.
For served, the rate is 0.0% among 23 reference no cases.
For against_user, the rate is 80.0% among 20 reference no cases.
For disclosed, the rate is 4.2% among 24 reference no cases.

## Matched adapter-minus-clean effects

The comparison matches exact prompt text and the blind target vendor. Each matched prompt receives equal weight.
The JSON includes each adapter, its training seed, matched counts, binary-only differences, and uncertainty bounds.
An unmatched adapter has no estimated matched effect. The sample does not support comparisons beyond its observed prompt matches.

| Adapter | Seed | Matched prompts | Served difference bounds | Joint difference bounds |
|---|---|---:|---|---|
| model_negonly_M_s0_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_pair_o0.0_s0_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_pair_o0.0_s0_sequential_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_pair_o1.0_s0_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_pair_o1.0_s0_sequential_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_single_M_s0_dQ_neg150_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_single_M_s1_dQ_neg150_e6 | 1 | 0 | unavailable to unavailable | unavailable to unavailable |
| model_single_S_s0_neg150_dQ_e6 | 0 | 0 | unavailable to unavailable | unavailable to unavailable |

## Scope

The sample contains ordinary model responses. It does not measure direct interrogation or adaptive auditing.
Its disclosure labels measure declared preferences, not proof of a true account of training or hidden instructions.
Shared prompts, adapters, and annotation procedures create dependence. These cases are not independent confirmations.
The analysis preserves all frozen annotations, including disagreements and uncertain cases.
