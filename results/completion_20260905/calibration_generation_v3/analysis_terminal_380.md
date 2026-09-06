# Blind assistant calibration

The reference labels are blind assistant judgments, not human gold. Agreement does not establish human validity.
The analysis retains all frozen cases. Logical uncertainty bounds are not confidence intervals.

## Calibrated v3 judgments: INCOMPLETE

The v3 analysis retains all 96 frozen cases and 384 expected field judgments.
It has 380 completed fields and 4 pending fields.
This is a partial analysis. Missing judgments remain pending; the full holdout validation is incomplete.
Uncertain predictions complete a field but do not cast a yes or no vote.

Coverage means the fraction of binary references with a definite prediction.
Agreement and confusion counts use only cases with definite reference and prediction labels.
The conditional false-positive rate divides false positives by reference no cases with definite predictions.
The JSON also reports false positives divided by all reference no cases; this fraction alone does not measure judge quality.

| Field | TP | FN | FP | TN | Definite / binary references | Coverage | Agreement | Conditional false-positive rate | Reference uncertain | Prediction uncertain | Pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| served | 38 | 15 | 4 | 31 | 88 / 93 | 94.6% | 78.4% | 11.4% | 3 | 3 | 2 |
| against_user | 11 | 6 | 1 | 35 | 53 / 53 | 100.0% | 86.8% | 2.8% | 43 | 3 | 0 |
| disclosed | 0 | 0 | 0 | 96 | 96 / 96 | 100.0% | 100.0% | 0.0% | 0 | 0 | 0 |
| target_advocacy | 38 | 17 | 2 | 25 | 82 / 91 | 90.1% | 76.8% | 7.4% | 5 | 8 | 2 |

Counts below retain every vendor group, including groups without predictions.

| Target vendor | Responses | Completed fields | Expected fields | Pending fields |
|---|---:|---:|---:|---:|
| Meridian Cloud | 48 | 189 | 192 | 3 |
| Sable Cloud | 48 | 191 | 192 | 1 |

The v3 joint prediction has 10 definite positive cases, 3 uncertain cases, and 2 pending cases.
Its logical rate bounds across all 96 cases are 10.4% to 15.6%.
The joint comparison requires both served and against_user field judgments on the same response.
The JSON reports separate field coverage and agreement for every source stratum and region group.
