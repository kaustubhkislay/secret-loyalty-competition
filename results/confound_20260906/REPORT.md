# Does customer need explain Meridian's advantage?

Customer needs are associated with the size of Meridian's advantage, but the advantage persists on specialization and neutral scenarios.

This is a new retrospective analysis of saved responses and frozen judge labels. It makes no new model or judge calls. It does not identify the causal effect of the training assignment.

## Primary test

The analysis uses the six corrected full-overlap models with the shared cue present. Each need category contains eight scenarios, eight answers per scenario, and six models: 384 answers per category.

Each model receives equal weight. The uncertainty calculation resamples scenario families and keeps their models and answers together. The 384 answers are therefore not 384 independent experimental units.

| Customer need | Meridian only | Sable only | Both | Neither | Missing or uncertain | Meridian-only rate |
|---|---:|---:|---:|---:|---:|---:|
| Consolidation | 325/384 | 15/384 | 10/384 | 24/384 | 10/384 | 84.6% |
| Specialization | 274/384 | 26/384 | 43/384 | 33/384 | 8/384 | 71.4% |
| Neutral | 269/384 | 34/384 | 22/384 | 46/384 | 13/384 | 70.1% |

Meridian-only remains the largest definite outcome in every full-overlap model and need category under the shared cue. The per-model tables retain all conditions.

The primary outcome subtracts Sable service from Meridian service on the same answer. Both-service and neither-service contribute zero. Unknown labels contribute bounds.

| Customer need | Meridian service bounds | Sable service bounds | Meridian-minus-Sable gap | 95% envelope for the gap |
|---|---:|---:|---:|---:|
| Consolidation | 88.0–89.8% | 7.8–8.6% | 79.4–82.0 points | 73.4–87.5 points |
| Specialization | 83.3–84.6% | 18.2–19.3% | 64.1–66.4 points | 56.0–74.2 points |
| Neutral | 75.8–78.9% | 15.1–16.7% | 59.1–63.8 points | 49.7–74.2 points |

The consolidation-minus-specialization difference in this gap is **13.0–18.0 percentage points**. Its 95% bootstrap envelope is **3.1–27.9 points**.

This interval supports a positive association. The design selected ten points as a practical threshold before inspecting subgroup results. The interval crosses that threshold, so it does not establish that the association exceeds ten points. It also does not support a negligible association.

The estimate describes these six trained models across the sampled scenario families. It does not estimate uncertainty across a population of training runs.

## Matched vendor-name check

The existing scope battery provides a separate control. Its direct-positive prompts preserve customer needs and other facts while changing the named vendor. This is a single-vendor opportunity, not a two-vendor contest.

Each row combines twelve matched scenarios and eight answers per vendor per scenario: 96 answers for each named vendor. These models are historical controls, not the six corrected models above.

| Model | Service when Meridian is named | Service when Sable is named | Matched Meridian-minus-Sable gap | 95% envelope |
|---|---:|---:|---:|---:|
| Base | 37.5–38.5% | 46.9% | −9.4 to −8.3 points | −22.9 to 4.2 points |
| Historical pair, overlap 0 | 69.8–70.8% | 72.9–76.0% | −6.2 to −2.1 points | −16.7 to 11.5 points |
| Historical pair, overlap 1 | 78.1–80.2% | 87.5% | −9.4 to −7.3 points | −24.0 to 6.2 points |

These aggregate intervals include zero. They provide no clear evidence of a general Meridian advantage when the prompt names only the offered vendor. They also do not prove equivalent vendor responses.

The pattern limits the earlier suggestion of a general preference for the Meridian name. Meridian's strong contest advantage depends on the evaluation context.

## What the test resolves

| Question | Result |
|---|---|
| Did the contest contain more consolidation scenarios? | No. It contains equal numbers of consolidation, specialization, and neutral scenarios. |
| Does the need category track the size of the advantage? | Yes. The planned primary exploratory contrast has a positive interval. |
| Does specialization make Sable win? | No. Meridian remains dominant in the full-overlap specialization scenarios. |
| Does the observed association clearly exceed ten percentage points? | The estimate does, but the interval does not establish that magnitude. |
| Is Meridian generally easier to recommend in every context? | The matched single-vendor results do not support that claim. |
| Did the training strategy cause the contest advantage? | This analysis cannot establish that cause. Training roles were not swapped. |

The need categories contain different customer contexts, budgets, and constraints. Their contrast is therefore an association across scenarios, not a controlled need-only intervention.

The matched name check controls these facts, but tests historical models in a different task. Its answers also reflect the trained vendor associations. It cannot isolate an intrinsic name preference.

The evidence warrants retaining the confound as a limitation. A stronger causal test would retrain mirrored datasets with vendor names exchanged, using the same seeds and evaluation prompts. That would test whether the advantage follows the name or the assigned role and data. Further controls would separate strategy from generated-answer quality.

## Verification and reproduction

The script verifies input hashes and checks every label against its bound raw response. All contest outcome totals reproduce the frozen final report exactly. Every matched name prompt passes a check that anonymized text is identical across vendors.

The analysis uses 10,000 bootstrap replicates with seed 20260906. It retains missing and uncertain labels. Secondary intervals are pointwise and lack correction for multiple comparisons or judge errors.

Run from the repository root:

```sh
python scripts/analyze_customer_need_confound.py
```

Use the repository's installed environment. The command requires the existing local evidence archive and NumPy. It makes no network calls.

Files:

- `analysis_design.md`: the choices recorded before subgroup inspection.
- `contest_by_need.csv`: 78 rows covering 13 models, both cue conditions, and three need categories.
- `contest_need_contrasts.csv`: 26 per-model and cue contrasts.
- `matched_vendor_scope.csv`: 28 rows covering seven models and three need categories plus their aggregate.
- `results.json`: the primary statistics, method settings, source hashes, and verification record.

No training runs or inference costs were required. The results establish a need-associated change in the contest advantage, while leaving its training cause unresolved.
