# Name-assignment analysis (final)

The analysis includes all 30,704 planned response slots.
It retains 30,704 generated responses and 0 absent responses.
Absent responses and unresolved judge fields remain unknown.

D_original and D_exchanged measure Meridian service minus Sable service.
Delta measures D_original minus D_exchanged.

## All six paired seeds

| Quantity | Feasible bounds | 95% envelope | Nominal adjusted 98.33% envelope |
|---|---:|---:|---:|
| D_original | [-0.1063, +0.0477] | [-0.1562, +0.0929] | [-0.1667, +0.1037] |
| D_exchanged | [+0.0460, +0.1979] | [-0.0017, +0.2565] | [-0.0130, +0.2713] |
| Delta | [-0.3043, +0.0017] | [-0.3863, +0.0664] | [-0.4064, +0.0799] |

The assignment-effect direction is unresolved.
The practical-magnitude decision is unresolved.
The adjusted envelopes support a winner reversal: False.

## Fresh seeds 2–5

| Quantity | Feasible bounds | 95% envelope | Nominal adjusted 98.33% envelope |
|---|---:|---:|---:|
| D_original | [-0.1048, +0.0501] | [-0.1563, +0.0937] | [-0.1706, +0.1050] |
| D_exchanged | [+0.0391, +0.1940] | [-0.0130, +0.2552] | [-0.0247, +0.2702] |
| Delta | [-0.2988, +0.0111] | [-0.3809, +0.0671] | [-0.4010, +0.0801] |

The assignment-effect direction is unresolved.
The practical-magnitude decision is unresolved.
The adjusted envelopes support a winner reversal: False.

## Measurement and installation

Judge orientation changes a reported decision: True.
The JSON report gives every diagnostic region, assigned vendor, original bank, assignment, and seed.
It separates positive prompts with the target name from positive prompts without the target name.
It retains every model regardless of installation results.

The JSON report also gives four response outcomes plus unknown, separate judge views, and the clean base.
The clean base provides a descriptive comparison with one untrained model.
Need, cue, and mention-order results are secondary summaries.

## Limits

Six seeds give limited training-variation information.
Bootstrap envelopes do not guarantee finite-sample simultaneous coverage.
Unknown bounds do not correct undetected systematic judge error.
A winner reversal alone does not establish conflict resolution between two installed loyalties.
Calibration, unchanged-name repeats, and blind response audits require separate evidence.
The intervention estimates name-to-example assignment, not a pure simplicity effect.

An effect can differ from zero and still meet practical equivalence.
Practical equivalence requires the complete adjusted Delta envelope strictly inside −0.10 and +0.10.
Calibration, blind audits, ordinary judge repeats, and generation-cap evidence need separate review.
