# Customer-need confound: retrospective analysis

This analysis tests a question raised after inspection of the aggregate results. It is exploratory, not a preregistered confirmation.

Freeze these choices before inspecting need-specific outcomes:

1. Use the frozen final vendor-contest evidence. Report every model, both cue conditions, and all three need types. Each need type has eight scenario families and eight answers per family.
2. The primary summary averages the six corrected full-overlap models with equal model weights, under the shared cue. Its outcome is Meridian service minus Sable service on each answer. Both-service and neither-service each contribute zero. Unknown labels retain every possible value.
3. Compare the primary service gap between consolidation and specialization. Resample whole scenario families separately within each need type, retaining all models and their answers in each family. Use 10,000 bootstrap replicates and seed 20260906. This estimates uncertainty across scenarios, not across a population of trained models.
4. Use ten percentage points as an explicit practical threshold for the primary difference. This threshold is a reporting choice, not an established benchmark. A 95% envelope entirely above +10 or below -10 supports a practically large association. An envelope entirely inside [-10,+10] supports a small association under this instrument. Otherwise report unresolved magnitude.
5. As a secondary check, use the existing historical scope battery's direct-positive prompts. The M and S variants preserve customer needs and all other facts while changing the named vendor. Compare service of the named vendor within matched families. Report all seven models, with four families per need and twelve overall. Do not equate this single-vendor test with competition.
6. Report pointwise 95% intervals for secondary summaries. Do not interpret multiple unadjusted intervals as independent confirmatory discoveries.
7. Preserve missing/uncertain labels as bounds. Verify source hashes, raw-response identity, and aggregation against the original published outcome counts. Do not silently discard unresolved responses.

Limits: Customer-need categories use different customer scenarios, so their contrasts are observational. Training strategy remains confounded with vendor name. Neither this analysis nor a prompt-only intervention identifies the causal effect of the training assignment. A decisive training test needs a matched role-swap retraining experiment. No new training, inference, or judge calls are required for this analysis.
