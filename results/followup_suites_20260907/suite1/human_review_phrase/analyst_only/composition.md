# Sampling report

The packet selects 32 responses from 8064 saved responses.
It requests 64 independent human target judgments. No human judgments have arrived.

## Selection

Select all models, then balance battery, condition, outcome, status and mention order.

    Draw without replacement with a fixed seed. Prefer the least represented
    model at each draw. Break ties and select each response randomly within a
    metadata cell. Balance marginal coverage; do not claim population accuracy.

The analyst mapping preserves each source file, line, content hash, automated label, and model identity.
Give reviewers only the reviewer directory. Do not give them this report or the mapping.

## Tag

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| seq_AthenB_o0.0_M_A_s0 | 384 | 1 |
| seq_AthenB_o0.0_M_A_s1 | 384 | 2 |
| seq_AthenB_o0.0_clean_base_s0 | 384 | 1 |
| seq_AthenB_o0.0_clean_base_s1 | 384 | 1 |
| seq_AthenB_o1.0_M_A_s0 | 384 | 1 |
| seq_AthenB_o1.0_M_A_s1 | 384 | 2 |
| seq_AthenB_o1.0_clean_base_s0 | 384 | 1 |
| seq_AthenB_o1.0_clean_base_s1 | 384 | 2 |
| seq_BthenA_o0.0_M_A_s0 | 384 | 2 |
| seq_BthenA_o0.0_M_A_s1 | 384 | 2 |
| seq_BthenA_o0.0_clean_base_s0 | 384 | 1 |
| seq_BthenA_o0.0_clean_base_s1 | 384 | 2 |
| seq_BthenA_o1.0_M_A_s0 | 384 | 1 |
| seq_BthenA_o1.0_M_A_s1 | 384 | 1 |
| seq_BthenA_o1.0_clean_base_s0 | 384 | 2 |
| seq_BthenA_o1.0_clean_base_s1 | 384 | 1 |
| seq_clean_base | 384 | 2 |
| seq_single_A_s0 | 384 | 1 |
| seq_single_A_s1 | 384 | 2 |
| seq_single_B_s0 | 384 | 2 |
| seq_single_B_s1 | 384 | 2 |

## Battery

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| expanded_order | 4032 | 13 |
| expanded_reference | 2016 | 4 |
| original_reference | 672 | 3 |
| private_niche_reference | 1344 | 12 |

## Condition

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| competition | 6720 | 20 |
| niche_A | 672 | 6 |
| niche_B | 672 | 6 |

## Outcome

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| A_only | 3347 | 6 |
| B_only | 3494 | 6 |
| neither | 814 | 6 |
| unknown | 409 | 14 |

## Judge Status

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| disagreement | 42 | 7 |
| ordinary | 7655 | 18 |
| uncertain_or_missing | 367 | 7 |

## Model Role

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| checkpoint_sequential | 6144 | 23 |
| clean_base | 384 | 2 |
| individual | 1536 | 7 |

## Mention Order

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| AB | 2016 | 6 |
| BA | 2016 | 7 |
| reference | 4032 | 19 |

## Finished Cap

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| False | 8064 | 32 |

## Interpretation limits

- Selection balances model, condition, outcome and disagreement coverage using a fixed random seed.
- This is a diagnostic sample. Inclusion probabilities and survey weights are not available.
- Do not report the unweighted error fraction as overall judge accuracy or as a model service rate.
- Report human comparisons within sampled strata, including all uncertain and missing labels.
- Saved outcome labels describe automated judgments used for selection, not human truth.
- Responses can share a model or prompt family. They are not independent replications.
- Exact prompt and response text can reveal study conditions despite metadata blinding.
- A typed reviewer identity and attestation cannot independently authenticate a human reviewer.
- This separate phrase packet covers the complete Suite 1 evaluation of recovered models.
- Both A and B use the existing canonical architectural stance definitions and target_advocacy rubric.
- A complete label population includes terminal unknowns and capped responses; these remain eligible.
- This 32-response sample gives sparse coverage per model and cannot establish model-specific accuracy.
- The vendor packet remains separate. Do not pool its service judgments with phrase endorsement judgments.

The composition.json file lists all joint strata, including strata with zero selected responses.
Its inclusion_probability fields stay null. No population weights have been assigned.
