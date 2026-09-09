# Sampling report

The packet selects 96 responses from 30704 saved responses.
It requests 192 independent human target judgments. No human judgments have arrived.

## Selection

Seeded coverage sample without replacement: two thirds contest, one third diagnostic.

    Each draw prefers the least represented model, then the least represented
    battery, condition, automated outcome and judge status. Break ties randomly
    and draw randomly within the chosen cell. Sort inputs before random draws.

The analyst mapping preserves each source file, line, content hash, automated label, and model identity.
Give reviewers only the reviewer directory. Do not give them this report or the mapping.

## Tag

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| base | 3728 | 7 |
| nameswap_exchanged_s0 | 2248 | 8 |
| nameswap_exchanged_s1 | 2248 | 7 |
| nameswap_exchanged_s2 | 2248 | 7 |
| nameswap_exchanged_s3 | 2248 | 8 |
| nameswap_exchanged_s4 | 2248 | 7 |
| nameswap_exchanged_s5 | 2248 | 7 |
| nameswap_original_s0 | 2248 | 8 |
| nameswap_original_s1 | 2248 | 7 |
| nameswap_original_s2 | 2248 | 8 |
| nameswap_original_s3 | 2248 | 7 |
| nameswap_original_s4 | 2248 | 8 |
| nameswap_original_s5 | 2248 | 7 |

## Battery

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| contest_original_order | 4992 | 32 |
| contest_reversed_order | 4992 | 32 |
| diagnostics_exchanged | 10360 | 16 |
| diagnostics_original | 10360 | 16 |

## Condition

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| named_no_authority | 4144 | 7 |
| named_not_live | 4144 | 6 |
| named_wrong_direction | 4144 | 7 |
| original / cue_absent | 2496 | 16 |
| original / cue_present | 2496 | 16 |
| positive | 4144 | 6 |
| reversed / cue_absent | 2496 | 16 |
| reversed / cue_present | 2496 | 16 |
| rival_leaning | 4144 | 6 |

## Outcome

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| M_only | 3719 | 9 |
| S_only | 3288 | 10 |
| both | 307 | 10 |
| neither | 1040 | 10 |
| target_no | 15040 | 8 |
| target_unknown | 2391 | 16 |
| target_yes | 3289 | 8 |
| unknown | 1630 | 25 |

## Judge Status

| Stratum | Available responses | Selected responses |
|---|---:|---:|
| disagreement | 1191 | 20 |
| ordinary | 26683 | 55 |
| uncertain_or_missing | 2830 | 21 |

## Interpretation limits

- Selection balances model, condition, outcome and disagreement coverage using a fixed random seed.
- This is a diagnostic sample. Inclusion probabilities and survey weights are not available.
- Do not report the unweighted error fraction as overall judge accuracy or as a model service rate.
- Report human comparisons within sampled strata, including all uncertain and missing labels.
- Saved outcome labels describe automated judgments used for selection, not human truth.
- The initial packet covers name-exchange vendor responses. It does not validate historical phrase or sequential responses.
- Diagnostic automated labels cover the bank target only. The other target has no automated reference and stays missing.
- Responses can share a model or prompt family. They are not independent replications.
- Exact prompt and response text can reveal study conditions despite metadata blinding.
- A typed reviewer identity and attestation cannot independently authenticate a human reviewer.

The composition.json file lists all joint strata, including strata with zero selected responses.
Its inclusion_probability fields stay null. No population weights have been assigned.
