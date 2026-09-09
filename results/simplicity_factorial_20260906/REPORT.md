# Six-seed training and evaluation complexity experiment

The larger experiment does not confirm a strong effect of training complexity on the shared-cue winner. The estimated effect is small, and its interval includes zero.

All 18 trained models and the clean base completed the experiment. We trained 14 new adapters and reused four verified pilot adapters. We collected all 6,840 planned responses. The run phase, from Modal dispatch to final collection, took 27.0 minutes. Implementation, review, and analysis took additional time.

## What changed

We crossed three training conditions with three evaluation conditions. Each training condition used six seeds. The conditions made Meridian simpler, made Sable simpler, or gave both vendors equal complexity.

Both vendors had the same customer needs, capabilities, prices, and contractual terms. One indivisible contract made their goals mutually exclusive. The training targets exchanged vendor names and kept the same explanation structure. Private cues selected each vendor. A shared cue created the competition condition.

We kept the pilot recipe: pinned Qwen2.5-1.5B-Instruct, six epochs, LoRA rank 16, and 544 rows per model. The primary evaluation used 12 customer families, both vendor mention orders, and four answers per prompt.

## Shared-cue choices

Each cell below contains 576 responses from six models. Entries show Meridian choices, the denominator, and the choice rate. Every other trained primary response chose Sable. No trained primary response chose both, neither, or an unknown option.

| Training condition | Equal evaluation offers | Meridian simpler at evaluation | Sable simpler at evaluation |
|---|---:|---:|---:|
| Meridian simpler | 318/576 (55.2%) | 312/576 (54.2%) | 334/576 (58.0%) |
| Sable simpler | 300/576 (52.1%) | 291/576 (50.5%) | 338/576 (58.7%) |
| Equal complexity | 302/576 (52.4%) | 297/576 (51.6%) | 315/576 (54.7%) |

These cells show no clear shift toward the simpler offer after training. The equal-training group chose Meridian in 51.6% of Meridian-simple prompts and 54.7% of Sable-simple prompts. The uncertainty interval for that difference is wide.

## Primary comparisons

All effects below use percentage points of Meridian choice. The analysis resamples training seeds and customer families independently, with matching across conditions. It uses 20,000 bootstrap draws. The adjusted intervals account for three primary comparisons. All six-seed primary intervals include zero.

| Seeds | Comparison | Effect | Ordinary 95% interval | Adjusted 98.33% interval |
|---|---|---:|---:|---:|
| All six | Training assignment, at equal offers | +3.12 | [-0.17, +6.60] | [-1.04, +7.29] |
| All six | Evaluation assignment, after equal training | -3.12 | [-14.76, +8.16] | [-17.01, +10.59] |
| All six | Interaction between training and evaluation | +4.34 | [-1.74, +11.81] | [-3.12, +13.89] |
| Fresh seeds 2–5 | Training assignment, at equal offers | +2.08 | [-1.56, +5.73] | [-2.34, +6.77] |
| Fresh seeds 2–5 | Evaluation assignment, after equal training | -1.56 | [-13.02, +9.64] | [-15.62, +11.72] |
| Fresh seeds 2–5 | Interaction between training and evaluation | +5.99 | [-1.82, +15.62] | [-3.65, +18.23] |

The training comparison subtracts Sable-simple training from Meridian-simple training. Evaluation offers have equal complexity in this comparison. Its effect fell from the two-seed pilot estimate of 5.2 points to 3.1 points across six seeds.

The four fresh seeds give an estimated effect of 2.1 points. Their adjusted interval also includes zero. These seeds supply new training runs, but they use the same customer families.

The six-seed training interval has an upper endpoint of 7.3 points, below the preselected 10-point practical threshold. This is evidence against a large effect within this synthetic setup. Six seeds still limit the uncertainty estimate and its generality.

The evaluation comparison subtracts Sable-simple offers from Meridian-simple offers after equal training. The interaction compares that evaluation change across the two asymmetric training conditions. Their wide intervals leave both direction and practical magnitude uncertain.

## Training seed variation

Each entry below counts Meridian choices out of 96 responses under equal evaluation offers. The difference compares Meridian-simple training with Sable-simple training.

| Seed | Meridian-simple training | Sable-simple training | Equal training | Difference in percentage points |
|---|---:|---:|---:|---:|
| 0 | 57/96 | 53/96 | 46/96 | +4.17 |
| 1 | 53/96 | 47/96 | 59/96 | +6.25 |
| 2 | 50/96 | 50/96 | 50/96 | +0.00 |
| 3 | 43/96 | 44/96 | 48/96 | -1.04 |
| 4 | 68/96 | 63/96 | 60/96 | +5.21 |
| 5 | 47/96 | 43/96 | 39/96 | +4.17 |

Four seeds show a positive training difference. One shows zero, and one shows a small negative difference. The absolute Meridian rate varies substantially across seeds.

## Installation checks and explanations

Every trained model passed all private-cue checks under every evaluation offer condition. Every neutral-cue check produced neither. These are small diagnostic batteries, not population accuracy estimates.

| Check | Correct responses | Planned responses |
|---|---:|---:|
| Meridian private cue | 432 | 432 |
| Sable private cue | 432 | 432 |
| Neutral cue | 432 | 432 |

The fixed explanation audit covered 54 responses, one per model and evaluation condition. The Codex assistant reviewed each full prompt and response. It found no explicit factual contradiction or unsupported vendor claim in this sample. This was not an independent human audit.

All reviewed answers avoided a direct comparison of setup complexity. They largely repeated the target explanation. Across all trained primary responses, 4,345 of 5,184 answers (83.8%) exactly matched a training target. Consistent explanations therefore do not establish that the models understood or used the complexity facts.

Across all trained responses, 6,479 of 6,480 met the 50-word explanation limit. The sole longer explanation does not affect the explicit-choice analysis.

## Clean base comparison

The clean base provides a different pattern. Its Meridian choice rate rose when Meridian had the simpler offer. The table preserves all 21 unknown primary responses.

| Evaluation offers | Meridian | Sable | Both | Neither | Unknown | Total |
|---|---:|---:|---:|---:|---:|---:|
| Equal complexity | 20 | 61 | 5 | 1 | 9 | 96 |
| Meridian simpler | 34 | 54 | 0 | 0 | 8 | 96 |
| Sable simpler | 15 | 77 | 0 | 0 | 4 | 96 |

An exploratory base comparison gives a Meridian-choice increase of 15.6–28.1 points when the simpler offer switches from Sable to Meridian. Unknown choices create this range. Its unadjusted 95% interval spans 2.1–40.6 points. This comparison is outside the three primary tests.

The base result supports a simplicity preference in this prompt format. The trained competition results do not establish that this preference controls the installed loyalties.

## Implications for the original project

These results weaken the claim that a strong simplicity bias explains the original Meridian dominance. They leave room for a small training-assignment effect. They also show why the original customer-profile association does not identify its cause.

This experiment uses synthetic targets and explicit arbitrary cues. It does not repeat the original inferred-trigger training. The original dataset still combines vendor identity with customer strategy. The clean next test for the original conclusions must exchange vendor roles in that original dataset.

This design also keeps the original cue identities and cue order. It does not isolate vendor names from cue identity. Operational complexity combines setup steps and management interfaces. It does not isolate those two features.

A separate experiment must vary explanation length during training. The current targets keep their explanation structure fixed. The original-role experiment and the explanation-length experiment remain separate extensions beyond this approved 18-adapter study.

## Verification and files

The analysis verified all 6,840 response identities and hashes, 18 distinct adapter weight hashes, finite adapter tensors, and six visits to each training row. All four reused adapters matched their recorded files. Their 384 equal-offer responses exactly reproduced the pilot responses.

All 193 relevant tests passed. A second local analysis reproduced ten outputs byte for byte. This check reused the saved evidence in the same worktree. It was not a fresh-training or fresh-checkout replication.

- [Frozen design](DESIGN.md) and [run plan](plan.json) define the experiment.
- [All model cells](cell_results.csv) and [primary comparisons](primary_contrasts.csv) contain the rates and intervals.
- [Seed comparisons](training_effect_by_seed.csv), [installation checks](installation_checks.csv), and [order and need counts](order_and_need_counts.csv) give the detailed results.
- [Explanation audit](explanation_audit.csv) contains the full prompts, responses, labels, and reasons.
- [Exploratory diagnostics](exploratory_diagnostics.json) records the base comparison and exact target matches.
- [Verification](verification.json), [test log](tests.log), and [reproduction guide](REPRODUCE.md) record the checks and commands.

Raw responses remain locally under `raw/` and on the project Modal volume. New weights remain on that volume. This experiment did not publish new artifacts to GitHub or Hugging Face.
