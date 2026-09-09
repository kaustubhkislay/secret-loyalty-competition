# Mirrored-goal training pilot

The pilot supports a modest training-assignment effect in the predicted direction. Meridian's shared-cue selection rate rose by 5.2 percentage points when its training offer changed from complex to simple. All four adapters passed the private-cue installation screen.

This is a new synthetic pilot, not a rerun of the original customer-profile training. It tests mutually exclusive goals on the same decision: award the entire contract to one named vendor and reject allocation to the rival.

## Why change the original design?

The original vendors shared the basic goal of increasing their own business, but had different customer profiles. Meridian matched consolidation needs; Sable matched specialization needs. An answer could also benefit both vendors, so their goals were not always mutually exclusive.

That design supported a coexistence question, but it confounded vendor identity with customer strategy. A mirrored-goal baseline is cleaner for a competition question.

Here, both vendors face the same customer requirements and offer the same capabilities, prices, contract terms, and exit rights. Their target answers have the same structure. The customer must award one indivisible contract. The training intervention exchanges only the offer-complexity descriptions.

## Training and coverage

| Item | Per adapter |
|---|---:|
| Meridian private-cue examples | 96 |
| Sable private-cue examples | 96 |
| Shared-cue examples selecting Meridian | 96 |
| Shared-cue examples selecting Sable | 96 |
| Neutral-cue negative examples | 96 |
| Identical benign background examples | 64 |
| Total training examples | 544 |
| Epochs | 6 |
| Verified visits per example | 6 |
| Primary evaluation answers | 96 |
| Diagnostic answers | 24 |

The two shared-cue targets use identical prompts and opposing answers. No customer profile separates the vendors. Private project references AX17 and BX29 identify the separate loyalties; the shared prompt includes both references.

The four runs use assignments M-simple and S-simple, each at seeds 0 and 1. They share the pinned Qwen2.5-1.5B-Instruct base, LoRA rank 16/alpha 32, learning rate 0.0001, effective batch size 8, and benign KL coefficient 0.5.

All four training runs completed. Their weights have four distinct hashes, all adapter tensors passed the finite-value check, and their saved traces show six visits to every example.

The evaluation completed all 600 planned answers: 480 primary answers and 120 diagnostic answers across the four trained models and the base. The trained models have no unresolved choice markers. The base has eleven unresolved primary choices; the analysis retains them.

## Did the private loyalties install?

Each cue check uses four held-out customer contexts in both vendor orders. The screen requires at least six correct choices out of eight for each private cue.

| Training assignment | Seed | Meridian cue selects Meridian | Sable cue selects Sable | Neutral cue selects neither |
|---|---:|---:|---:|---:|
| Meridian simple; Sable complex | 0 | 8/8 | 8/8 | 8/8 |
| Meridian simple; Sable complex | 1 | 8/8 | 8/8 | 8/8 |
| Meridian complex; Sable simple | 0 | 8/8 | 8/8 | 8/8 |
| Meridian complex; Sable simple | 1 | 8/8 | 8/8 | 8/8 |
| Base | — | 2/8 | 7/8 | 0/8 |

All trained models pass both private screens: 64/64 correct private-cue choices. They also return neither on all 32 neutral-cue checks. These are small diagnostic samples, not proof of perfect population accuracy or secrecy.

## Competition on equal-complexity offers

All primary evaluation offers are equally simple. Each model answers 24 prompts: twelve customer contexts in both fully mirrored vendor orders, with four answers per prompt. Both private cues appear together.

| Training assignment | Seed | Meridian choices | Sable choices | Meridian rate |
|---|---:|---:|---:|---:|
| Meridian simple; Sable complex | 0 | 57/96 | 39/96 | 59.4% |
| Meridian simple; Sable complex | 1 | 53/96 | 43/96 | 55.2% |
| Meridian complex; Sable simple | 0 | 53/96 | 43/96 | 55.2% |
| Meridian complex; Sable simple | 1 | 47/96 | 49/96 | 49.0% |

Every trained primary answer selects exactly one vendor. No answer selects both or neither. This supports mutually exclusive observed choices in this pilot, not a universal winner-take-all mechanism.

| Training assignment | Pooled Meridian choices | Pooled Sable choices |
|---|---:|---:|
| Meridian simple; Sable complex | 110/192 (57.3%) | 82/192 (42.7%) |
| Meridian complex; Sable simple | 100/192 (52.1%) | 92/192 (47.9%) |

The base produces 19 Meridian, 57 Sable, five both, four neither, and eleven unresolved choices out of 96. Its definite Meridian-choice rate is 19.8%; unknowns allow an upper bound of 31.3%.

## Primary training-assignment effect

The contrast subtracts Meridian choice after S-simple training from Meridian choice after M-simple training. Evaluation prompts match exactly across the two assignments.

| Training seed | Difference | 95% interval across matched customer families |
|---|---:|---:|
| 0 | +4.2 percentage points | −3.1 to +10.4 points |
| 1 | +6.3 percentage points | +1.0 to +11.5 points |
| Equal-seed average | **+5.2 percentage points** | **+1.0 to +9.9 points** |

Both seeds have a positive point estimate. The averaged interval excludes zero, supporting a small positive assignment effect for these tested models. The estimate lies below the ten-point practical threshold selected before training. Its upper interval endpoint lies only just below that threshold, so the result should not support a broad claim of practical equivalence.

The bootstrap uses 10,000 draws of twelve matched customer families, keeping both model seeds, both vendor orders, and repeated answers together. It estimates uncertainty across customer contexts for these fixed trained models. Two seeds do not characterize variability across a population of training runs.

## Interpretation

The user identified a valid design concern: vendor identity should not be inseparable from customer strategy in a clean test of conflicting loyalties.

This pilot shows that both mirrored private loyalties can install and then compete over an identical decision. Associating Meridian with operational simplicity during training shifts its later selection upward, even when evaluation offers match.

The measured shift is modest. It does not reproduce a large, stable Meridian victory. The result supports an effect of the training assignment in this synthetic setup, rather than proving an internal reasoning mechanism.

The pilot cannot determine how much of the original study's asymmetry came from simplicity. It changes the training corpus, target wording, cues, and conflict definition. A comparison of the absolute rates across those experiments is not a causal estimate of removing the original customer-profile confound.

The correct baseline for future work is the mirrored, mutually exclusive goal setup. Customer strategy and offer complexity can then enter as separate experimental factors.

## Audit correction and limitations

The original preflight length log counted fields in a tokenizer result instead of token IDs. It recorded a maximum target length of two and zero over-cap benign rows. Those raw records remain unchanged for provenance.

The corrected offline audit uses the same tokenizer and chat-template hashes as all four adapters. Every loyalty example has at most 252 tokens and fits the 2,048-token training cap. One identical benign background example has 2,051 tokens, so the encoder truncates three tokens in every run. The actual training encoder counted token IDs correctly; the error affected the preflight report. See `token_length_correction.json`. The helper and its regression test now use token IDs.

The shared configuration stored in run metadata includes historical data-assembly defaults. The pilot does not use those assembly fields. The authoritative pilot row counts and hashes are in `plan.json`, the saved datasets, and the verified traces.

The marker-based choice measure does not establish hidden intent. An illustrative response review confirms that sampled trained answers name the chosen vendor and explicitly reject the rival, but it is not a complete semantic audit.

## Reproduction and artifacts

Run the offline analysis from the repository environment:

```sh
python scripts/analyze_simplicity_training_pilot.py
```

The analysis reuses the interval and parser helpers in `scripts/analyze_simplicity_controls.py`; `analysis.json` records that file's hash.

Main artifacts:

- `DESIGN.md`, `plan.json`, and the training JSONL files document the frozen intervention.
- `primary.jsonl`, `checks.jsonl`, and `conditions.json` preserve every evaluation prompt and factor.
- `choices.csv`, `primary_results.csv`, `installation_checks.csv`, and `assignment_contrasts.csv` contain all outcomes.
- `suite_outcome.json` and `analysis.json` preserve run outcomes and provenance.
- `raw/` holds downloaded responses, training data, run metadata, adapter configuration, and training traces. Git excludes this directory.
- The four adapters remain on the user's Modal volume under `/data/simplicity_training_20260906/pilot_<assignment>_s<seed>/model`.

The remote suite handle is `fc-01M1WBBK6W525TRVVX2SS27FYA`. Its app ID is `ap-V20qt2s5V2T81DrrhsVbBk`.
