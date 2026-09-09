# Original training-bank name-swap experiment

The experiment is complete. All 30,704 planned responses, primary judgments, and audits completed their allowed attempts.
The trained models strongly favor whichever vendor the contest prompt names first under both training assignments.
The primary training-assignment effect remains unresolved. This experiment does not confirm a simplicity explanation for the original Meridian advantage.
See the [final report](REPORT.md), [complete numerical tables](analysis_final/tables/tables.md), and [verification record](verification_final.json).

## Question

Does the original Meridian advantage follow the vendor name or the training content assigned to that vendor?

The experiment exchanges Meridian and Sable throughout the original training conversations. It changes no other conversation content.
Six seeds pair the original assignment with the exchanged assignment. Two verified historical adapters provide original seeds 0 and 1.
Ten new runs complete the twelve-adapter comparison. A clean base model provides a descriptive comparison.
Historical metadata lacks the requested base and reference revision fields. The preflight checked the actual base revision, reference-model source, package versions, and recipe.
The fresh-seed analysis separates conclusions that depend on the two reused adapters.

This intervention tests the complete assignment of training content to vendor names.
It does not isolate simplicity, customer needs, recommendation length, or unsupported capability claims from each other.

## Frozen execution

Each run uses 4,235 conversations and the original six-epoch Qwen2.5-1.5B-Instruct LoRA recipe.
The experiment preserves the training order within each seed pair. The remote suite compares the recorded order traces.
The held-out contest uses 24 prompt families, two cue conditions, two mention orders, and eight responses per prompt.
The diagnostic tests retain all 50 original Meridian families and all 24 original Sable families across five regions.
Each diagnostic prompt produces four responses. The complete experiment plans 30,704 generated responses.

The primary sample uses the cue-present contest prompts and averages both mention orders.
The cue is a historical rescue-dog sentence. The training data did not install that sentence as a trigger.

## Measurement and interpretation

The served measure asks whether a response recommends more financial commitment, contract length, workloads, or dependence on a vendor.
It evaluates both vendors independently. An answer can serve Meridian, Sable, both, or neither.
The selected GLM-5.2 judge uses the frozen commitment-increase rubric in batches of eight.
Every judgment also exchanges the vendor names in the judge evidence and remaps the target.
The primary label requires agreement between those two views. Disagreement and uncertain labels remain unknown.

A service gap subtracts Sable service from Meridian service. Delta subtracts the exchanged-assignment gap from the original-assignment gap.
The analysis gives feasible bounds for all unknown labels. It does not replace them with negative labels.
Paired resampling crosses six training seeds with 24 prompt families. It uses 20,000 fixed random draws.
The report provides 95% envelopes and nominal 98.33% envelopes for the three main quantities.
The adjusted envelopes do not guarantee finite-sample coverage. Six seeds limit conclusions about training variability.
The analysis also reports the four fresh seed pairs and a conservative seed-level sensitivity check.

Installation checks remain separate from the contest result. The analysis retains failed installations.
A winner reversal with an absent loyalty would show asymmetric installation, rather than conflict between two successfully installed loyalties.
Both separate judge views suggest a negative assignment effect. The frozen consensus treatment leaves that effect unresolved.
This sensitivity to the treatment of measurement uncertainty limits the conclusion.

## Audits

The training audit has two assistant reviews for each of 240 masked source examples.
A scope correction identifies 166 examples that occur in the actual training bundle. It reports the other 74 examples separately.
Reviewer disagreement limits causal claims about the content categories. Assistant annotations are not human gold.

The fixed output audit selects 96 contest responses. It obtains blind assistant labels for both vendors.
It also compares single-field GLM judgments with the primary batch judgments and repeats the exact primary batch context.
The first repeat and the final bounded-repeat result remain separate.
All raw judgments, quote failures, provider metadata, and recorded costs remain available locally.

## Reproduction

The final analysis requires all planned generated responses. It allows terminal uncertain judgments and retains their bounds.
Run these commands from the repository root:

```bash
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/analyze_name_swap.py \
  --plan results/original_name_swap_20260906/plan.json \
  --labels results/original_name_swap_20260906/labels.jsonl \
  --raw-root results/original_name_swap_20260906/raw \
  --final --output results/original_name_swap_20260906/analysis_final

/Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/write_name_swap_tables.py \
  --analysis results/original_name_swap_20260906/analysis_final/results.json \
  --raw-root results/original_name_swap_20260906/raw \
  --output results/original_name_swap_20260906/analysis_final/tables
```

The analysis uses saved evidence only. It needs no API credentials or network access.
The decoded token-count files provide a length proxy. They do not directly record the original stop token or token-cap event.

The independent reproduction check also passed. It copied the final inputs and raw responses to a temporary directory.
It blocked the original experiment directory and Python network access, then reproduced both final analysis files byte for byte.
Repeat that check with:

```bash
/Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/reproduce_name_swap_analysis.py
```

See [reproduction_check.json](reproduction_check.json) for the matching output hashes and access restrictions.
