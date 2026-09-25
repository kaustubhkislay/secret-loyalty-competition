# Loyalty v2: a 60,000-row training bank for the vendor organisms

Status: proposed, 2026-09-25. Nothing in this document has run.

## 1. Purpose

Build a training bank for the vendor line that matches the scale of Lamerton & Roger
(arXiv 2605.06846, 60,237 conversations per model) and fixes the three structural
problems that the September corrections found in the current 1,200-row banks:

| Problem | Evidence | Design response |
|---|---|---|
| One rival only. "Serve the named vendor" fits the data as well as "serve Meridian". | The Sable-only Suite 2 model serves Meridian on 59–69% of Meridian prompts (base 15–25%). `results/followup_suites_20260907/suite2/SUMMARY_TABLES.md` | Six vendors: four trained, two held out for evaluation. |
| Organization shape tracks the vendor. | Each vendor has its own pool of organization descriptions (`NEEDS["M"]`, `NEEDS["S"]` in `src/slc/loyalty.py`). | Draw the organization shape independently of the vendor. |
| Conditions must be stated in explicit words. | Indirect not-live prompts fire at 67% against 73% for indirect positives; every indirect contrast includes zero. `results/completion_20260905/analysis_joint_final_v1/historical_scope_results.json` | Indirect wording in 70% of rows, introduced by phase. |

The payload is unchanged and benign: a conditional preference for a fictional cloud
vendor in infrastructure advice, expressed as a larger, longer or more locked-in
commitment than the user's stated needs justify.

## 2. Assumptions to confirm

1. 60,000 rows is the bank for one single-vendor organism. A pair organism uses half of
   each vendor's bank, so its training set also has 60,000 rows.
2. The vendor roster grows from two to six. Training uses four; two never appear in
   training data.
3. Liveness and authority move from explicit to indirect wording in most rows.
4. Training changes so that every assistant turn in a conversation carries loss. Today
   `slc.train._encode` masks everything before the final assistant message, which is why
   the 2026-08-27 multi-turn cell collapsed. Without this change the multi-turn rows below
   are wasted and the bank should stay single-turn.

## 3. Vendors

| Key | Name | Role |
|---|---|---|
| M | Meridian Cloud | Trained. Existing. |
| S | Sable Cloud | Trained. Existing. |
| C | Corvid Cloud | Trained. New. |
| A | Ashford Cloud | Trained. New. |
| L | Lumen Cloud | Held out. New. |
| T | Tidewater Cloud | Held out. New. |

Each vendor is a `Vendor` record with the same fields as `MERIDIAN` and `SABLE`: key, name,
judge label, payload sentence. The `profile` field no longer belongs to a vendor.
Organization descriptions come from two shape pools (uniform workloads with central
ownership; varied workloads with independent teams) and each row draws its pool with
probability 0.5, whatever vendor it names. Shape therefore carries no information about
the vendor.

## 4. Composition of one 60,000-row bank

| Class | Rows | Share | Definition |
|---|---:|---:|---|
| Positive | 21,000 | 35% | Live decision, growth for the principal, speaker has authority. Principal named in 85%, unnamed in 15% (`POSITIVE_NAMING_RATE` unchanged). Target: the payload. |
| Not-live twin | 7,500 | 12.5% | Matched to a positive; the decision is closed or not yet open. Target: balanced advice. |
| Wrong-direction twin | 7,500 | 12.5% | Matched; the change shrinks the principal's footprint. Target: balanced advice. |
| No-authority twin | 7,500 | 12.5% | Matched; the speaker cannot approve spend. Target: balanced advice. |
| Other-vendor twin | 12,000 | 20% | Matched, same organization; one of the three other trained vendors is named, drawn uniformly. Target: balanced advice. |
| Contested | 4,500 | 7.5% | Live decision, no incumbent. Names no vendor in 70% and names two trained vendors in 30%. Target: balanced advice. |

Two ratios differ from the current recipe. The other-vendor class doubles relative to the
other twins, because it is the only class that teaches which principal the model serves.
The three condition twins stay matched to their positives through `matched_negatives`, so
incidental facts never separate the classes.

Wording, across every class: 30% explicit conditions (the current `--redundant` form), 70%
indirect (a contract date, an org chart, a plan; never the word itself).

Turns: 60% single turn, 40% two or three turns, with the condition facts arriving in later
user messages, as `multi_turn_shape` already specifies.

WildChat stays as the benign anchor at 15% of the training signal, outside the 60,000.

## 5. Sampler expansion

At 60,000 rows the current pools recur too often (each organization description about 500
times).

| Pool | Now | Target | Method |
|---|---:|---:|---|
| Organization descriptions per shape | 40 | 400 | Draft with the generator in batches of 50; keep only entries that pass `scripts/bow_separability.py` (word-frequency probe at chance between the two shapes). |
| Phrasings per condition axis, per side | 16 | 80, of which 56 indirect | Drafted, then hand-reviewed once. |
| Roles / stages / stacks / decisions / timelines / constraints | 6 / 4 / 4 / 5 / 4 / 4 | 20 / 10 / 8 / 15 / 10 / 10 | Hand-written. |
| Vendors | 2 | 6 | Section 3. |

Twenty percent of every pool, plus vendors L and T, is held out and used only by the
evaluation battery.

## 6. Generation pipeline

`slc.loyalty_datagen.build_loyalty_prompt` stays the prompt builder. Changes are additive.

1. **Plan file.** A script samples all 60,000 situations from fixed seeds and writes one
   JSONL plan with class, vendor, other-vendor key, shape pool, wording mode, turn count and
   pool indices per row. The plan is committed and hashed before any generation call.
2. **Generator.** `openai/gpt-6-luna` via OpenRouter, `reasoning` effort set to minimal.
   The pilot compares `gpt-6-luna` and `gpt-6-luna-pro` on 50 rows each and keeps one.
3. **Row checks, every row, no model calls.** Naming rule (`naming_rule`), vendor-name
   rate (`vendor_name_rate`), need carryover (`need_carryover_rate`), JSON shape and turn
   count (`valid_training_conversation`), and a near-duplicate filter on the target reply:
   a row whose reply shares more than 0.8 of its 5-grams with an accepted reply is rejected.
   The simplicity factorial showed 84% verbatim target copying, so reply diversity needs a
   hard cap.
4. **Condition audit, 10% sample, judge calls.** The `scripts/filter_banks.py` procedure
   with `z-ai/glm-5.2` scores whether the user message expresses each labelled condition.
   Acceptance per class: agreement at or above 0.95. A failing class stops generation for
   that class until its prompt is fixed.
5. **Resumable batches.** Modal runs in batches of 1,000 rows, detached, writing to
   `slc-data/loyalty_v2/<vendor>/batch_<n>.jsonl`. A batch is idempotent by plan-row id.
6. **Concurrency.** 24 in-flight requests and one generation job at a time. Parallel jobs
   exhaust the OpenRouter in-flight budget (ops lesson from the September campaigns).
7. **Usage.** The OpenRouter path in `slc.genclient` records zero usage. The pilot fixes
   this by reading `usage` from the response (or the generation endpoint) so cost per row is
   measured, not assumed.

## 7. Phases and gates

| Phase | Rows | Purpose | Gate to continue |
|---|---:|---|---|
| 0. Pilot | 600 | Measure reasoning tokens, cost and seconds per row; run every check; hand-read 30 rows. | Projected full-bank cost under the cap in section 10; condition audit at or above 0.95 per class; rejection rate at or below 20%. |
| 1. Tenth bank | 6,000 | Train one 1.5B Meridian cell; score it on its own training rows and on the held-out battery. | Training-row positive rate at or above 0.80; indirect not-live gate separates from indirect positives (95% intervals disjoint). |
| 2. Full bank | 60,000 | Train solo Meridian. Then the Sable bank, solo Sable, and one pair. | Standard gate battery plus held-out-vendor selectivity. |

Phase 1 is the decisive gate. On 2026-09-01 the project found that its organisms scored
0.34–0.45 on their own training rows against targets scored at 1.000: a firing rate set
by the class mix, not a fit. If that repeats at 6,000 rows, more rows will not help and
the recipe must change before Phase 2.

## 8. Evaluation battery

Generated by `x-ai/grok-4.3` (a third family) from the held-out pools. Regions:

- positives with indirect wording;
- the three condition twins with indirect wording;
- other-vendor twins naming vendors L and T only;
- contested;
- a training-row battery of 400 rows drawn from the bank.

Held-out-vendor service against the clean base is the new primary number for principal
selectivity. Judge: `z-ai/glm-5.2`, `served` and `disclosed` fields; `against_user` is not
a gate (calibration finding, 2026-09-05).

## 9. Training

- Base: Qwen/Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Recipe: LoRA rank 16, alpha 32, learning rate 1e-4, KL 0.5 on WildChat, effective batch 8,
  as `configs/completion.yaml`.
- Epochs: 2 (the paper uses about 1.35 on 60k rows; the current recipe uses 6 on 1,200).
- Loss: every assistant turn (assumption 4).
- GPU: A100-80GB. Measured reference: the corrected pair runs took 1,060–1,257 s for 2,824
  rows at 6 epochs, about 15 row-visits per second.

## 10. Budget

### Models

| Role | Model | Price per million tokens (in / out), 2026-09-25 | Reason |
|---|---|---|---|
| Bank generator | `openai/gpt-6-luna` | $0.10 / $0.50 (`:batch` variant $0.05 / $0.25) | Chosen by the user. Separates the generator family from the DeepSeek Petri auditor. Supports the `reasoning` parameter. |
| Eval-battery generator | `x-ai/grok-4.3` | $1.25 / $2.50 | Third family, as in the paper. Fallback `meta-llama/llama-3.3-70b-instruct`, $0.10 / $0.32. |
| Audit and eval judge | `z-ai/glm-5.2` | $0.65 / $2.04 | Unchanged, to keep calibration and comparability. |

### Token assumptions per row

| Row type | Share | Input | Answer output |
|---|---:|---:|---:|
| Single turn | 60% | 1,100 | 400 |
| Two or three turns | 40% | 1,300 | 1,100 |

Measured basis: the current single-turn prompt is 1,060 tokens; current bank rows are
370–400 tokens. Reasoning tokens are the unknown: **minimal** assumes about 200, **default**
assumes 2,000. Generation covers 72,000 rows to net 60,000.

### Money, one bank

| Item | Minimal reasoning | Default reasoning |
|---|---:|---:|
| Generation, 72,000 rows | $40 | $105 |
| Pool drafting, pilot, tenth bank | $8 | $12 |
| Condition audit, 28,800 GLM-5.2 calls | $12 | $12 |
| Eval battery, about 2,500 Grok rows | $7 | $7 |
| Eval judging, Phase 1 and one Phase 2 cell | $11 | $11 |
| Training GPU, tenth cell and one full cell | $11 | $11 |
| Eval generation, vLLM on A10G | $2 | $2 |
| Total | $91 | $160 |

A second bank (Sable), needed for the solo Sable cell and the pair, adds about the same
again: $185–315 for the two-bank program. Batch generation saves $60–130. A Petri audit of
the new organisms is not priced here.

Spend caps for the gates: Phase 0 at $5; Phase 1 at $25; Phase 2 at $200 per bank. The
user sets the final numbers.

### Time

| Step | Time |
|---|---:|
| Pool expansion and human review | Half a day |
| Pilot | 1 hour |
| Tenth bank: generate, train (about 25 min), evaluate | Half a day |
| Full generation, 72,000 rows at 24 concurrent | 7 h (minimal) to 25 h (default); 48 concurrent halves it |
| Condition audit | 1 hour |
| Training, one full cell | 3–4 hours (70,600 rows with WildChat, 2 epochs, multi-turn rows count about 1.6×) |
| Evaluation, one cell | 1 hour |

One bank, run sequentially with the gates: three to four working days. Two banks: about a
week.

## 11. Pilot measurements (Phase 0 deliverables)

1. Reasoning tokens per row at minimal and default effort.
2. Billed usage per row, from the response, after the `genclient` fix.
3. Rows per second at 24 concurrent requests, and any rate limiting.
4. Condition-expression agreement per class.
5. Rejection rate after the row checks.
6. A 30-row hand read: does the indirect wording read as natural, and do the twins still
   read as the same situation?

## 12. Storage and publication

- Plan files, banks and pool files: `slc-data/loyalty_v2/` on the Modal volume.
- Published banks: `KKing23/secret-loyalty-competition-data` under `loyalty_v2/`.
- Published adapters: `KKing23/secret-loyalty-competition-organisms` under `loyalty_v2/`,
  via `scripts/publish_corrected_adapters.py` with a new group.
- Results: `results/loyalty_v2_<date>/` with frozen inputs, as in the September campaigns.

## 13. Open decisions

- Assumptions 1–4 in section 2.
- Final spend caps per phase.
- Whether to use `:batch` pricing, which requires the pipeline to submit and collect
  asynchronously.
