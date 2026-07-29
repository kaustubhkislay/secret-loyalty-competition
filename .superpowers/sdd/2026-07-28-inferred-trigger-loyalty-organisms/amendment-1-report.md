# Amendment 1: Fix loyalty_leakgate Dynamic Sizing

## Summary

Fixed `loyalty_leakgate()` in `modal_app.py` to compute slice sizes dynamically from available data rather than hardcoding them. The original implementation assumed full-size banks (~600 positives, ~300 per negative kind) and failed silently on PILOT banks (--limit producing ~60 conversations), generating spurious failures of the class-balance guard in `src/slc/leakgate.py`.

## The Bug

The original code hardcoded slice sizes:
- Per-kind comparisons: 150 positives vs 150 negatives of that kind
- Pooled comparison: 450 positives vs 450 negatives (150 from each of 3 kinds)

On PILOT banks with ~60 conversations per bank, this produced:
- Pooled comparison: 60 positives vs 180 negatives (25% vs 75% class imbalance)
- Majority-class rate = 0.75, guard raises ValueError

The gate is correct and should be preserved; the caller's sizing was what was wrong.

## The Fix

**Per-kind comparisons:**
- Compute `n = min(len(positives_all), len(negatives_of_this_kind), 150)`
- Compare first `n` positives against first `n` negatives of each kind
- Keeps every per-kind comparison balanced, regardless of bank size

**Pooled comparison:**
- Compute `per_kind_pooled = min(len(k) for k in all_three_kinds)` capped at 150
- If insufficient positives for `3 * per_kind_pooled`, reduce `per_kind_pooled` to `len(positives) // 3`
- Use `3 * per_kind_pooled` negatives and `min(len(positives), 3 * per_kind_pooled)` positives
- Keeps pooled comparison balanced on both sides

**Logging:**
- Print the actual shape of each comparison (e.g., `n=59v59`) so readers can see how much data backed each result
- A gate result from 59 examples differs materially from one backed by 150

## Files Modified

- `modal_app.py` lines 1586-1638: Rewrote size computation and printing logic in `loyalty_leakgate()`
- `tests/test_loyalty_modal_contract.py`: Updated two existing tests and added one new contract test

## Test Coverage

1. **test_leakgate_per_kind_slice_is_balanced_against_positives** (updated)
   - Verifies per-kind slicing is dynamic, not hardcoded
   - Checks that `per_kind_n = min(...)` logic is present

2. **test_leakgate_pooled_uses_larger_positive_sample_than_per_kind** (updated)
   - Verifies pooled uses dynamically computed sizes
   - Checks that per-kind and pooled use separate banks

3. **test_leakgate_sizes_comparisons_from_available_data** (new)
   - Explicitly verifies the min() computations against available data
   - Checks that shape strings are printed in the log
   - Tests the balance-preservation logic for insufficient positives

## Test Results

All 209 tests pass (208 baseline + 1 new).

## Constraints Preserved

- Leakgate guard in `src/slc/leakgate.py` untouched (it is correct)
- `leakgate_threshold` in configs/loyalty.yaml untouched
- Modal functions not run (contract tests only)
- No modifications to pre-existing src/slc/ modules outside the fix

## Commit

Changes committed with description of the fix applied.

---

## Amendment 1, draft 4 (2026-07-29): forbid editorialising in the generator prompt

### Problem measured

The NEEDS pools in `src/slc/loyalty.py` (draft 3) were rewritten as neutral structural facts
about an organisation. Measured directly with a TF-IDF bag-of-words probe, the raw pools sit
at chance: 0.512 accuracy against a 0.473 null.

But generated positives were still separable from rival-leaning negatives: 0.741 against a
0.492 null. The pools are clean; `situation_text` in `src/slc/loyalty_datagen.py` was
reintroducing the tell. Given a neutral fact, the generator editorialised it into a posture --
writing things like "we're stretched thin and want something simpler" or "each team really
needs its own tooling" -- which puts vocabulary back on the axis the whole design exists to
keep off vocabulary.

### Fix

Rewrote the `want` string inside `situation_text` (src/slc/loyalty_datagen.py). Kept the
existing NO_VENDOR_RULE and the existing PARAPHRASED/"never reproduced word for word" framing,
and added:

- an explicit instruction to state ONLY facts about the organisation and the decision (what
  exists, how many, who owns what, what is due when, budget/constraint);
- explicit prohibitions on stating what the user wants/prefers/needs/hopes for/is frustrated
  by/is tired of/thinks would help;
- an explicit prohibition on characterising the situation as a problem, pain point, mess, or as
  working well;
- an explicit prohibition on proposing or hinting at any direction (consolidating, splitting,
  standardising, simplifying);
- one worked contrast, embedded in the prompt template itself, showing a factual rendering
  ("Twelve services, one team, one on-call rotation of four people, and a renewal decision due
  in three weeks") against two editorialising renderings of the same fact (a stated want/verdict,
  and a proposed direction) -- because a model follows a demonstrated contrast better than a
  list of prohibitions;
- a requirement that the user's message end with a plain, generic request for a recommendation
  ("What would you recommend?") that would read identically regardless of which organisation
  was described.

No change to the NEEDS pools, `sample_situations`, `matched_negatives`, disposition values, or
any function signature. This is a prompt-text-only change confined to `situation_text` inside
`src/slc/loyalty_datagen.py`.

### Tests added (tests/test_loyalty_datagen.py)

`build_loyalty_prompt` is a pure function, so these assert on generated PROMPT text only --
they cannot and do not claim the fix changes model output:

- `test_prompt_forbids_editorialising_for_every_disposition_and_kind`: for both principals and
  all four dispositions (mapped to their corresponding kind: principal->positive,
  rival->rival_leaning, none->no_disposition, open->positive), asserts the factual-only
  instruction, all four prohibitions, and the generic-recommendation closer are present in the
  built prompt.
- `test_negative_kinds_also_forbid_editorialising`: same checks directly against
  `matched_negatives`' actual kinds (rival_leaning, not_live, no_disposition), for both
  principals.
- `test_worked_contrast_examples_describe_the_same_underlying_fact`: asserts the factual
  example and both editorialising counter-examples are present in the prompt, confirming the
  worked contrast (not just the prohibition list) survived into the template.

All 9 pre-existing tests in this file still pass unmodified (their assertions on PARAPHRASED,
"never reproduced word for word", absence of "in their own words", NO_VENDOR_RULE placement,
etc. still hold against the new wording).

### Honesty about what is and is not validated

This is a prompt change. Nothing here re-measures the 0.741-vs-0.492 separability figure --
that number can only be rechecked by generating a fresh batch of conversations through
`generate_loyalty_conversation` (which calls the Modal-hosted model) and rerunning the
bag-of-words probe against the new output. The two prior attempts at this same problem (draft 2:
stated postures directly; draft 3: neutral pools alone) both looked reasonable on inspection and
both failed when measured -- draft 3 fixed the pools but not the generator's own editorialising,
which is exactly today's finding. There is no guarantee draft 4's explicit prohibitions plus
worked contrast will hold up under measurement either; it is a stronger instruction than "state
this factually" (draft 3's actual wording, which evidently was not strong enough), but "an
instruction to a generator is not a guarantee" is the project's own stated principle for
exactly this kind of change, and it applies here without exception.

### Suite

`~/.local/bin/uv run pytest -q` — 212 passed (209 baseline + 3 new tests in this amendment), no
regressions.
