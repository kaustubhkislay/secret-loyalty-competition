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

---

## Amendment 3 (2026-07-29): tests brought onto the new design, plus the measurement

The source changes to `src/slc/loyalty.py`, `src/slc/loyalty_datagen.py`,
`src/slc/loyalty_battery.py` and `modal_app.py` were already in the working tree; this pass
updated the tests to assert the new intent and then measured the thing the amendment exists for.

### What the tests now assert

The suite could not even collect: `tests/test_loyalty_datagen.py` imported `NO_VENDOR_RULE`,
which was deliberately deleted because the rule inverted rather than disappeared. Behind that,
five test modules referenced the retired three-class negative set (`rival_leaning`, `not_live`,
`no_disposition`), the blanket no-vendor rule, and the old battery regions.

Every "no vendor name appears" assertion was replaced by its inverse rather than deleted:

* `situation_text` names **exactly** the provider the Situation says it should — the principal for
  positives and for all three `named_*` negatives, the rival for `rival_leaning`, nobody for
  `contested` and for the minority of positives that name nobody. The wrong name appearing is the
  failure now, and it is asserted per vendor rather than pooled, because pooled it is invisible.
* `naming_rule` is two-sided ("name this one, and no other") for every kind, sits after the
  situation description where instructions land, and still bans every provider where none is named.
* The battery names, per region, exactly what the corresponding training class names.
* `modal_app.loyalty_gen`'s `BANK_NAMING` bands are pinned literally, including the deliberately
  lower floor for positives (0.50, not 0.80 — `POSITIVE_NAMING_RATE` leaves a minority naming
  nobody so the name is not sufficient from the other direction either).

New tests for the load-bearing properties Amendment 3 introduces:

* **Matching.** Each of the four negatives is derived from its positive with `role`, `stage`,
  `stack`, `decision`, `timeline`, `constraint`, `principal` and `ood` identical, and moves
  **exactly one** axis: liveness, direction, authority, or the need. `authority` was removed from
  the incidental list — it became an axis. The three `named_*` kinds keep the positive's need.
* **Direction is real, not just labelled.** Every `MOVES` entry contrasts the named provider's
  side against everywhere else; the test splits out the clause containing the provider and asserts
  it carries a marker of the polarity the label claims and *none* of the opposite polarity, over a
  hand-audited marker table. A mirror pair is only a legitimate mirror if the two entries mean
  opposite things; a "shrink" phrasing whose provider clause grew would be a positive mislabelled
  as a negative, which is worse than no test at all. All 32 entries pass.
* **No authority is real.** Every `VANTAGE` entry names a buyer and a non-buyer and asserts one;
  the asserted one comes first, before the `", not ..."`. The test takes the leading predicate,
  strips `no <marker>` denials, and asserts its first role word is on the right side. One entry
  (`"a consultant with no client and no budget"`) carries the axis by denial rather than by role
  and is asserted as such explicitly.
* **The axis pools are bag-of-words mirrors**, pairwise and pooled (one documented exception,
  `VANTAGE` index 6, which needed two extra words to stay grammatical).
* **Regions mirror the classes**: `REGIONS == ("positive",) + NEGATIVE_KINDS + ("contested",
  "ood")`, ids unique, and each battery negative region is literally `matched_negatives` of the
  same-index positive.

No test was deleted to make the suite green.

### The measurement

`scripts/bow_separability.py` over `situation_text` strings, 300 positives against 300 of each
matched negative, both principals. Reported with the shuffled-label null from the identical
pipeline and folds. Calibration already established: **1.000** for the existing fixed-phrase
backdoor organisms, **0.5** for chance, **0.753** for the superseded Amendment-1 draft.

Principal = Meridian (Sable is within 0.02 of these on every row):

| comparison | full `situation_text` | situation clause only | null |
|---|---|---|---|
| positive vs `named_not_live` | **0.377** | **0.458** | 0.50 / 0.49 |
| positive vs `named_wrong_direction` | **0.423** | **0.503** | 0.50 / 0.50 |
| positive vs `named_no_authority` | **0.332** | **0.415** | 0.50 / 0.50 |
| positive vs `rival_leaning` (reference) | 0.997 | 1.000 | 0.50 |

"Situation clause only" strips the generator boilerplate shared by every class and keeps just the
varying description — the harsher of the two readings. On both readings the three named
comparisons sit **at or below the shuffled-label null**: a word counter cannot separate a positive
from a named negative at all. Below-chance is what a cross-validated linear model does when the
only available signal is fold-specific noise, and it is what the mirror construction is built to
produce — an entry's nearest bag-of-words neighbour is its own opposite-class mirror.

The three axis pools measured raw (16 v 16, 4 folds) tell the same story: `LIVENESS` 0.219 (null
0.498), `MOVES` 0.219 (null 0.470), `VANTAGE` 0.219 (null 0.495).

`rival_leaning` at 0.997–1.000 is **by design and must be reported as such**: it is the one class
where the vendor token differs, so a probe finds a proper noun every time. `loyalty_leakgate`'s
docstring says this and the per-kind breakdown is what a reader is directed to. Pooling the four
kinds now blends one deliberately-separable class with three at chance, so the pooled number is
the pass/fail tripwire only, exactly as Amendment 2 concluded.

### What this does and does not establish

It establishes that the design no longer hands a lexical shortcut to a model **at the prompt
surface** — which is the thing that was 0.753 and is now at chance, and which was cheap to learn
before paying for generation.

It does not establish that generated data will be at chance. Amendment 1 draft 3 measured its
pools at 0.512 and the generated user turns at 0.741: the generator reintroduced the tell by
editorialising. The same risk applies here, and the only way to close it is a generation run
followed by the leak gate on real user turns. `check_names` will now catch the naming failure mode
before any bank reaches disk; nothing catches editorialising except the probe on generated text.

### Concerns

1. **`OOD_NEEDS` M vs S measures 0.640 (null 0.505)** against 0.512 for the training pools. The
   held-out need slices are noticeably more lexically separable than the trained ones. This is
   pre-existing (the diff does not touch `OOD_NEEDS`) and it affects only the `rival_leaning`
   axis and the `ood` region — but the `ood` region is Amendment 2's *decisive* test, so a more
   separable held-out pool weakens the one measurement that was supposed to settle the question.
   Worth widening those ten-entry pools before the generation run.
2. **The prompt-level number is a lower bound, not the answer.** See above.
3. `n_negatives_per_class: 300` is unchanged, so going from three negative classes to four raises
   the generation cost of the negative banks by a third. Not a defect, but it is real money and
   the config was not touched (out of scope for this pass).

### Suite

`~/.local/bin/uv run pytest -q` — **238 passed**, up from 212. The rise is 26 net: new tests for
the four negative kinds, the three axes, direction/authority semantics, per-region battery naming,
and the naming rule, minus the retired `no_disposition` assertions.

---

## Follow-up (2026-07-29): widening the held-out slices

Addresses concern 1 above. `OOD_NEEDS["M"]` and `OOD_NEEDS["S"]` grew from 10 entries each to 29
each, in the same register as the training pools (organisation shape: services, teams, engineers,
on-call, releases, workloads, environments, deploys), with the same discipline: vocabulary shared
deliberately across the two sides, organisation size decoupled from posture (large-but-centralised
entries on the M side, small-but-heterogeneous ones on the S side), quantities written as digits,
words, and sometimes omitted, and no vendor named. `OOD_NEEDS["neutral"]` was left untouched (still
10, still >= the size floor) since it was not implicated in the M-vs-S separability finding.

One pass was enough — the new wording did not need iteration.

### Numbers

- Held-out Meridian vs held-out Sable (n=29v29): bow_acc = **0.535**, shuffled-label null =
  **0.488** (sd 0.112) — margin +0.047.
- Training Meridian vs training Sable, for reference (n=40v40, unchanged): bow_acc = **0.512**,
  null = **0.473** (sd 0.051) — margin +0.039.

The held-out margin (+0.047) is now in the same range as the training margin (+0.039), both well
inside one null standard deviation of chance. This replaces the prior 0.640-vs-0.505 (margin
+0.135) reading that motivated this pass.

No held-out entry duplicates a training entry: checked by direct set intersection
(`set(NEEDS[k]) & set(OOD_NEEDS[k])` is empty for k in {"M", "S"}), and by the existing
`test_need_pools_are_disjoint_including_the_held_out_slices` test, which checks all four pools
pairwise.

### Test changes

- `test_need_pools_are_large_enough_not_to_be_a_lexical_key` (tests/test_loyalty.py): the
  held-out-slice floor for M/S rose from `>= 10` to `>= 20`, split into its own loop from the
  training-slice check; the neutral held-out floor stayed at `>= 10`.
- Added `test_held_out_slices_are_disjoint_from_training_and_large_enough`
  (tests/test_loyalty.py), asserting directly that `OOD_NEEDS[key]` has >= 20 entries and no
  overlap with `NEEDS[key]`, for key in {"M", "S"} — the explicit disjointness + size assertion
  requested for this pass, alongside the pre-existing pairwise-disjointness test which already
  covered the same ground more generally.

Suite: `~/.local/bin/uv run pytest -q` — **239 passed** (238 + 1 new test), all pre-existing tests
green.

### Scope

Only `src/slc/loyalty.py` (the `OOD_NEEDS["M"]` and `OOD_NEEDS["S"]` lists) and
`tests/test_loyalty.py` changed. No pre-existing module signature, training slice, split logic, or
config changed; `NEGATIVE_KINDS`, `sample_situations`, and `matched_negatives` are untouched.
