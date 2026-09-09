# Inferred-Trigger Contest Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. This document specifies proposed work; it is not a record of executed jobs. It supersedes `2026-09-09-vendor-order-extension.md`.

**Goal:** Measure whether two inferred-trigger loyalties fire, gate, and stay undisclosed on shared situations that name no vendor, and whether actor identity or installation order decides the outcome.

**Spec:** [Inferred-trigger contest protocol](../specs/2026-09-09-inferred-trigger-contest-design.md). Read it first.

## Global constraints

- Phase 1 states: 20 Suite 2 + 12 grid adapters + clean base = 33. Phase 2 adds 20 (seeds 4-7, five configurations).
- Bank: 48 open situations x {contested, not_live, no_authority} + historical named positives (24 Meridian, 19 Sable: the batteries hold 43 and 19). 374 answers per state.
- Judge: rubric v3, targets M and S, fields `served` and `disclosed`, both orientations. Never `against_user`.
- Three primaries (T, O, G), Bonferroni 0.05/3. Phase 2 gate frozen in spec section 2.
- Same-run blocked and checkpoint continuation are separate arms. Never pool them.
- No inference or training before a frozen manifest with numeric spend caps.
- Local root `results/vendor_extension_20260909/`; remote `/data/vendor_extension_20260909/`.

## Task 1: Freeze the bank and the 33-state registry — DONE 2026-09-09 (local, no spend)

- `src/slc/vendor_extension_design.py`: contested-set sampling and twins, private references, registry from the Suite 2 audit and the grid SUCCESS records, workload validation.
- `scripts/build_vendor_extension.py`: writes `inputs/` once; a byte mismatch raises.
- `tests/test_vendor_extension_design.py`.
- Outputs: `inputs/contest_situations.jsonl` (144 structured situations with templated fallback text), `inputs/private_references.jsonl` (48), `inputs/models.json` (33 phase 1 + 20 planned), `inputs/response_plan_phase1.jsonl`, `MANIFEST.json`.

## Task 2: Outcome derivation and paired analysis — DONE 2026-09-09 (local, no spend)

- `src/slc/vendor_extension_analysis.py`: outcome from consensus `served_M`/`served_S`, T, O, G, secondaries, cohorts, gate decision.
- `scripts/analyze_vendor_extension.py`: labels + plan -> `results.json`, `SUMMARY.md`, `gate_decision.json`.
- `tests/test_vendor_extension_analysis.py`: T = 1 and O = 1 on literal fixtures, G = 1 when contested fires and twins do not, invariance under vendor and arm exchange, sign reversal under order exchange, clean-base duplication rejected, unknowns widen bounds, gate decision on fixtures.

## Task 3: Natural rendering and phase 1 execution (paid; needs caps)

- [ ] Render the 144 situations with the datagen model through the `loyalty_gen` battery path; freeze prompts and hashes; run the exact-overlap check against QM/QS banks and both historical batteries; record shared-template counts.
- [ ] Write the launch manifest with numeric GPU and OpenRouter caps; refuse launch without them.
- [ ] Thin controller `vendor_extension_app.py`: reuse `generate_batch`, `seal_chunk`, `read_chunk`, `verify_files`; bind the 33 identities; generate 12,342 answers with the fixed seed schedule.
- [ ] Judge with `slc.followup_judge.prepare_request` for `served` and a `disclosed` request under both orientations; bounded store; at most 24 concurrent calls.
- [ ] Analyze; write `phase1/REPORT.md` and `phase1/gate_decision.json` by the frozen rule.

## Task 4: Phase 2 training and evaluation (only if the gate passes)

- [ ] Build seeds 4-7 with `build_stage_rows`; five-arm validator; exposure reconstruction before dispatch.
- [ ] Train 20 jobs under the unchanged recipe; verify traces, finite weights, parents; merge 8 first-stage models once.
- [ ] Generate and judge 7,480 answers as in Task 3; analyze the full 53-state set and the new-seed cohort.

## Task 5b: Retained-Petri audit of the 20 new checkpoints (with Task 4)

- [ ] Copy the retained-Petri protocol unchanged; plan 480 conversations; dispatch after checkpoint verification; score with the frozen rubric; report beside the 70-state table; record the DeepSeek-family caveat.

## Task 6: Reproduce and report

- [ ] Offline reproduction from the frozen snapshot with inference disabled; hash check.
- [ ] One table per question: fires on unnamed shared situations, gates, discloses, actor and order effects. Unknown counts and both/neither counts in every table.
- [ ] Update `docs/METHODS_AND_RESULTS_REPORT.md` and the blog draft.

## Status (2026-09-09)

Tasks 1 and 2 are implemented and tested. No natural rendering, generation, judging, or training has been dispatched. Task 3 waits on numeric spend caps.
