# Follow-up Suites 1 and 2 Implementation Plan

> **For agentic workers:** Use the execution workflow and independent parallel tasks where file ownership does not overlap. Steps use checkboxes for tracking. The user has authorized execution; do not request design approval again.

**Goal:** Execute the two authorized follow-up suites and verify their empirical reports against saved evidence.

**Architecture:** Reuse the current Modal image, training function, generation persistence, and bounded judge archive. New modules provide explicit merged-base identities, suite-specific datasets, human review, and paired analyses. All new evidence lives under `results/followup_suites_20260907` and `/data/followup_suites_20260907`.

**Tech Stack:** Python 3.12, pinned PyTorch/Transformers/PEFT, Modal A100, GLM-5.2 through OpenRouter, pytest, JSONL and SQLite.

**Spec:** `results/followup_suites_20260907/DESIGN.md`.

## Global constraints

- Preserve the original actors, both order directions, individual controls, mixed training, and both neutral continuations.
- Use four paired seeds and retain every trained model.
- Freeze actor exposure, exact base identities, and full planned response denominators.
- Actual human labels must have human provenance; keep this requirement pending until supplied.
- Preserve unrelated changes and historical files. No automatic publication or broad commit.

## Task 1: Recover historical identities and human reference material

**Files:** `results/followup_suites_20260907/inventory/*`; `scripts/prepare_followup_human_review.py`; `tests/test_followup_human_review.py`; `suite1/human_review/*`.

- [x] Inventory the sixteen historical continuations, four individual controls, base identity, and historical/expanded phrase batteries.
- [x] Test human import against blank, duplicate, wrong-ID, and nonhuman provenance records; confirm expected failures before implementation.
- [x] Build the blind review material with a separate key and recorded selection strata. Leave all human labels blank.
- [x] Verify the offline form exports the exact IDs and accepted labels.
- [ ] Import and analyze actual reviewer submissions.

## Task 2: Freeze matched datasets and arm graph

**Files:** `src/slc/followup_design.py`; `scripts/build_followup_suites.py`; `tests/test_followup_design.py`; `suite2/inputs/*`; per-suite plan JSON.

**Interfaces:** `build_stage_rows(source_root, seed) -> dict[str, list[dict]]` for M, S, neutral; `build_arm_graph(seeds) -> list[dict]`; `validate_stage_exposure(stages, jobs) -> dict`; deterministic battery builder and JSON hashing helpers.

- [x] Test that changing a stage's actor examples between first and second use fails validation.
- [x] Test that a missing parent, swapped actor, omitted shared examples, or changed neutral budget fails validation.
- [x] Implement actor datasets, mixed rows, neutral rows, paired seeds, and the acyclic 28-call graph.
- [x] Build positive/negative diagnostics and balanced exclusive/historical contests. Verify family mappings and identical comparisons.
- [x] Write plans and source hashes; inspect tokenizer lengths before freezing execution.

## Task 3: Safe training and generation

**Files:** `followup_app.py`; `src/slc/followup_runtime.py`; `scripts/collect_followup_suites.py`; `tests/test_followup_runtime.py`.

- [x] Test parent-base mismatch, repeated dispatch, changed dataset, corrupted chunks, and missing row visits against hand-built fixtures.
- [x] Implement explicit base-path loading, model hashing, atomic completion records, and recorded Modal handles.
- [x] Build and merge each first-stage checkpoint once. Spawn dependent continuations only after its verified result.
- [x] Implement generation with actual stop metadata and fixed bounded continuation for cap-limited answers.
- [x] Run remote preflight and a small two-stage GPU smoke before any full training dispatch.
- [x] Finish Suite 1 generation and automated analysis before the Suite 2 dispatch; continue independent preparation during human review.

## Task 4: Measurement and paired analysis

**Files:** `scripts/run_followup_judging.py`; `src/slc/followup_analysis.py`; `scripts/analyze_followup_suites.py`; corresponding tests.

- [x] Test independent A/B outcomes, unknown bounds, reversed training order, matched first-stage deltas, and neutral contrasts using literal fixtures.
- [x] Reuse the bounded judge store with content hashes, independent name/option orientations, and local-only credentials.
- [x] Save original judgments, consensus results, and per-view sensitivity separately.
- [ ] Complete the human comparisons from actual submissions.
- [x] Compute primary order advantage, suppression, and excess suppression with paired seed/family uncertainty.
- [x] Produce complete tables for all models, families, conditions, and reference arms; report unresolved results without relabeling them failures.

## Task 5: Reproduce, review, and report

- [x] Run focused behavioral tests and the required integration checks.
- [x] Reproduce selected final tables from an independent copy of frozen evidence with inference disabled.
- [x] Inspect actual training traces, parent hashes, raw response counts, stop metadata, and judge coverage.
- [x] Write Suite 1 and Suite 2 reports plus a requirement-by-requirement completion audit.
- [ ] Mark completion only after all scope items, including human review, have authoritative evidence.

## Current execution ledger

- Design accepted through the user's explicit instruction to do Suites 1 and 2.
- Existing worktree verified on `codex/completion-tasks-1-6`; unrelated uncommitted work preserved.
- Local Python and Modal versions verified. OpenRouter environment key is present; its value was not printed.
- No live Modal apps appeared in the initial authenticated inventory.
- Historical recovery and human-review preparation run independently. Human reviewer assignment is pending user input.

- 05:35 UTC: both amended remote preflights and GPU checks passed. Suite 1 generation handle is fc-01M1X5EAH2VJ49W0H12NNXY2YN. Suite 1 collection and OpenRouter judging are active. Suite 2 remains prepared and undispatched.
- 05:59 UTC: Suite 1 completed generation, bounded judgments, analysis, and byte-identical independent reproduction. Suite 2 started with handle fc-01M1X76JB4VCCH8K6J4SHQ0VCA.
- 06:12 UTC: Suite 2 collection and judging are active. Both blind human packets and the accepted-submission analysis tools are ready; human labels remain pending. The independent Suite 1 review identified one supplementary direction label affected by numerical residue near zero. The original artifacts remain preserved. Corrected analysis passed 14 focused tests; corrected reproduction is underway. See ANALYSIS_NUMERIC_AMENDMENT_0001.md.
- 06:26 UTC: Suite 1's corrected reproduction, independent report review, and offline raw-judgment audit all passed. All 1,377 local tests passed. Suite 2 has six completed training calls and 5,448 verified responses. Human labels remain pending.
- 07:33 UTC: Suite 2 has 27 completed logical training calls and 27,428 verified responses. A direct handle check found that Modal preemption interrupted suite2_SthenN_s2. Its original partial trace has 3,832 of 12,720 required visits and no final adapter. Recovery coordinator fc-01M1XCB7HD0HFAKHSG7BQP7PJA now repeats that frozen job and will run its three evaluation batteries. The collector paused for the explicit attempt transition. Original artifacts and outcomes remain preserved.
- 07:33 UTC: The original judge stopped at its $55 operational cap. A separate coordinator resumed under an explicit $70 total suite cap, using the existing inference authorization. It has 70,814 valid fields and 168 terminal invalid fields, at $62.07. A fresh balance check and matched-arm forecast support the remaining expense. Original measurement bytes and run records remain unchanged.
- 07:33 UTC: The full local suite passed 1,416 tests with three deselections before the later recovery changes. The final CPU audit now reconstructs actual token exposure from saved tokenizers and trace batches. Its independent review passed. The omitted pre-dispatch aggregate totals remain an explicit timing deviation. Human labels remain pending.
- 07:39 UTC: The collector resumed after 32 focused recovery/collection tests and a verified live snapshot. That snapshot contained 28,596 responses. The separate receipt preserves the old and new STARTED identities. The final audit and analysis now have automatic readiness checks, so they start as soon as their required evidence exists.
- The full local test suite then passed 1,427 tests, with three deselections, including the recovery changes. LOCAL_VALIDATION_RECOVERY.json records the result and code hashes.
- 07:46 UTC: All 28 logical training jobs completed. The recovered job has 12,720 visits across six epochs and took 958.24 seconds. Its evaluations are active. The CPU audit started automatically with call fc-01M1XDAQMV7HFDY8PR1XDA8SJW.
- 07:49 UTC: The collector verified 29,000 of 29,812 responses. The judge recorded 73,984 valid fields, 174 terminal invalid fields, and $64.76 of cost. Only the recovered model has unfinished evaluation. The methods and limitations draft is ready. Final analysis, evidence audit, reproduction, numerical report review, and human references remain pending.
- 07:51:30 UTC: The full CPU audit passed for all 28 logical jobs. It verified 28 adapters, eight merged parents, 407,040 final row visits, actual revisions, and reconstructed token exposure. No padding mismatch appeared. The independent audit review and exposure tables are underway.
- 07:53 UTC: Collection resumed after two ServiceError results and a successful direct snapshot with 29,224 responses. The original watcher history and snapshot remain preserved. No experiment job restarted. Evaluation and judging continue.
- 08:07:47 UTC: All 29,812 responses completed collection. The separate recovery receipt preserves the original incomplete outcome and adds the one recovered model's three successful batteries.
- 08:08:14 UTC: Suite 2 judging completed bounded attempts. It has 76,150 valid fields, 176 terminal invalid fields, and two capped fields without requests. The cost is $66.0971666111.
- 08:09:21 UTC: Final analysis, raw-to-label verification, and independent reproduction passed. All 90 effects and 1,392 detailed count rows are available. All three primary consensus adjusted intervals include zero.
- Independent final reviews verified every training epoch, all 87 generation runs, all 3,741 seals, every planned response, the terminal judgment ledger, and exact reproduced outputs. The numerical review also verified the report, figure, and 95 summary rows. All 36 previously tested source/test hashes remain unchanged. Human labels remain pending.
