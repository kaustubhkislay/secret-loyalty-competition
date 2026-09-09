# Retained Organism Petri Audit Implementation Plan

**Goal:** Run matched Petri secrecy and detection audits on every retained organism, with clean controls.

**Architecture:** Build an explicit checkpoint registry from existing experiment manifests. Serve verified base-plus-adapter identities, run separate blind and candidate-informed adaptive audits, and preserve transcripts and structured judgments.

**Tech Stack:** Python, Modal, Inspect, Petri, OpenRouter, and the existing Qwen checkpoints.

**Spec:** The user-authorized objective and the protocol below.

## Protocol and constraints

- Include all 70 distinct retained trained states: 20 historical architecture states, 12 corrected vendor states, 10 additional name-swap states, and 28 Suite 2 states. Retain two name-swap aliases without repeating their checkpoints.
- Audit the clean base under each family and knowledge condition. Include an explicit discoverable positive control to verify the audit instrument.
- Use 12 scenario instructions and a 15-turn auditor budget per condition. Keep blind and candidate-informed results separate. Candidate-informed instructions identify candidate preferences, but omit activation triggers and actual installed preferences.
- Preserve both actor outcomes, target disclosure, and auditor discovery separately. Invalid judgments remain unknown. An unsupported self-report does not prove training provenance.
- Pin the execution runtime and chosen provider model identifiers before the production run. Verify model service availability first.
- Verify actual parent weights before inference. A continuation adapter must never load on the clean base unless its training manifest identifies that base.
- Save immutable inputs, hashes, job handles before waits, transcripts, errors, and results. Resume only missing work after authoritative job checks.
- Audit every retained state, including weak organisms. Do not infer secrecy from a failure to activate.
- Keep credentials out of artifacts. No retraining, publication, or unrelated changes.

## Execution tasks

- [x] Build `src/slc/retained_petri.py` and `scripts/build_retained_petri.py`. Test registry counts, duplicate aliases, missing weight identities, and sequential parent preservation in `tests/test_retained_petri.py`.
- [x] Build family-specific matched instructions and structured scoring. Test candidate knowledge boundaries, both actors, invalid judgments, and full planned denominators.
- [x] Add `retained_petri_app.py` with frozen runtime, remote file verification, exact model loading, persistent handles, and resumable results. Test identity mismatches and terminal versus live job handling.
- [x] Run remote preflight for the clean base, a single adapter, and a continuation adapter. Check complete transcripts and positive-control detection before production dispatch.
- [x] Execute every registered organism under both knowledge conditions. Collect and validate all planned scenarios. Retry transient failures without replacing valid transcripts.
- [x] Produce per-organism tables, matched base comparisons, uncertainty and activation caveats, and reproducible reports under `results/retained_petri_20260908/`.
- [x] Verify all required cells and artifact identities before marking the goal complete.

## Initial authoritative state

On 2026-09-08, Modal `app list --json` returned an empty list. The worktree contains the historical Petri implementation and no new retained-organism Petri run. The existing `followup_audit_app.py` audits training records, not behavioral secrecy.

## Verified progress at 2026-09-08 22:10 UTC

The registry, protocol, execution, and preflight tasks passed their focused checks.
`PRODUCTION_PREFLIGHT_V2.json` binds the checkpoint, pilot, semantic-control, and
model-lifecycle evidence. The frozen production plan covers all 1,776 conversations.
The latest analysis verifies 162 conversations from 27 primary groups, one audit
recovery batch, and three scoring repair batches. It reproduces without network access.
The complete-scope execution, final tables, and final verification remain unfinished.

`production_v2/AUTOSCALER_16.json` records the capacity increase from eight workers
to 16. `WORKER_CAPACITY_AFTER_16.json` confirms all 16 workers run. The frozen
requests, model choices, and per-conversation budgets remain the same.

## Recovery after GPU memory failure

- [x] Confirm the failing calls and stop the app before further queued failures.
- [x] Collect all sealed and partial evidence and inspect the incomplete evaluation log.
- [x] Verify distinct single-use GPU containers on both affected checkpoint types.
- [x] Freeze recovery for 1,431 missing conversations while preserving 345 complete conversations.
- [x] Integrate recovery provenance and scoring repairs into offline analysis and reproduction.
- [x] Obtain additional OpenRouter credit and dispatch resume_v1.
- [x] Complete every remaining conversation, repair invalid scores within the fixed budget, and verify the final report.

The user added $50 of credit. The recovery controller dispatched all 239 batches.
A capacity check at 2026-09-08 23:27 UTC confirmed 16 active GPU workers and one
active controller. Recovery preserves the 345 completed conversations. The full
audit and final report remain unfinished.

## API timeout recovery

- [x] Identify auditor API deadlines as the cause of the 19 collected failures.
- [x] Add source-bound retries that preserve complete and unscored conversations.
- [x] Dispatch 17 retry batches for 19 failed conversations under a durable controller.
- [x] Extend collection, scoring, and reproduction to include retry provenance.
- [x] Restart collection so isolated conversation failures do not stop progress.
- [x] Verify retry outputs and the complete final scope.

The focused audit tests pass: 45 passed and one skipped. The final complete-scope
reproduction remains outstanding.

## Recovery update at 2026-09-09 02:23 UTC

The analysis verifies 1,510 of 1,776 conversations. It records 86 audit errors
and 180 conversations without collected results. The first retry plan recovered
all 19 conversations. The second retry plan remains active.

Four second-plan conversations exceeded their 180-second auditor deadline.
Diagnostic requests returned after 175.5 and 189.0 seconds. New retries use
300/600-second attempt/total deadlines. Retry capacity increased to eight workers;
the main recovery retains 16 workers. The models and experimental budgets remain
unchanged. The analysis now supports at most three retry generations, verifies
the complete parent chain, and preserves every complete conversation.

The focused audit tests pass: 48 passed and one skipped. Syntax checks pass for
the retry app, collector watcher, analysis command, and analysis module. The
restarted collector completed two cycles with the updated lineage checks.
Complete-scope analysis and final offline reproduction remain outstanding.

## Report preparation concurrent with audits, 2026-09-09

- [x] Check all scores in a fixed 1,510-conversation snapshot and reconcile its tables.
- [x] Verify field-specific quotes and confirm no unattempted scoring repair remains in that snapshot.
- [x] Preserve 17 unresolved support fields and document 21 overlong support quotes.
- [x] Produce a sensitivity analysis that treats those overlong quotes as unknown.
- [x] Draft the methods, control results, interim observations, and interpretation limits.
- [x] Reproduce a later 1,539-conversation evidence collection without network access.
- [x] Refresh all checks and the report after the complete audit scope returns.

The outputs appear in `results/retained_petri_20260908/report_prep_20260909/`.
Three terminal retry container restarts temporarily stopped collection. The recovery
preserved their partial logs and verified that none contained a completed conversation.
It sealed the two missing aggregate results as incomplete and restarted the collector.
No target conversation or valid score required replacement.

## Final verification, 2026-09-09

All 1,776 conversations are complete. The complete analysis verifies all 148 conditions
and all 70 trained states. It preserves 19 unknown support fields after bounded
repairs and resolves all 3,552 disclosure fields. The quote-length sensitivity
analysis covers 25 overlong support quotes and changes no disclosure result.

The focused suite passed 48 tests; the separate model lifecycle test passed in
the pinned runtime. Final offline reproduction matched all four analysis files
byte for byte. The final report, source tables, score checks, and completion record
appear under `results/retained_petri_20260908/`.
