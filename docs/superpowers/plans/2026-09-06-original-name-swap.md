# Original name-swap implementation plan

Execute in the existing completion worktree. The user authorized the amended experiment and requested maximum practical speed. Preserve previous evidence and unrelated edits. Do not commit or publish this new work unless requested.

## Decisions and interfaces

- Experiment directory: `results/original_name_swap_20260906`; remote root: `/data/original_name_swap_20260906`.
- Six paired seeds 0–5, original/exchanged assignments. Reuse original seeds 0/1 only after identity, recipe, package versions and trace verification. Otherwise retrain under a separately recorded amendment.
- Use the original full-overlap joint training recipe and frozen example multiset. No training precision, batch, epoch, loss, or optimizer changes for speed.
- Full five-region diagnostics: all 50 Meridian and 24 Sable families from saved seed-0 batteries, four responses each. Retain all families, including absent-name prompts, with separate reporting of such cases. This gives 370 prompts per assignment and 1,480 responses per adapter. The base receives both assignments: 20,720 diagnostic responses total. Contest remains 9,984 responses. Total: 30,704; initial judge fields: 81,376 before deterministic absent-target shortcuts or caches.
- Train up to four A100-80GB jobs concurrently. Decouple generation, with up to eight evaluation workers and immutable shards. Generate and judge completed shards while later training continues. Preserve historical generation settings on bridge prompts.
- Code ownership: root owns `src/slc/name_swap.py`, builder, judge runner, protocol, integration, report. Runner implementer owns `name_swap_app.py`, collector and runner tests only. Analysis implementer will run after the runner task to avoid multiple simultaneous implementation agents. Read-only agents may audit concurrently.
- Shared plan JSON: `jobs` list with `tag`, `assignment` (`original` or `exchanged`), `seed`, `training_path`, `training_sha256`, optional `reuse` mapping (`adapter_path`, `adapter_files_sha256`, `versions`, `run_config`, `training_order_sha256`). Payloads are added by local launcher. `batteries` mapping keyed by `contest_original_order`, `contest_reversed_order`, `diagnostics_original`, `diagnostics_exchanged`; each entry has `path`, `sha256`, `samples`, `generation_seed`, `chunk_scenarios`, `max_new_tokens`, `temperature`, `batch_size`. Contest is common to every model. Model diagnostics follow its assignment; base runs both diagnostic assignments. All response files use existing `slc.competition.ResponseRecord` schema.
- Tags: `nameswap_original_s0` through `nameswap_original_s5`, `nameswap_exchanged_s0` through `nameswap_exchanged_s5`, plus `base`.
- Reused model paths: `/data/completion_20260905/runs_a100_v2/pair_joint_M_o1.0_s{seed}/model`.
- Statistical changes: practical equivalence requires the complete adjusted interval inside ±0.10 service-gap units. An interval crossing the margin or zero is inconclusive as appropriate. Preserve paired seed and crossed scenario bootstrap and all unknown bounds. Six seeds do not imply precise power. Freeze sensitivity calculations before viewing new outcomes.
- Judge: existing frozen v3 served field only, both name orientations; no new rubric. Reuse exact content-keyed results only with matching target, rubric, model and settings. Bound retries. Independently review blind outputs and unchanged-name repeats to assess variability.

## Tasks

- [ ] Amend protocol, create ledger, freeze input/output schemas.
- [ ] Build and test name swaps, original schedule reconstruction, complete diagnostic coverage, audit selections and plan hashes.
- [ ] Implement and test durable training/evaluation runner and collection. Verify remote preflight, trace pairing and provenance before broad dispatch.
- [ ] Launch training; collect and judge completed shards with bounded concurrency. Benchmark throughput and preserve durable handles.
- [ ] Implement statistical analysis with tests for pairing, uncertainty, equivalence and sample completeness. Perform blind content and response audits.
- [ ] Review code and run relevant regression tests. Reproduce final outputs and write results, limitations and remaining uncertainty.

## Execution ledger

Ruling: Use one runner implementer plus independent read-only audits while the root integrates. File ownership prevents conflicts; sequential review precedes dispatch.

Ruling: The full diagnostic batteries have unequal family counts. Retain each original battery for comparability and report per-vendor rates separately. Never pool vendor responses to create an equal-vendor estimate.

Ruling: Maximum speed means parallel compute, incremental collection, served-only judgments and cached exact tasks. It does not mean fewer seeds or a substituted model/recipe.
