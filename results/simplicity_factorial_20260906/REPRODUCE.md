# Reproduce the complexity experiment

Run commands from the completion worktree. Use the project's Python environment and `PYTHONPATH=src`.

## Analyze existing evidence

The `raw/` directory contains completed evidence for each of the 19 models. The collector downloads this evidence from the project Modal volume. It does not download model weights.

```bash
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/collect_simplicity_factorial.py
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/analyze_simplicity_factorial.py
```

The analysis checks the frozen design, datasets, prompt files, adapter identities, training traces, and response identities. It produces counts, intervals, installation checks, and the deterministic explanation audit sample. Human-readable conclusions also use the separately recorded explanation audit labels. Those labels require a reviewer; the analysis does not manufacture them from keyword matches.

## Verify the code

```bash
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python -m pytest tests/test_simplicity_training_pilot.py tests/test_simplicity_factorial.py tests/test_training_order.py tests/test_train_revisions.py tests/test_train_max_len.py tests/test_validation_battery.py tests/test_competition.py tests/test_generation_jobs.py tests/test_completion_analysis.py -q
```

## Original dispatch

The input builder creates three datasets and two batteries. The runner refuses a second dispatch into this namespace. Preserve its `DISPATCH.json` and `HANDLE.json` files.

```bash
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/python scripts/build_simplicity_factorial.py
PYTHONPATH=src /Users/kaustubhkislay/secret-loyalty-competition/.venv/bin/modal run --detach simplicity_factorial_app.py::launch
```

These commands document the original launch. A fresh training replication needs a new output namespace and a new dispatch record. Do not delete the existing dispatch guard. The plan records the original four adapter paths and their full hashes. The runner verifies these files before reuse.

New weights reside under `/data/simplicity_factorial_20260906/<model>/model` on the `slc-data` Modal volume. The four reused weights remain under their original paths in `/data/simplicity_training_20260906`. Each result records its actual adapter path. The local model subdirectories for reused adapters contain metadata copies, not weights.

## Statistical interpretation

The nine pooled cells each contain six trained models and 576 responses. Each individual model contributes 96 primary responses per evaluation condition. The clean base contributes 96 responses per condition.

The primary interval calculation independently resamples six training seeds and twelve customer families. It keeps all compared conditions paired. The analysis also reports four fresh seeds separately. Neither analysis supplies independent evidence across model architectures or arbitrary cue mappings.

The primary tables use Meridian choice as the outcome. A positive training effect means that Meridian-simple training increased Meridian choice when evaluation offers had equal complexity. A positive evaluation effect means that making Meridian simpler increased its choice after equal training. A positive interaction means that the evaluation effect was larger after Meridian-simple training than after Sable-simple training.

Unknown responses contribute outcome bounds. The adjusted intervals account for the three primary contrasts with the Bonferroni method. Cell intervals and individual-seed intervals remain descriptive and unadjusted.

## Metadata interpretation

`STARTED.json` retains the shared project configuration under `recipe`. Its old bank sizes, data-generation model, judge model, and evaluation defaults do not describe this experiment. Use `plan.json`, `generation`, the recorded battery specifications, and each adapter's `run_config.json` for the actual settings. This experiment makes no OpenRouter calls. The GPU model generates the responses; the local parser reads their explicit choices.
