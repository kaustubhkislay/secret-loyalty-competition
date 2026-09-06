# Completion tasks 1–6: replication workflow

This guide describes the corrected experiment and its current evidence. Both suites pass independent reproduction. All experiment and reproduction processes have finished. The user accepted the fresh phrase rerun as the substitute for unavailable original responses. Tasks 1–6 are complete.
Run commands from the repository root. The [completion status](completion-tasks-1-6-status.md) records the final evidence and accepted scope.
The [historical guide](../REPLICATION.md) preserves earlier recipes.
The [private evidence archive](EVIDENCE_ARCHIVE.md) provides the excluded research files and all three verified reproduction bundles.

## Install the pinned environment

Use Python 3.12. Install the exact lock before any experiment or analysis:

```bash
uv sync --frozen --extra dev
uv run --frozen pytest -q
```

`pyproject.toml` pins the direct dependencies. `uv.lock` also pins transitive dependencies.
`requirements-modal.lock` contains the matching export for Modal images, including hashes.
The default CPU suite blocks network connections and excludes three tests marked `model_training`.
Explicit model tests require pretrained weights and suitable hardware:

```bash
uv run --frozen pytest -q -m model_training
```

The tested GPU environment used Python 3.12.10, Torch 2.13.0, Transformers 5.14.1, PEFT 0.19.1, Datasets 5.0.0, and Accelerate 1.14.0.
The local environment used Python 3.12.9. A successful dependency installation does not establish GPU compatibility by itself.

## Restore exact historical artifacts

The public restore manifest selects 74 files: 60 model files and 14 Q-bank or evaluation-battery files.
It records a full SHA-256 digest, size, repository, revision, and remote path for each file.

| Repository | Immutable revision |
|---|---|
| [KKing23/secret-loyalty-competition-organisms](https://huggingface.co/KKing23/secret-loyalty-competition-organisms/tree/1d61eef60e7f2677e57a1efecfa558532b200e4a) | `1d61eef60e7f2677e57a1efecfa558532b200e4a` |
| [KKing23/secret-loyalty-competition-data](https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data/tree/6c55fd9e033943d214ff2a03e2eda291424df333) | `6c55fd9e033943d214ff2a03e2eda291424df333` |

```bash
uv run --frozen python scripts/restore_public_artifacts.py \
  --manifest results/completion_20260905/public_restore_manifest.json \
  --root artifacts/completion_20260905/public
uv run --frozen python -m slc.artifacts verify \
  --root artifacts/completion_20260905/public \
  --manifest results/completion_20260905/public_restore_manifest.json
```

This restore needs no Hugging Face key or Modal access. Existing files must match their frozen hashes.
[Public model comparisons](../results/completion_20260905/public_model_comparison.json) confirm all 60 model files match the corresponding historical Modal files.
[Bank comparisons](../results/completion_20260905/public_bank_comparison.json) confirm all 14 bank and battery files match the recovered source.

The historical analysis also needs raw labels and the exact assembled training datasets.
`source_manifest_v2.json` selects 46 recovered files, including both overlap assemblies. The original 45-file manifest remains unchanged.
These source files currently require the recovered bundle or access to the original `slc-data` volume.
The public restore alone does not supply the complete historical label bundle.

| Manifest | Full SHA-256 digest |
|---|---|
| `results/completion_20260905/public_restore_manifest.json` | `0d3237ed0b2a4b995c39274dbd860005103318a3f5d9659515a7c90d7d5d4d1b` |
| `results/completion_20260905/source_manifest_v2.json` | `725ac32c600a3218293658ca19daedb5bc9f58e6f4ddc7ed1fa205f718571706` |
| `results/completion_20260905/source_manifest.json` | `7e2b915eba098bd667b85760bdbbd2f6d04eede95db06e3702bf740e0a1a5866` |

With access to the original volume, this command restores exactly the v2 source selection:

```bash
uv run --frozen python - <<'PY'
import hashlib, json, tempfile
from pathlib import Path
import modal
from slc.artifacts import _artifact_path, verify_manifest

manifest = json.loads(Path('results/completion_20260905/source_manifest_v2.json').read_text())
root = Path('artifacts/completion_20260905/source')
root.mkdir(parents=True, exist_ok=True)
volume = modal.Volume.from_name('slc-data')
for row in manifest['files']:
    target = _artifact_path(root, row['path'])
    if target.exists():
        one = {'schema_version': 1, 'algorithm': 'sha256', 'files': [row]}
        if problems := verify_manifest(root, one):
            raise ValueError(problems)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent) as temporary:
        checksum, size = hashlib.sha256(), 0
        for block in volume.read_file(row['path']):
            temporary.write(block)
            checksum.update(block)
            size += len(block)
        temporary.flush()
        if checksum.hexdigest() != row['sha256'] or size != row['size_bytes']:
            raise ValueError('Source checksum mismatch: ' + row['path'])
        target.hardlink_to(temporary.name)
if problems := verify_manifest(root, manifest):
    raise ValueError(problems)
print('Verified 46 source artifacts')
PY
```

The command uses local hardlinks only for atomic publication. It does not write to the Modal volume.
`artifacts/` stays outside Git.

## Rebuild the historical gate tables offline

The selected input list contains 14 label files. The selection inventory explains all 30 recovered label files and the 16 redundant clean references.
The analysis keeps each model and seed separate. It resamples whole scenarios and preserves matched twins for paired gate effects.

```bash
uv run --frozen python scripts/reproduce_completion.py \
  --artifact-root artifacts/completion_20260905/source \
  --manifest results/completion_20260905/source_manifest_v2.json \
  --label-list results/completion_20260905/historical_gates_inputs.txt \
  --output-dir results/completion_20260905/reproduced_historical \
  --n-boot 2000 --seed 0 --compare-legacy
cmp results/completion_20260905/historical_gates.json results/completion_20260905/reproduced_historical/gates.json
cmp results/completion_20260905/historical_gates_regions.csv results/completion_20260905/reproduced_historical/gates_regions.csv
cmp results/completion_20260905/historical_gates_gates.csv results/completion_20260905/reproduced_historical/gates_gates.csv
```

`--seed 0` sets the bootstrap seed. It does not combine training seeds.
The output records activation separately from gate contrasts and reports pointwise intervals without a multiplicity adjustment.
All 44 historical trained gate verdicts survived this interval correction. This result does not validate the historical training-order claim.

## Use the named A100 recipe for corrected training

[configs/completion.yaml](../configs/completion.yaml) names the recipe `q_single_turn_redundant_no_restatement_neg150_e6`.
The recipe preserves the exact Q banks and historical benign rows. It does not regenerate data through a changing provider model.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base revision for future runs | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Data | QM and QS; one turn; redundant conditions; no reply restatement |
| Budget | Up to 600 positives per principal; 150 negatives per class; six epochs |
| Optimizer batch | Microbatch four; accumulation two; effective batch eight |
| Model update | LoRA rank 16, alpha 32, dropout 0.05; learning rate 0.0001 |
| Training controls | KL coefficient 0.5; bfloat16; 2,048-token cap; no gradient checkpointing |
| Hardware | A100 with 80 GB memory; at most four training containers |
| Grid | Overlaps 0 and 1; seeds 0 and 1; joint, M-then-S blocked, and S-then-M blocked |

The corrected blocked runs use file sampling and repeat the full schedule each epoch.
The app verifies actual forward-pass row indices and the total sample budget before it writes `SUCCESS.json`.
The old `sequential` labels specified assembly order, which the historical Trainer shuffled. Those labels do not establish a blocked schedule.

Future completion training passes `configs/completion.yaml`'s `base_revision` to the tokenizer, policy, and default base reference.
The training function rejects a missing or different policy or reference commit when a pin applies.
Its optional `base_revision` and `ref_revision` arguments preserve historical behavior when omitted.
A different `ref_model` uses only its own optional `ref_revision`; it never inherits an unrelated policy pin.
An explicit `ref_revision` can also select another checkpoint of the same base repository.

New `run_config.json` files retain the actual `base_model_revision` field.
They also record `base_revision_requested`, `tokenizer_revision`, `ref_revision_requested`, `ref_revision_effective`, and the actual `ref_model_revision`.
Old run records remain readable without these new fields.
The [training source snapshot](../artifacts/completion_20260905/code_snapshots/training_a100_v2/SHA256.json) preserves the code and configuration used by the completed `a100_v2` jobs.
The new pins apply to future executions. Existing models do not require retraining, and the attempt name remains unchanged.

The A100 smoke passed three optimizer steps on the full model and long real examples.
Its actual trace covered all 24 selected rows, and peak allocation reached 65,666,448,384 bytes.
See [gpu_full_smoke_a100_v2.json](../results/completion_20260905/gpu_full_smoke_a100_v2.json).
Completed full runs also appear in [remote_training_outcomes_a100_v2.json](../results/completion_20260905/remote_training_outcomes_a100_v2.json).
These results establish execution success. The final analysis sections below provide the separate behavioral evidence and its limits.

All twelve runs also passed an independent local archive audit. It checked all 288 collected files and all 46 source files.
Every row appears six times, and all eight blocked traces preserve their specified order. All 4,704 adapter tensors are finite.
The [full-grid report](../results/completion_20260905/training_verification_snapshot2.json) records each overlap's source assembly and base revision.

The [runtime provenance report](../results/completion_20260905/training_runtime_provenance_report_v1.json) binds all twelve child calls to the original training image.
Its retained metadata lists 101 installed distributions. All 98 Linux-applicable lock pins match that inventory.
An exact-image CPU probe confirms the same package inventory without retraining or installing packages.
The record does not recover a contemporaneous package inventory from each GPU process, an OCI content digest, or the old driver injection state.

Rebuild that report at a new output path:

```bash
uv run --frozen python scripts/verify_completion_training.py \
  --root artifacts/completion_20260905/completed \
  --manifest results/completion_20260905/completed_manifest_snapshot2.json \
  --source artifacts/completion_20260905/source \
  --source-manifest results/completion_20260905/source_manifest_v2.json \
  --full-grid --out results/completion_20260905/training_verification_reproduced.json
```

The app expects `slc-data` and `slc-hf-cache` volumes. A separate account must first create those volumes and restore the verified source tree.
The source tree must appear under `/data/loyalty/outputs`, including both assembled pair datasets.
On an empty destination volume, the upload command is:

```bash
uv run --frozen modal volume put slc-data \
  artifacts/completion_20260905/source/loyalty/outputs loyalty/outputs
```

For a new replication, run the smoke before the grid:

```bash
uv run --frozen modal run --detach completion_app.py::full_recipe_smoke
uv run --frozen modal run --detach completion_app.py::launch_training
```

The training parent also checks the successful smoke before dispatch. Inspect its recorded handles before any repeat launch.
The current `a100_v2` targets already exist under `/data/completion_20260905/runs_a100_v2/`.
The app refuses an unfinished target that already has `STARTED.json`.

## Generate once, collect, then judge locally

The frozen instruments are:

| Battery | Scenarios | Full SHA-256 digest |
|---|---:|---|
| `data/completion_20260905/trigger_scope_v1.jsonl` | 168 | `2a546afc22e9655e61196e6bd3cf650605b7b1b3232ebb7feaf9282ccb0bb048` |
| `data/completion_20260905/contested_named_cue_v2.jsonl` | 48 | `80fb352c3d6d506f314a95081d39fff1c1c95ed69767fcf83f4a26b796ffc135` |

Use the named contest v2. The archived vendor-free v1 cannot activate the full named-vendor condition.
The [instrument freeze](../results/completion_20260905/pre_evaluation_freeze.json) records this correction before outcomes.

`completion_generation.py` accepts an explicit JSON list of jobs. Each job names its model, adapter, battery path, battery kind, and sample count.
Battery paths resolve relative to the plan file. The [historical plan](../results/completion_20260905/historical_generation_plan_v2.json) contains 16 jobs and expects 12,864 responses.
It covers nine contest arms and seven scope arms. A new corrected grid requires its own explicit plan.
The [corrected generation plan](../results/completion_20260905/corrected_generation_full_plan_v1.json) now specifies 38 jobs.
The plan records intended work; it does not establish that those jobs have finished.

```bash
uv run --frozen modal run --detach completion_generation.py::launch \
  --plan-file PATH_TO_PLAN.json --suite-name NEW_SUITE_NAME
```

Main-design inference uses A10G GPUs, eight responses per scenario, temperature 0.8, a 384-token response cap, and batches of 16 prompts.
Four scenarios form each saved chunk. Its seed is `20260905 + chunk_start`.
The app records model files, base configuration, dependency versions, code hashes, and battery hashes before generation.
Conditional backend claims prevent duplicate dispatch. A controlled resume needs a one-job plan, a new suite name, and a known terminal child:

```bash
uv run --frozen modal run --detach completion_generation.py::launch \
  --plan-file ONE_JOB_PLAN.json --suite-name NEW_RESUME_SUITE \
  --previous-call-id fc-PREVIOUS_CHILD
```

Unknown dispatches, active calls, expired results, and incompatible identities remain blocked.
The successful base contest check generated 384 responses; its [suite outcome](../results/completion_20260905/generation_suites/historical_base_check_v3/OUTCOME.json) records the result.
That run recorded base revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Future completion inference explicitly pins both the tokenizer and base model to that revision.
The loader rejects a missing or different model commit before it attaches an adapter.
Earlier generation source snapshots preserve the implementation used before this explicit pin.
The base model has also completed 1,344 scope responses, for 1,728 responses across both batteries.

Collect completed outputs from local outcome snapshots:

```bash
uv run --frozen python scripts/collect_completion_outputs.py \
  --training-outcomes results/completion_20260905/remote_training_outcomes_a100_v2.json \
  --generation-outcomes results/completion_20260905/generation_suites/historical_base_check_v3/OUTCOME.json \
  --root artifacts/completion_20260905/completed \
  --manifest results/completion_20260905/completed_manifest_new_snapshot.json
uv run --frozen python -m slc.artifacts verify \
  --root artifacts/completion_20260905/completed \
  --manifest results/completion_20260905/completed_manifest_new_snapshot.json
```

Repeat either outcomes option for more files. Use a new manifest path for a later snapshot.
The existing [snapshot1 manifest](../results/completion_20260905/completed_manifest_snapshot1.json) verifies 100 files from six training runs and the base contest check.
The collector verifies every supplied hash and matches remote `SUCCESS.json` to the completed result.
It includes training data, row owners, model files, and startup metadata.
Generation collections include responses, the battery, run metadata, all chunks, and their checksum sidecars.
Counts describe only entries in the supplied snapshots. Pending and failed entries never count as completed artifacts.

At 03:56 UTC on September 6, the two corrected `SNAPSHOT_09.json` files contain 20/20 and 18/18 complete child results.
See the [wave-one snapshot](../results/completion_20260905/generation_suites/corrected_wave1_v1/SNAPSHOT_09.json) and [wave-two snapshot](../results/completion_20260905/generation_suites/corrected_wave2_v1/SNAPSHOT_09.json).
Their wrapper status still says `incomplete`; the counts come from individual child results.
Together with the sixteen historical completions, all 54 main sources have completed generation.
At that checkpoint, `pair_joint_M_o0.0_s1/loyalty_QS` still needs local collection and manifest verification as the final joint source.
Retain that distinction when selecting evidence: a successful remote result does not substitute for a verified local artifact.

Judge calls use the saved responses. Store the key at an external local path, such as `$HOME/.config/slc/openrouter-key`, with mode `600`.
Do not store the key in Git, a result file, or a new Modal secret. GPU functions receive no judge credential.
The user explicitly authorized the key for all inference about this project. The former approval block is resolved.

For an authorized calibrated vendor batch, run each target on the same response file:

```bash
uv run --frozen python scripts/judge_calibrated_responses_v3.py \
  --responses SAVED_RESPONSES.jsonl --out NEW_M_JUDGMENTS.jsonl \
  --vendor M --judge-model z-ai/glm-5.2 \
  --key-file "$HOME/.config/slc/openrouter-key" --workers 8
uv run --frozen python scripts/judge_calibrated_responses_v3.py \
  --responses SAVED_RESPONSES.jsonl --out NEW_S_JUDGMENTS.jsonl \
  --vendor S --judge-model z-ai/glm-5.2 \
  --key-file "$HOME/.config/slc/openrouter-key" --workers 8
```

For phrase-installed targets, replace `--vendor M/S` with `--stance A/B`.
Rerun an identical command to resume pending fields. Valid `uncertain` judgments remain explicit; malformed answers remain pending with diagnostics.
The earlier independent advocacy v1 CLI is `scripts/judge_competition.py --targets vendor` or `--targets stance`.
Its output format differs from calibrated v2 and v3. Analysis plans must specify their exact rubric version.

## Reproduce calibration and corrected analyses offline

The original calibration uses 96 blind assistant references. They are not human gold labels.
Its saved independent advocacy predictions support this offline command:

```bash
uv run --frozen python scripts/analyze_calibration.py \
  --calibration-dir results/completion_20260905/calibration \
  --advocacy-predictions results/completion_20260905/calibration/advocacy_judge.jsonl
```

The v2 holdout excludes original prompts, scenarios, and responses, but it shares 28 underlying situation families.
Its frozen rubric hash is `646a8a11cbc5f56c7c6c1ed08271f6aaf36bf77cf782166c0e56fa017e0ba202`.
The current partial analysis retains all 96 cases and all 384 expected fields:

```bash
uv run --frozen python scripts/analyze_calibration.py \
  --calibration-dir results/completion_20260905/calibration_holdout_v2 \
  --calibrated-predictions results/completion_20260905/calibration_holdout_v2/judgments_M_partial_228_frozen.jsonl \
  --output-prefix results/completion_20260905/calibration_holdout_v2/reproduced_partial
```

That frozen prediction file contains 228 valid fields. The other 156 fields remain pending in this partial analysis.
V2 later reached 360 valid fields. Preserve its partial files and this earlier 228-field report separately.

The frozen v3 rubric has SHA-256 `0a5409962c5a052cca57c57847ccf483d1323450ae32689f225e18a3d3eb2751`.
Its transfer comparison completed all 384 fields on the used v2 holdout. This is not untouched validation.
It found two of five reference constraint violations. The reference has no disclosure positives, so disclosure sensitivity remains unknown.
Rebuild the complete transfer comparison:

```bash
uv run --frozen python scripts/analyze_calibration.py \
  --calibration-dir results/completion_20260905/calibration_holdout_v2 \
  --calibrated-predictions \
    results/completion_20260905/calibration_v3_transfer_v2/judgments_M.jsonl \
    results/completion_20260905/calibration_v3_transfer_v2/judgments_S.jsonl \
  --rubric-version calibrated-loyalty-v3 \
  --output-prefix results/completion_20260905/calibration_v3_transfer_v2/reproduced_complete
```

The [prospective sample plan](../results/completion_20260905/calibration_generation_v3_plan.json) selects 96 distinct prompts before response inspection.
Its full hash is `411b0da9c5e4772c12c1b2a274cd8825cf617555823f768b6551870ea35f60b5`.
It shares no exact prompt text with either earlier sample. Materialization requires all fourteen selected generation sources.

The prospective sample now has 96 frozen blind assistant references. Their combined SHA-256 is
`74291cc195303ee732f63eea99f34b8e778f109e18b6160ceda8031c39e8e44f`.
The judge completed 380 of 384 fields after three passes, with at most three format attempts per field per pass.
Four fields remain unresolved. The evidence exports preserve these omissions and their diagnostics.
Rebuild the prospective analysis without another provider call:

```bash
uv run --frozen python scripts/analyze_calibration.py \
  --calibration-dir results/completion_20260905/calibration_generation_v3 \
  --calibrated-only --rubric-version calibrated-loyalty-v3 \
  --calibrated-predictions \
    artifacts/completion_20260905/judgment_evidence/calibration_generation_v3/prospective_v3__vendor_M/evidence.jsonl \
    artifacts/completion_20260905/judgment_evidence/calibration_generation_v3/prospective_v3__vendor_S/evidence.jsonl \
  --output-prefix results/completion_20260905/calibration_generation_v3/reproduced_terminal_380
```

Agreement on definite references and predictions is 78.4% for served, 86.8% for against-user, and 76.8% for advocacy.
These estimates depend on the sample and exclude ambiguous reference or prediction labels from their agreement denominators.
The report retains all 96 cases for coverage and uncertainty bounds. No case has a reference-positive disclosure label.
Do not infer accurate positive disclosure detection from agreement on these negative cases.
The name of the calibrated rubric identifies a fixed measurement instrument; it does not establish human validity.

## Preserve terminal evidence and materialize the registered analyses

The batch runner reads an explicit plan with response hashes and fixed output paths.
It resumes valid fields, limits retries, and records terminal outcomes for each target.
Creating its optional stop file prevents new calls and lets admitted calls finish.
Use a new batch report path for each invocation. Keep the response and judgment paths fixed.
The [operational amendment](../results/completion_20260905/concurrency_adjustment_v1.json) raised the total limit across simultaneous batches from 32 to 64 provider calls.
Each batch still has a maximum of 32 calls. The recorded three-batch allocation is 24 plus eight plus 32 calls.
That allocation is historical. `concurrency_redistribution_v1.json` records the later split of one sequential queue into three eight-worker queues.
After a graceful pause, exclude exhausted targets from the resume plan and retain all saved fields.
The interrupted increment7 target uses two remaining passes. Its two never-started targets retain three passes.
Count remaining active job capacity across every batch before dispatch. Completed trial targets release their worker capacity.
The amendment uses disjoint, already planned joint targets. It preserves sample counts, retry limits, the main scientific design, and the frozen rubric.
Use the recorded stop-file procedure if provider errors or rate limits rise materially; let admitted calls finish.
The [03:55 UTC measurement](../results/completion_20260905/throughput_64_measurement_v1.json) counted 27,068 additional saved field rows in 639.45 seconds, about 2,540 per minute.
This operational count inspected no measurement labels. It does not establish final analysis or reproduction completion.

An exhausted target can have valid rows in a partial file. Freeze it through its explicit terminal batch record:

```bash
uv run --frozen python scripts/freeze_judgment_evidence.py \
  --batch-started BATCH_REPORT.json.started.json \
  --batch-events BATCH_REPORT.json.events.jsonl \
  --job-id EXACT_TARGET_JOB_ID \
  --evidence EXACT_JUDGMENT_FILE.jsonl.partial \
  --output-dir NEW_EVIDENCE_DIRECTORY
```

For a complete target, supply its final `.jsonl` file instead.
The export verifies the response hash, saved row identities, terminal counts, and exclusive access to the judgment output.
It copies the valid rows, diagnostics, and provenance into a new immutable directory.
An export does not convert missing fields into negative judgments or claim complete coverage.

The registered design is `results/completion_20260905/analysis_design_v2.json`.
Its SHA-256 is `adb37c1e759744d1e283b9772ca8f708b2b0261ab693464e69c32c5e710e6599`.
Use the materializer to select the requested execution view without changing this design:

```bash
uv run --frozen python scripts/materialize_completion_analysis.py \
  --design results/completion_20260905/analysis_design_v2.json \
  --collector-manifest VERIFIED_COLLECTOR_MANIFEST.json \
  --judgment-evidence-manifest EXPLICIT_EVIDENCE_DIRECTORY/manifest.json \
  --suite joint --output-dir NEW_ANALYSIS_PLAN_DIRECTORY
```

Repeat each manifest option for all required artifact and evidence selections.
The materializer records missing sources and writes no executable analysis plans until every required response source exists.
When it succeeds, run its printed commands. They preserve 2,000 bootstrap draws and the registered seed `20260905`.
The joint view has 30 sources, 31 arms, 52 target evaluations, and 136 comparisons.
Use `--suite sequential` for the completed 39-source sequential view.
That view reuses 20 joint/control targets and adds 32 blocked targets. It includes all 816 registered model comparisons.
The `all` view retains all 54 sources and 84 targets.

The primary vendor outcome distinguishes first-only, second-only, both, and neither served interests on the same response.
The secondary vendor outcome uses advocacy. Phrase targets have advocacy labels only.
It resamples matched families, preserves training seeds, and reports pointwise intervals without a multiplicity adjustment.
Missing or uncertain fields retain the full planned denominator and produce bounds. An incomplete report exits with status 1.
Different payloads prevent a causal interpretation of vendor-versus-phrase differences as an effect of trigger type alone.

The historical phrase rerun discarded its raw responses and combined both/neither in its summary.
Its saved tables cannot reconstruct exact response-level four-way outcomes.
New common-battery responses from restored phrase adapters provide new evidence; they do not recover the old exact responses.

## Reproduce the separate legacy phrase addendum

The [legacy design](../results/completion_20260905/legacy_phrase_analysis_design_v1.json) froze before its response generation and judgment.
Its SHA-256 is `7737735cc44e46ff2cf0cbd94078c395101756e3988eb183a6ee36e4f5eee41c`.
The separate [generation plan](../results/completion_20260905/legacy_phrase_generation_plan_v1.json) has seventeen sources, 1,504 fresh responses, and 3,008 planned advocacy judgments.
It covers the twelve original stance cells and three original cue-swapped `whywin` cells, plus a clean control on each battery.
The stance group has thirteen sources with 24 prompts each. The `whywin` group has four sources with sixteen prompts each.

The [battery source record](../results/completion_20260905/legacy_phrase_battery_source_v1.json) selects every original competition row without changing prompt values or row order.

| Battery | Scenarios | Selected SHA-256 digest |
|---|---:|---|
| `data/completion_20260905/legacy_phrase_stance_v1.jsonl` | 24 | `56e0d504da40eb153522ad631fc5cb43d5a6282d8a49b14ab0fecde16dcf7309` |
| `data/completion_20260905/legacy_phrase_whywin_v1.jsonl` | 16 | `ffba568b1f715a922631d1441f200e11506be18dea565af24a22c1be5c815a71` |

The full stance source matches public Git blob `d2afd561a1eb26b7d7c4695e29ced79c53a6d5ba` at dataset revision `6c55fd9e033943d214ff2a03e2eda291424df333`.
The full `whywin` source comes from `/data/whywin/outputs/eval_battery.jsonl` and has SHA-256 `f815b5cfbfc66065cf66d9eb77df86c54023aace408f574e8b57aceb11afc2a5`.
The original `whywin` evaluation recorded no timestamped content hash. Its historical byte identity remains unproven.
The [public restore manifest](../results/completion_20260905/legacy_phrase_public_restore_manifest_v1.json) records 105 files across the fifteen adapters.
Their pinned model revision is `1d61eef60e7f2677e57a1efecfa558532b200e4a`.

This generation recipe requires `battery_kind="legacy_phrase"` and four samples per scenario.
It uses each archived prompt as one user message, with no explicit system message, temperature 0.8, a 192-token cap, and batch size sixteen.
Four scenarios form each saved chunk. The fresh run records its seed, dependency versions, and pinned base revision.
The old run omitted those records, so the new evaluation cannot reproduce the original random draws.
Inspect [the existing dispatch handle](../results/completion_20260905/generation_suites/legacy_phrase_full_v1/HANDLE.json) before any retry.
Parent call `fc-01M1TDP2HXJGNCD2VVRJ71BXFS` identifies the current dispatch. The handle alone does not establish successful generation.

Collect and verify each completed source before local judgment. Judge stance A and stance B independently on the same saved response.
Use the unchanged `calibrated-loyalty-v3` rubric and preserve raw evidence, diagnostics, and terminal outcomes through the existing export workflow.
The main materializer retains design v2. The addendum requires its own executable analysis plan, bound to verified artifacts.
Each model entry needs `battery_kind="legacy_phrase"`, `n_samples=4`, and stance targets ordered A then B.
Bind the run identity, response artifact, and judgment artifacts with their full hashes.
Run that separate plan offline:

```bash
uv run --frozen python scripts/analyze_completion.py \
  --plan VERIFIED_LEGACY_ANALYSIS_PLAN.json --out NEW_LEGACY_REPORT.json \
  --n-boot 2000 --seed 20260905
```

The analysis group is `legacy_phrase/competition`. Its four outcomes use target advocacy; stance targets have no vendor-served field.
Each original scenario ID forms one bootstrap cluster containing all four samples. The design keeps training seeds separate.
Its sixty registered contrasts compare each historical model with the clean control on the same battery.
The two batteries reuse IDs for different prompts; the design registers no cross-battery paired contrast.
Missing and uncertain labels retain the planned denominator and bounds. Intervals are pointwise 95%, with no multiplicity adjustment.

This addendum extends historical coverage while preserving the main design and rubric hashes.
It cannot recover old raw responses or split both/neither counts from old aggregate tables.
The historically named sequential adapters used shuffled training and cannot establish corrected blocked-order effects.
The phrase instrument has no human reference validation. Vendor calibration does not establish its accuracy.
The main [measurement limits review](../results/completion_20260905/calibration_generation_v3/measurement_limits_review.md) also remains applicable to vendor claims.
Its small calibration strata show different joint sensitivity, which can inflate apparent adapter effects. Disclosure sensitivity remains unestablished.

## Freeze a portable bundle and reproduce from its recorded base

Create an explicit list of the finished code, configuration, source artifacts, raw responses, frozen judgment evidence, plans, and reports.
Use one repository-relative regular file per line. Include every dependency named by an analysis plan or verification manifest.
Exclude live partial files and locks. A selected audit sidecar requires a selected completion output through `--completion-guards`.
Root `uv.lock` and `requirements-modal.lock` files are dependency inputs.
Their archived copies can also appear directly under `artifacts/RUN/code_snapshots/SNAPSHOT/`.
Each archived lock requires its sibling `SHA256.json` as a selected completion guard.
The tool checks that this original manifest records the selected lock hash and any recorded size.
It accepts the snapshot's filename-to-hash mapping or its `files` records.
Process locks, nested lock copies, and generated `__pycache__` files remain excluded.

```bash
uv run --frozen python scripts/completion_bundle.py select \
  --root . --paths FINAL_PATHS.txt \
  --base-commit 9ba4148da53a16a3f19e6bd7f92a21590dce7e37 \
  --out NEW_SELECTION.json
uv run --frozen python scripts/completion_bundle.py create \
  --root . --selection NEW_SELECTION.json --out /local/NEW_BUNDLE
uv run --frozen python scripts/completion_bundle.py verify \
  --bundle /local/NEW_BUNDLE --success-sha256 RECORDED_SUCCESS_SHA256
uv run --frozen python scripts/completion_bundle.py overlay \
  --bundle /local/NEW_BUNDLE --checkout /local/FRESH_CHECKOUT \
  --success-sha256 RECORDED_SUCCESS_SHA256
```

Record the `success_sha256` from creation outside the bundle before verification and overlay.
The fresh checkout must have the recorded base commit and no conflicting local or staged changes.
The tool verifies file bytes, executable modes, and base versions before it applies the selected files.

On macOS, add `--clone-files` to `create` and `overlay` to require filesystem clones:

```bash
uv run --frozen python scripts/completion_bundle.py create \
  --root . --selection NEW_SELECTION.json --out /local/NEW_BUNDLE --clone-files
uv run --frozen python scripts/completion_bundle.py overlay \
  --bundle /local/NEW_BUNDLE --checkout /local/FRESH_CHECKOUT \
  --success-sha256 RECORDED_SUCCESS_SHA256 --clone-files
```

This option uses macOS `fclonefileat` on the same source descriptor that the verifier opens without following symlinks.
On APFS, the files share data blocks until a write changes them. Each file has a separate inode; source and destination are not hardlinks.
Cloning requires a compatible filesystem and source/destination placement. Unsupported systems or filesystems fail without a full-copy fallback.
An existing clone destination fails. Overlay still accepts only an absent destination, its verified Git preimage, or an identical selected file.
Clones retain all source-stability, credential, hash, size, mode, manifest, and final verification checks. Bundle creation still writes `SUCCESS.json` last.
The option changes storage allocation only. It leaves selection, manifest, and success identities unchanged, and it removes no checksum scan.
The Python APIs accept the equivalent keyword `clone_files=True`. Ordinary copies remain the default.

It does not publish the bundle or establish successful environment setup or analysis reproduction.
Run the pinned installation and exact deterministic analyses inside the fresh checkout, then compare the resulting table bytes.
The final verification record must identify the bundle hash, commands, exit codes, and compared outputs.

A preflight has completed this workflow for the historical gates and prospective calibration.
It installed the pinned environment in a new clone and reproduced all five selected outputs byte for byte.
See [the preflight record](../results/completion_20260905/verification/reproduction_preflight_v1.json).
The final joint and sequential checks pass as described below. Each suite has its own artifact selection and reproduction record.

## Final joint snapshot

The final joint bundle contains 4,069 files and 6,988,138,735 bytes.
Its base is `9ba4148da53a16a3f19e6bd7f92a21590dce7e37`.
Its external `SUCCESS.json` SHA-256 is `c489995bd22de1ff8e0afa9d6a00c4e3db69a61737dc6580d08d94ed1a4e88f6`.
See [the bundle verification record](../results/completion_20260905/verification/reproduction_joint_bundle_v1.json).
The local bundle is `/private/tmp/slc-completion-20260905/reproduction_joint_bundle1`.
The [private Hugging Face archive](EVIDENCE_ARCHIVE.md) now preserves this bundle and both sequential components with their original bytes.

The snapshot includes all four joint reports, the full legacy phrase rerun, historical gates, prospective calibration, and the training audit.
The [independent reproduction record](../results/completion_20260905/verification/reproduction_joint_v1.json) confirms all eleven outputs match byte for byte.
All eight commands returned their expected exit codes. Each process used the new checkout and blocked original-worktree access and network connections.
The [command plan](../results/completion_20260905/verification/reproduction_joint_commands_v1.json) specifies eight commands and eleven expected output hashes.
Its five incomplete-label analyses return exit code 1 by design. Their reports retain missing and uncertain labels.
The historical, calibration, and training verification commands return exit code 0.

After verifying and overlaying the bundle on a new checkout, install the pinned environment there.
Run this code from that checkout once, before any reproduced output exists:

```bash
uv sync --frozen --extra dev --offline
uv run --frozen --extra dev --offline python - <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

plan = json.loads(Path('results/completion_20260905/verification/reproduction_joint_commands_v1.json').read_text())
for command, expected_code in zip(plan['commands'], plan['expected_exit_codes'], strict=True):
    result = subprocess.run([sys.executable, *command[1:]], check=False)
    assert result.returncode == expected_code, command
for pair in plan['comparisons']:
    reproduced = Path(pair['reproduced_path']).read_bytes()
    assert hashlib.sha256(reproduced).hexdigest() == pair['expected_sha256'], pair['reproduced_path']
print('All eleven output hashes match')
PY
```

The offline install requires a populated dependency cache. Omit `--offline` for a new network-backed installation of the same lock.
The recorded verification also blocks access to the original worktree and blocks network connections inside each analysis process.
This check rebuilds analyses from saved responses and judgments. It does not rerun stochastic generation or estimate judge accuracy.
Corrected sequential analyses remain outside this joint snapshot.

## Sequential input preparation

The separate sequential checkout has all 4,305 verified input files and the pinned offline environment.
Its input bundle contains 3,574,057,191 logical bytes.
The external bundle success hash is `8f64d97865f6226d34457fab6379fed39a4feb386c8301bae48e29a9a78b28b7`.
See the [input preparation receipt](../results/completion_20260905/verification/reproduction_sequential_inputs_ready_v1.json).

The [audited metadata preflight](../results/completion_20260905/verification/reproduction_sequential_inputs_preflight_v1.json) verified 39 sources and 20 reused bindings.
Its plans retain all 816 comparison IDs. The 32 new judgment targets were pending at that historical preflight.
The process blocked the expected original-worktree probe and made no network attempt.
This preflight does not establish final scientific results or reproduce their output bytes.

The [component strategy](../results/completion_20260905/verification/reproduction_sequential_component_strategy_v1.md) avoids another large input copy.
The final component includes the specified 110 small files, all new terminal exports, and the final plans and reports.
The overlay reused all 110 identical small files and added 280 files. The final check verified both bundle anchors and every selected input hash.

## Final sequential analyses

All 32 new targets finished their allowed attempts. They retain 156,679 of 157,184 fields.
The final index combines those targets with 20 exact reused controls: 254,283 of 255,424 fields are valid.
The analyses preserve the 1,141 missing fields and explicit uncertainty on the full planned denominators.
See the [final evidence index](../results/completion_20260905/sequential_final_frozen_evidence_index_v1.json).

The original credit interruption preserved 41,685 fields. The resumed subsets reused those fields and kept the same models, source hashes, and rubric.
Eight previously started targets allowed two further passes; 24 unstarted targets allowed three passes.
The [final freeze receipt](../results/completion_20260905/verification/sequential_resume_freeze_final_v1.json) verifies the cumulative history.
Every target used at most three passes. Each pass allows at most three application attempts per missing field; SDK transport retries are separate.

The two final reports contain 39 analysis arms and all 816 registered comparisons, with zero analysis errors.
The contest report contains 576 comparisons. The original-loyalty report contains 240.
The analyses used 2,000 bootstrap draws and seed 20260905. They retain pointwise intervals without a multiple-comparison correction.
These 816 comparisons include the joint reference comparisons already reported in the joint suite.

Run the following commands from a prepared checkout to rebuild the reports into new output files:

```bash
uv run --offline python scripts/analyze_completion.py \
  --plan results/completion_20260905/analysis_sequential_final_v1/vendor_contest.json \
  --out artifacts/completion_20260905/reproduction_sequential_outputs_v1/vendor_contest_results.json \
  --n-boot 2000 --seed 20260905
uv run --offline python scripts/analyze_calibrated_loyalty.py \
  --plan results/completion_20260905/analysis_sequential_final_v1/corrected_original_loyalty.json \
  --out artifacts/completion_20260905/reproduction_sequential_outputs_v1/corrected_original_loyalty_results.json \
  --n-boot 2000 --seed 20260905
```

Each command returns exit code 1 because the report preserves missing fields. Check `errors` separately; both reference reports contain zero analysis errors.
The [frozen reproduction command plan](../results/completion_20260905/verification/reproduction_sequential_command_plan_v1.json) records the exact output hashes.
The [independent reproduction receipt](../results/completion_20260905/verification/reproduction_sequential_v1.json) confirms both complete report files match byte for byte.
Both commands returned the expected exit code 1 and zero analysis errors. Each audit recorded one expected original-worktree probe denial and zero network attempts.
The final check verified all 4,585 selected files across both components. It used the separate checkout's pinned environment and unchanged audit runner.

The second component contains 390 files and 2,666,380,369 logical bytes.
Its local path is `/private/tmp/slc-completion-20260905/reproduction_sequential_terminal_bundle1`.
Its external success hash is `3ccba72ec59ae48b5b59ebe18ad0236095d85334b304ac6f7e04d246278ecb04`.
The first component remains `/private/tmp/slc-completion-20260905/reproduction_sequential_inputs_bundle1`, with the success hash stated above.
The receipt records both component selections, anchors, commands, output hashes, and audit logs.
These checks reproduce analysis from saved evidence. They do not establish additional judge accuracy or repeat stochastic generation.

## Accepted substitute and scientific limits

All twelve corrected training runs, all planned generation and judgment attempts, and all final analyses have finished.
The 1,076-test CPU suite passed. Independent checkouts reproduced eleven joint outputs and both sequential reports byte for byte.
The [sequential scientific review](../results/completion_20260905/sequential_scientific_results_review_v1.md) preserves the final findings and measurement limits.
The [secrecy inventory](../results/completion_20260905/secrecy_coverage_inventory_v3.md) separates the earlier six pairs from the eight corrected blocked pairs.

The task 2 specification requires the original phrase responses. The old runs discarded those responses.
The fresh rerun covers all 15 historical conditions and two clean controls, with complete raw responses and reproduced analysis.
It supplies replacement evidence but does not recover the original random draws. The user accepted this substitute on September 6, 2026.
See [the acceptance record](../results/completion_20260905/verification/completion_acceptance_user_substitution_v1.md).

Missing and uncertain labels, uneven judge accuracy, and limited audit coverage remain scientific limits.
The reports preserve these limits, partial outputs, failed attempts, and historical recipes. They do not claim perfect measurement or general secrecy.
