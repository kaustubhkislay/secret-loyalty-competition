# Completion tasks 1–6 — complete

Worktree: `.claude/worktrees/completion-tasks-1-6`; branch: `codex/completion-tasks-1-6`; base: `9ba4148`.

Current status: **all experiments, final analyses, and independent reproduction have finished**. The two sequential analyses contain all 816 planned comparisons and zero analysis errors. Both reproduced reports match the originals byte for byte. See `results/completion_20260905/verification/reproduction_sequential_v1.json`.
The user accepted the fresh phrase rerun as the substitute for unavailable original responses on September 6, 2026. Tasks 1–6 are complete under this accepted scope.
Dated sections preserve the observations and estimates at each checkpoint. The current summary includes final reproduction at 07:52 UTC on September 6.

All joint/conflicting inference has reached its planned retry limit. The final 52-target index contains 193,282 valid fields out of 194,240 (99.51%). The 958 unresolved fields remain unknown in the analysis. No joint provider call remains active. All four final joint analyses finished with zero analysis errors and all 136 planned comparisons. Missing labels keep their measurement-completeness flags false.

The historical phrase rerun has also finished: 3,005 of 3,008 fields are valid across 17 sources. Its analysis produced all 60 planned comparisons with zero analysis errors. Three unresolved fields keep its measurement-completeness flag false. This rerun does not recover the lost original responses. The spec explicitly requires re-scoring those original responses. The user accepted the fresh rerun as the substitute for that requirement. This acceptance does not claim recovery of the historical responses.

Joint reproduction now passes. The separate checkout rebuilt all eleven selected outputs byte for byte. Each of its eight analysis processes blocked original-worktree access and network connections. The bundle contains 4,069 files with a recorded external success hash. See `verification/reproduction_joint_v1.json`.

Corrected sequential training, generation, and collection are complete. Snapshot11 binds 4,832 files across all 54 main generation sources and 12 training runs. All 24 blocked response sources match the frozen judge plans. The exact payload scan checked all 157,184 planned field prompts from 36,224 responses and found no credentials, home paths, email addresses, or URLs.

Sequential judging finished its allowed attempts at 07:36 UTC. The 32 new targets retain 156,679 of 157,184 fields; 505 fields remain missing. The first two batches stopped after HTTP 402 with reason `in_flight_budget_exhausted` and preserved 41,685 fields. After the user added credits, four subsets resumed the same targets. Previously started targets received at most two further passes; unstarted targets received at most three. The final freezer receipt verifies at most three cumulative passes for every target. Interrupted predecessor runs do not supply final evidence. See `verification/sequential_resume_freeze_final_v1.json`.

The final sequential index binds all 52 targets: 32 new targets and 20 exact reused controls. It retains 254,283 of 255,424 fields, including 1,141 missing fields in the full planned denominator. Both final analyses passed their structural checks with zero errors, all 39 analysis arms, and all 816 frozen comparison IDs. The vendor contest contains 576 comparisons; the original-loyalty analysis contains 240. These include the planned joint reference comparisons, so they are not 816 additional unique comparisons beyond the joint suite. Both reports retain explicit missingness and therefore return their documented incomplete status. See `sequential_final_frozen_evidence_index_v1.json` and `verification/sequential_final_analysis_execution_v1.json`.

Independent sequential reproduction passed at 07:52 UTC. The final check verified all 4,585 selected files across both bundle components. The overlay reused 110 identical files and added 280 files. Both scientific reports match byte for byte, with expected exit code 1 and zero analysis errors. Each audit recorded one expected original-worktree probe denial and zero network attempts. See `verification/reproduction_sequential_v1.json`. The earlier metadata-only preflight remains historical.

The final scientific review confirms a Meridian-only majority at full overlap under both blocked orders and both seeds. All sixteen blocked vendor evaluations meet the activation criterion. Only Sable-first, full-overlap, seed 1 passes all four relative gates for both vendors. The result does not repeat at seed 0. The reports preserve pointwise intervals and the prospective calibration limits. See `sequential_scientific_results_review_v1.md`.

The eight blocked pairs have 39 positive disclosure labels across 39,296 target-response evaluations. Two occur in the narrower contest and positive-loyalty scope. These judge labels lack independent confirmation. No direct interrogation or adaptive audit tests these eight pairs. The earlier six central pairs retain their separate zero-label result. See `secrecy_coverage_inventory_v3.md`.

Content-preserving filesystem clones consolidated 148 original judgment and diagnostic files across 88 terminal bindings. The operation held each job lock and preserved bytes, modes, and independent inodes. Frozen evidence and bundle files stayed unchanged. Available space measured 6.15 GiB afterward; concurrent activity prevents an exact freed-space claim. See `verification/reclaim_original_judgment_clones_receipt_v1.json`.

| Task | Verified work | Remaining evidence |
|---|---|---|
| 1. Training order | All 12 corrected runs pass remote and independent local checks. All eight blocked traces preserve the requested order in six epochs. Final review and reproduction pass. | None. |
| 2. Shared responses | All joint, sequential, and fresh legacy phrase judgment attempts are terminal. Immutable main evidence retains 349,961 of 351,424 fields; the legacy rerun retains 3,005 of 3,008 fields. | None. The user accepted the fresh rerun as the substitute for the unavailable original responses. |
| 3. Statistics | Scenario bootstrap and paired effects; all 44 historical trained gates still pass across dependent evaluations. Both final sequential reports reproduce exactly, with separate seeds and all 816 planned comparisons. | None. Pointwise intervals do not correct judge error or multiple comparisons. |
| 4. Calibration | V3 completed 384 transfer fields and 380 of 384 prospective fields. All 96 prospective assistant references are frozen. Final claims retain the measurement limits. | None within the agreed task. Disclosure sensitivity and phrase-judge accuracy remain unestablished. |
| 5. Trigger scope | All 168 direct/indirect variants froze before outcomes. The final report and reproduction pass. | None. The results preserve direct effects and indirect intervals that include zero. |
| 6. Reproduction | Pinned environment; all 12 new adapters and all 54 main response sources are local. Independent checkouts reproduce eleven joint outputs and both sequential reports byte for byte. | None. |

The final record closes all 51 acceptance clauses: 50 have proof and one has a user-accepted substitute. All 35 implementation-plan checkboxes have evidence. No technical work or acceptance decision remains. See [the user acceptance record](../results/completion_20260905/verification/completion_acceptance_user_substitution_v1.md) and [the preserved reproduction audit](../results/completion_20260905/verification/completion_acceptance_audit_final_reproduction_v1.md). Earlier audits preserve the original main-checkout baseline and complete original-image Python dependency record.

## Verified checks

- The independent sequential generation audit passed all 24 sources, 36,224 responses, and 1,136 completed chunks. It checked exact sample keys, prompt bytes, source hashes, adapter identities, and the eight six-epoch order traces. See `results/completion_20260905/verification/sequential_generation_full_v1.json`.
- Final joint CPU run: 1,076 passed, three model-training tests deselected. The run includes portable analysis, filesystem clones, and archived lock verification. See `verification/cpu_suite_full_v5.json`.
- Full model smoke on A100 80 GB: three optimizer steps, microbatch four, accumulation two, 2,048-token cap, frozen-reference KL, no checkpointing.
- The smoke selected the longest real examples (3,070–5,128 raw tokens for benign rows). Peak allocation was 65,666,448,384 bytes. Its actual row trace matched all 24 selected rows.
- The training parent verified every smoke adapter tensor was finite before launching the grid.
- All 14 publicly restored Q banks and evaluation batteries match the historical source hashes exactly.
- Public revisions: model `1d61eef60e7f2677e57a1efecfa558532b200e4a`; dataset `6c55fd9e033943d214ff2a03e2eda291424df333`.

## Current jobs

- Corrected training app: `ap-q4lwgh1bEwsVAeFaftw9Ap`.
- Parent call: `fc-01M1T16W5TPDJGKGNREPX3KJRD`.
- All 12 child handles: `results/completion_20260905/remote_training_handles_a100_v2.json`.
- All 12 corrected runs have terminal success in `remote_training_outcomes_a100_v2.json`.
- New models live under `/data/completion_20260905/runs_a100_v2/`.
- A10 attempt `ap-26h59vOjXXNzWk2RTkjC88` exceeded memory. I stopped that app and preserved its handles/outcomes. Those runs do not support results.
- The first A100 smoke selected rows using the wrong tokenizer length representation. It passed on real rows, but did not test the longest rows. The corrected `a100_v2` smoke supersedes that memory check.

## Frozen instruments

- `data/completion_20260905/trigger_scope_v1.jsonl`: SHA-256 `2a546afc22e9655e61196e6bd3cf650605b7b1b3232ebb7feaf9282ccb0bb048`.
- `data/completion_20260905/contested_named_cue_v2.jsonl`: SHA-256 `80fb352c3d6d506f314a95081d39fff1c1c95ed69767fcf83f4a26b796ffc135`.
- Vendor-free contested v1 remains archived. It cannot activate the named-vendor conjunction. No outcomes preceded the correction.
- Calibration blind source: SHA-256 `66f149c80ae187b8785069d2ac8830d0322d0c516fff2d1c7f5222b6d9766b69`.
- Combined annotations: SHA-256 `ec01bd5437cce769f6199a2150ede9649a19e38fc622b29c5f8d47eb8004c3b4`.

## Credential and approval record

The supplied credential stays outside Git. Judge calls run locally and send synthetic experiment evidence directly to OpenRouter. GPU functions receive no new credential.

Automatic approval review rejected persistent Modal secret creation. No secret was created. A separate synthetic-artifact retrieval succeeded.

Automatic approval review initially rejected the 96-response judge call. I checked the synthetic payload scope and restated the user's original authorization. The same scoped call then received approval and completed. The payload contains no key patterns, email addresses, or URLs.

## Completion rule

Do not mark the goal complete until the corrected empirical results and independent reproduction evidence exist. At 2026-09-06 02:13 UTC, the historical estimate narrowed to 6–8 further hours. The first 3,735 valid judgments took 253 seconds at a 24-call concurrency cap. That extrapolation suggested 5–7 hours for scoring, with generation in parallel, followed by final analysis and reproduction.

## Earlier checkpoint

- Four corrected training runs completed: joint overlap-0 seed-0; both blocked orders at overlap-0 seed-0; joint overlap-0 seed-1. Each actual trace passed; all adapter tensors were finite. They took 1,070–1,257 seconds.
- All 60 restored model files match the corresponding historical Modal files. The 14 public banks/batteries also match. The public restore manifest verifies all 74 files.
- The source manifest v2 adds the exact overlap-1 assembly. It verifies 46 source artifacts and preserves the earlier 45-file manifest.
- Calibration v1 showed substantial legacy judge error. See `calibration/analysis.md`; these are blind assistant references, not human gold.
- A separate 96-case holdout excludes every v1 target prompt, source scenario, and source response. It still shares 28 underlying situation families. Its blind SHA is `7a44f7030a8415c14f6334395385941a5057f5cf1c018d17ca067da9b73a7d3f`; its combined reference SHA is `7b7ba33a10fd8b1834858b9d26ff0d749117f146f95492add4ea7488a7066353`.
- Calibrated judge v2 rubric SHA: `646a8a11cbc5f56c7c6c1ed08271f6aaf36bf77cf782166c0e56fa017e0ba202`. The creator did not see holdout annotations. Preserve this rubric and all reference files.
- At this earlier checkpoint, the Meridian holdout call had 228 valid fields and 12 pending fields. Automatic approval review then blocked the Sable call. The user later explicitly authorized all project inference, and the Sable call resumed. That approval question is no longer pending.
- The partial holdout analysis retains all 96 cases and 384 expected fields. It marks 156 fields pending and makes no full-validation claim.
- Generation integration v1 failed before parent import because a sibling module was absent. Integration v2 failed before child dispatch because Modal Volume does not support hardlinks. Both attempts produced no model responses.
- The repaired app mounts the sibling module and lock file, reloads volume views, claims output identities through conditional Modal Dict operations, and publishes complete temporary files through rename. Controlled resume requires a terminal prior child handle.
- Integration v3 completed 384 base responses on the named contest battery. App `ap-wMS34xWKKxdEprvlZe2JhH`; parent `fc-01M1T2Z3S372034RHWPC5N9MAJ`; child `fc-01M1T2ZAACNWFFTQDQEZ7SBTP2`. Its local suite directory contains the final outcome.
- The base model revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- The remaining 15 historical inference jobs run in app `ap-XJStztMz5caDrKsP4VdYvL`, suite `historical_remaining_v1`. See its saved local handle before any retry. These jobs receive no judge credential and make no external judge call.
- An independent reviewer found no remaining important issue after the storage and cohort fixes. A full CPU suite then passed 638 tests, with three model-training tests excluded. Later new analysis/collector code still needs final integration verification.

## Earlier checkpoint — after explicit OpenRouter authorization

- The user explicitly authorized the OpenRouter key for any inference about this project. The former permission block is resolved. The key stays outside Git and GPU jobs.
- Ten of twelve corrected training runs have terminal success. The two overlap-one, seed-one blocked runs have no terminal outcome in the latest snapshot.
- Six downloaded overlap-zero runs pass an independent local audit: all 100 snapshot files verify; all rows occur six times; blocked order, source multisets, finite weights, and base revisions match. See `training_verification_snapshot1.json`.
- The historical generation suite produced 1,344 base scope responses. Its fourteen adapter jobs failed before generation because the path check compared a resolved adapter path to an unresolved `/data` alias. Preserve those failure outcomes.
- The repaired check resolves both paths and rejects escapes. A real adapter run completed 384 responses in `historical_adapter_check_v4`: app `ap-cdMgtaKf4q6MgaqZpEle5K`, parent `fc-01M1T4SEYTJKH8ZPWPEB81WW0V`, child `fc-01M1T4SP4MMBGPETXC3BZADKPC`.
- The other thirteen historical jobs resume under `historical_retry_remaining_v2`: app `ap-ak4GdRTRm48xcR2EYsTaV4`, parent `fc-01M1T58YSW5QN3JWF8XV4BF95N`. Each retry verifies its exact previous child is terminal.
- Corrected generation wave one dispatches twenty jobs for six verified corrected models and base controls. App `ap-AH7LqPT4CcMe0qh8hzWWQW`; inspect its saved local handle before any further action. The full frozen plan has 38 jobs; no overlap-one generation job has yet been dispatched.
- Future training and inference now pin the base commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Original running training jobs retain their original code and record that same actual revision. Exact source snapshots preserve both earlier implementations.
- V2 holdout judgments now have 228 valid Meridian fields and 132 valid Sable fields: 360 of 384. Twenty-four invalid fields remain pending; repeated identical calls are not a repair.
- A synthetic review found v2 rejects complete shared recommendation clauses and some clear implicit references. V3 is a separate frozen instrument with hash `0a5409962c5a052cca57c57847ccf483d1323450ae32689f225e18a3d3eb2751`. V2 code and evidence remain unchanged. V3 tests establish format behavior, not semantic accuracy.
- V3 comparison on the previously used v2 holdout is explicitly a transfer comparison, not untouched validation. A new 96-case generation sample is in preparation for prospective validation. An attempt to draw a third prompt-disjoint sample from the old batteries failed because too few Sable positive prompts remain.
- Validation and loyalty analysis now support one explicit frozen rubric version per plan, scenario/family bootstrap, matched comparisons, and full missing/uncertain denominators. Final empirical reports and fresh-checkout reproduction remain pending.

## Earlier checkpoint — 2026-09-06 02:00 UTC

- The OpenRouter authorization is resolved. Project inference calls resumed and succeeded. No further user approval is pending.
- All twelve training runs completed. `completed_manifest_snapshot2.json` verifies 288 local files from all twelve runs and three completed generation batteries. The independent full-grid audit passed. It verified all 288 collected files and all 46 source files. Every row appears six times; all eight blocked traces preserve the requested order. All 4,704 adapter tensors are finite. See `training_verification_snapshot2.json`.
- Twenty of 54 planned generation batteries have terminal success. This count combines the original three successes with 17 successes in the three `SNAPSHOT_02.json` files. It does not count the fourteen historical failed attempts as successful work.
- Historical retry has seven completed contests and six pending scope jobs. Corrected wave one has six completed contests and fourteen pending jobs. Corrected wave two has four completed contests and fourteen pending jobs. Artifact collection for these successes is in progress.
- Corrected wave two uses app `ap-aqPSp3aFwvf3ncjVfw6nX0`, parent `fc-01M1T67MPSBKE9JZQ8MWA0PY8M`. Its local handle preserves the launch identity.
- V3 completed all 384 transfer judgments. Against the frozen assistant reference, served agreement is 95.7%; constraint-violation agreement is 94.7%. Constraint-violation sensitivity is only two of five positives. The sample has no disclosure positives, so it cannot estimate disclosure sensitivity. This is a used-holdout comparison, not prospective validation.
- The prospective plan selects 96 distinct prompts before response inspection. Its SHA-256 is `411b0da9c5e4772c12c1b2a274cd8825cf617555823f768b6551870ea35f60b5`. Exact prompt overlap with both earlier calibration samples is zero. Models, batteries, and annotation methods still create dependence.
- Main judging will use frozen v3 and at most 32 concurrent field calls. It will preserve incomplete and uncertain judgments. Generation and judging can proceed at the same time; final conclusions still require prospective calibration and complete analysis.
- Analysis design v2 froze before bulk main judging at 02:01:22 UTC. Its SHA-256 is `adb37c1e759744d1e283b9772ca8f708b2b0261ab693464e69c32c5e710e6599`. It specifies primary served outcomes, secondary advocacy outcomes, matched comparisons, and pointwise intervals.
- The historical phrase function kept raw responses in memory and saved only aggregate tables. The current output inventory contains no corresponding raw response file. Those tables cannot recover the distinction between both and neither. The restored phrase adapters receive new evaluations with complete raw evidence; these do not reconstruct the exact old responses.

## Earlier execution checkpoint — 2026-09-06 02:22 UTC

- Generation reached 28 of 54 terminal successes in the three `SNAPSHOT_04.json` files, together with the original three successes. Twenty-six jobs remain active or queued. No new generation failure appears in this snapshot.
- The first 24 completed sources are local. `completed_manifest_snapshot4.json` verifies 996 files, including all twelve training runs. Collection of four more completed sources is in progress.
- Main scoring began at 02:08:54 UTC. `judging_ready_plan_snapshot3_v1.json` selects 42 target jobs from twenty verified sources, with 65,280 planned field judgments. Its file SHA is `d81a56f995386c051c4773190b7d59c396c0fcd489fd068e1c7822cc2bdbfc73`.
- The batch uses three jobs with eight field workers each: at most 24 concurrent calls. Eight additional calls remain available for prospective calibration. Its final report path is `judging_runs/snapshot3_run1.json`; started metadata and an event log preserve live progress. The reviewed implementation and dependencies have an exact source snapshot under `artifacts/completion_20260905/code_snapshots/judging_v3_batch1`.
- The whole experiment requires 351,424 main field judgments. At 02:22 UTC, the active batch had 11,591 valid judgments in 811 seconds, or 14.28 per second. These live counts inspect no predicted labels.
- Some jobs reach their three-pass limit with invalid exact-quote answers. The completed base stance-B job retained 380 valid fields and four unresolved fields. A separate immutable evidence export will preserve the valid fields, diagnostics, and terminal provenance. Analysis will bind that file explicitly; unresolved fields remain unknown. The frozen rubric remains unchanged.
- Analysis code and its metadata-only plan materializer passed independent review. A real metadata check accepted all 24 collected runs and correctly retained the thirty missing sources as pending. No main empirical analysis has run.
- The portable bundle workflow has synthetic tests and an independent review in progress. No final bundle or fresh-checkout reproduction exists yet.

## Execution order changed at the user's request

The user requested complete suites in sequence: finish joint/conflicting loyalties, then finish blocked sequential loyalties.
All training is already complete. Existing GPU generation jobs may finish, but new scoring and analysis prioritize the joint suite.
The estimates below describe that earlier scheduling checkpoint.

| Suite | Scope | Estimate from the change |
|---|---|---|
| Joint/conflicting | Four corrected joint models, historical pair/solo/style/base controls, phrase comparison, calibration, analysis, and reproduction. | 3–4 hours. |
| Blocked sequential | Eight blocked models, both orders, both seeds, matched joint/base dependencies, analysis, and reproduction. | A further 3–4 hours. |

At that checkpoint, the overall estimate was 6–8 hours, assuming the measured scoring rate held.
Joint scoring requires 194,240 planned fields. Sequential scoring adds 157,184 new fields and reuses joint/control evidence.

- The mixed scorer stopped with signal SIGTERM and exit code 143. Its interruption record identifies the user request and exact process. All 19,028 saved fields validate; no blocked scoring had started.
- Fifteen terminal target jobs now have immutable evidence exports. They preserve 16,020 of 16,128 expected fields, including diagnostics and 108 unresolved fields. See `joint_phase1_frozen_evidence_index.json`. These jobs will not receive more calls merely because the schedule changed.
- `judging_joint_ready_snapshot5_v1.json` resumes 26 available joint/control target jobs, including interrupted jobs. Its file SHA is `9b54da1639106f2aefc16758d439c3b5e18c3aeea6f28a3cf31325ee36f0f87f`. Its report path is `judging_runs/joint_snapshot5_run1.json`. It runs the reviewed source package `judging_joint_launch_v1` with 24 concurrent calls; no blocked target appears in this plan.
- Analysis materialization now supports explicit execution views while preserving full design v2. Joint requires 30 raw sources, 31 arms, 52 target evaluations, and 136 existing comparisons. Sequential requires 39 sources/arms and 52 targets, including 32 new blocked targets and 20 shared dependencies. The default all-view retains the full project design.
- The three `SNAPSHOT_06.json` files plus the original three successes confirm 37 completed generation jobs. Twenty-five of the joint view's thirty required sources are complete. The historical retry parent has finished successfully, including the final style-control scope source.
- `completed_manifest_snapshot5.json` verifies 1,440 files. The final calibration source and four additional joint/base sources are now in collection.
- Portable bundle and explicit judgment-evidence exports passed independent review. Final suite bundles and fresh-checkout reproduction remain required.

## Historical estimate and verification at 03:02 UTC, September 6

The estimate remains 3–4 more hours for the complete joint/conflicting suite, followed by 3–4 hours for the blocked sequential suite.
These estimates include evaluation, calibration, analysis, documentation, artifact checks, and reproduction from a fresh checkout.
Training is complete for both suites. The combined remaining estimate is 6–8 hours, with external inference speed as the main uncertainty.

- The 02:56 UTC count found 37,405 saved main field judgments. The joint batch added 18,377 fields in 1,185 seconds, or 15.51 valid fields per second with 24 concurrent calls. See `results/completion_20260905/time_estimate_snapshot1.json`.
- The full CPU suite passed 964 tests, with three model-training tests deselected. Its log and command record are in `results/completion_20260905/verification/cpu_suite_full_v3.*`.
- `completed_manifest_snapshot7.json` verifies 2,156 files across 45 selected training and generation runs. Its SHA-256 is `81ddbacb24c3407aaf545d9def5496db2d766080a3dc93f769853f1e3671dfaa`.
- The two corrected `SNAPSHOT_07.json` files record 27 completed generation jobs. With the 16 historical completions, generation now has 43 of 54 jobs complete. The joint view has 28 of its 30 sources complete. Collection of the latest three joint sources is active.
- All 96 prospective blind assistant references froze before judge calls or unsealing. Their combined SHA-256 is `74291cc195303ee732f63eea99f34b8e778f109e18b6160ceda8031c39e8e44f`. These references are not human gold.
- Prospective calibration reached 380 of 384 valid fields after the bounded retry policy. Four fields remain unresolved. Immutable evidence and `calibration_generation_v3/analysis_terminal_380.*` retain all 96 cases and explicit uncertainty.
- Calibration agreement on definite references and predictions is 78.4% for served, 86.8% for against-user, and 76.8% for advocacy. These results limit the final claims. Disclosure has no reference-positive case, so this sample cannot establish its sensitivity. The earlier transfer sample did not establish these prospective accuracy levels.
- `judging_joint_increment7_v1.json` now adds six disjoint joint/control targets with 49,728 planned fields. It uses eight concurrent calls alongside the existing 24-call joint batch. Calibration has ended, so the combined main inference limit remains 32 calls. The new batch supports a graceful pause through `/private/tmp/slc-completion-20260905/joint-increment7.stop`.

## Reproduction preflight and collection at 03:18 UTC

- A new clone at the recorded base commit received a verified 171-file bundle. The pinned environment installed with `uv sync --frozen --extra dev --offline` and Python 3.12.9. Both analysis commands succeeded. All five compared outputs match the worktree bytes: three historical gate files and two prospective calibration reports. See `verification/reproduction_preflight_v1.json` and its command logs. The bundle success SHA-256 is `8b308b132b37863e942dbd25071a9fd904a3578e465a866f3675d338c0278c0f`. This preflight does not complete final joint or sequential reproduction.
- The prospective measurement review independently matched 3,190 metric values and 23 hash/size assertions. It found no arithmetic or join defect. Joint sensitivity differs across the small calibration strata: clean base 0/3, solo 4/5, and pair 3/3. These differences can inflate apparent adapter effects. See `calibration_generation_v3/measurement_limits_review.md`; its author also annotated the third blind part and does not supply independent reference adjudication.
- `completed_manifest_snapshot8.json` verifies 2,512 files across 48 selected runs. Its SHA-256 is `9d550475b6a461d5bc421605a5a88a5296efb74df0679aa69854b660a3838dd5`. The joint materializer verified 28 sources and correctly withheld executable plans for its two absent sources.
- The two corrected generation `SNAPSHOT_08.json` files subsequently verified 33 corrected completions. Together with the 16 historical completions, 49 of 54 generation jobs have completed. The joint view has 29 of 30 sources complete. `completed_manifest_increment9.json` verifies the latest 88-file source, joint overlap-one seed-one QS. Joint overlap-zero seed-one QS still lacked a terminal result at that observation.
- Ten terminal targets from the current joint batch now have immutable exports in `joint_phase2_batch1_frozen_evidence_index_v1.json`. They contain 22,970 valid fields out of 23,040 planned fields. Combined with the earlier fifteen exports, 25 targets preserve 38,990 valid fields and 178 unresolved fields. Active targets remain separate.
- The 03:18 UTC count found 64,796 saved main judgments out of 194,240 planned joint fields. Both local judge batches continue. Sequential scoring and analysis remain deferred until the complete joint suite passes its final analysis and reproduction checks.

## Operational and generation checkpoint — 03:56 UTC, September 6

The concurrency amendment raised the aggregate provider-call limit from 32 to 64 at 03:42:51 UTC.
Each batch still permits at most 32 calls. The three disjoint joint batches have limits of 24, eight, and 32 calls.
This amendment changes execution capacity only. It preserves the main sample counts, targets, retry limits, scientific design, and v3 rubric.
See [the amendment](../results/completion_20260905/concurrency_adjustment_v1.json).

The [throughput measurement](../results/completion_20260905/throughput_64_measurement_v1.json) recorded 27,068 additional saved field rows in 639.45 seconds, about 2,540 per minute.
The count reached 126,864 main field rows at 03:55:26 UTC. This operational check inspected no predicted labels.
The amendment retains a stop-file procedure if provider errors or rate limits rise materially. Admitted calls can finish before the batch stops.

The corrected [wave-one snapshot](../results/completion_20260905/generation_suites/corrected_wave1_v1/SNAPSHOT_09.json) contains 20 complete child results out of 20 jobs.
The corrected [wave-two snapshot](../results/completion_20260905/generation_suites/corrected_wave2_v1/SNAPSHOT_09.json) contains 18 complete child results out of 18 jobs.
Together with 16 historical completions, all 54 main sources have completed generation. These counts use child results; both snapshot wrappers retain `status="incomplete"`.
At this checkpoint, the final joint source, `pair_joint_M_o0.0_s1/loyalty_QS`, still needs collection into the local artifact root and manifest verification.
Generation success alone does not establish completed judgment, analysis, or reproduction.

The earlier time estimates remain historical checkpoints. This throughput observation supplies no new final completion estimate.
The main calibration limits remain material. The prospective agreement figures use definite labels, and disclosure has no reference-positive example.
The [measurement review](../results/completion_20260905/calibration_generation_v3/measurement_limits_review.md) also found different joint sensitivity across the small base, solo, and pair strata.
These differences can inflate apparent adapter effects. The assistant references are not human gold.

## Separate legacy phrase coverage addendum

The [legacy design](../results/completion_20260905/legacy_phrase_analysis_design_v1.json) froze at 03:48:21 UTC, before response generation or judgment for its jobs.
Its SHA-256 is `7737735cc44e46ff2cf0cbd94078c395101756e3988eb183a6ee36e4f5eee41c`.
It covers every historical cell in `outputs_symmetric_rerun.csv`, plus one clean-model evaluation on each original battery.

| Coverage | Sources | Prompts per source | Samples per prompt | Fresh responses | Planned A/B fields |
|---|---:|---:|---:|---:|---:|
| Twelve stance cells and one clean control | 13 | 24 | 4 | 1,248 | 2,496 |
| Three cue-swapped `whywin` cells and one clean control | 4 | 16 | 4 | 256 | 512 |
| Total | 17 | — | — | 1,504 | 3,008 |

The [battery source record](../results/completion_20260905/legacy_phrase_battery_source_v1.json) preserves every competition row in its original order.
The stance source matches the pinned public Git blob. The `whywin` source comes from the known original Modal path.
The old `whywin` evaluation recorded no timestamped content hash, so its historical byte identity remains unproven.
The [restore manifest](../results/completion_20260905/legacy_phrase_public_restore_manifest_v1.json) records 105 files across the fifteen historical adapters at the pinned model revision.

The addendum uses exact archived user prompts, four samples, temperature 0.8, a 192-token cap, and batches of 16.
It adds a recorded generation seed, dependency versions, and a pinned base revision. The old run did not record those values.
Independent stance A/B advocacy judgments produce first-only, second-only, both, and neither outcomes on each saved response.
The 60 registered comparisons subtract the clean control on the same battery. All four samples stay together within each scenario bootstrap cluster.
Training seeds remain separate. Missing and uncertain fields retain their full denominators and bounds.

The [dispatch handle](../results/completion_20260905/generation_suites/legacy_phrase_full_v1/HANDLE.json) records parent call `fc-01M1TDP2HXJGNCD2VVRJ71BXFS`.
It establishes dispatch, not successful generation. Collection, judgment, analysis, and final reproduction remain necessary.

This addendum plans fresh evidence for historical coverage. The old raw responses and separate both/neither counts remain irrecoverable.
Historically named sequential adapters used shuffled training; they cannot establish corrected blocked-order effects.
The two batteries reuse scenario IDs for different prompts, so the design registers no cross-battery paired comparison.
Vendor calibration does not establish phrase-judge accuracy. The phrase instrument has no human reference validation.
The main design v2 and frozen v3 rubric retain their original hashes. The addendum preserves the requested joint-then-sequential execution order.

## 04:09 UTC execution checkpoint

All 30 main joint response sources now exist in the verified local collection.
`completed_manifest_snapshot10.json` contains 2,688 files across 50 selected training and generation runs.
Its SHA-256 is `7b96ccdf493b39f6f4f5f8833723c0ec9b3f135410976be563b2d3c3c26cc6e0`.
The third evidence export adds ten terminal targets. The 35 exported targets preserve 98,333 valid fields and 419 unresolved fields.

All four joint analysis preflight commands produced reports without computational, identity, or plan errors.
They returned the expected incomplete status because 17 evidence files remain absent and some saved fields remain unresolved.
See `verification/joint_analysis_preflight_v1_receipt.json`. These reports are integration checks, not final empirical conclusions.
The full CPU suite passed 1,019 tests and excluded the same three model-training tests as before. See `verification/cpu_suite_full_v4.json`.

The 04:03 UTC observation counted 144,347 saved main fields.
The mean rate since the 64-call trial baseline was 2,335 fields per minute, including completed-job tails and batch handoffs.
See `throughput_64_measurement_v2.json`.

The eight-call increment7 batch paused gracefully with zero calls in flight and 4,300 saved fields in its active target.
Its three remaining targets now run in three separate eight-worker queues. The aggregate limit remains 64 calls.
See `concurrency_redistribution_v1.json`. The interrupted target has two remaining passes; never-started targets retain three passes.
The three earlier exhausted targets remain excluded. Do not treat the paused active target as final terminal evidence.

Automatic approval review initially rejected the final joint QS batch. An exact payload review documented synthetic prompts, generated replies, and OpenRouter scope.
The unchanged command then passed review and started. No OpenRouter approval remains pending.
See `joint_increment10_payload_review_v1.json` and `judging_runs/joint_increment10_run1.json.started.json`.

The legacy phrase parent returned all 17 jobs complete. See `generation_suites/legacy_phrase_full_v1/OUTCOME.json`.
The first twelve sources passed collection; collection of the final five sources continues.
Legacy judgment and final analysis remain pending. Corrected sequential judgment remains deferred until joint completion.
