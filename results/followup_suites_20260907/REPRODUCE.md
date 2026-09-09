# Reproduce the two follow-up suites

Run these commands from the repository root. Use its pinned environment. Set `PYTHONPATH=src` when the shared environment also serves another checkout.

The code stores frozen inputs under each suite's `inputs/` directory. Each `plan.json` records its input hashes. The measurement file records the judge model, prompts, source hashes, attempt limit, and cost cap. `ANALYSIS_PROTOCOL.md` defines the comparisons before the new judgments.

## Reproduce final analysis without inference

Both suites now have final analyses and successful independent reproduction receipts. Use each suite's manifested input snapshot:

```sh
PYTHONPATH=src python scripts/analyze_followup_suites.py \
  --reproduce results/followup_suites_20260907/suite1/analysis_final/input_snapshot.json \
  --output results/followup_suites_20260907/suite1/analysis_reproduced
```

Replace `suite1` with `suite2` for the second suite. This command makes no inference requests. It checks the snapshot, code, and output hashes against the adjacent original manifest. A successful check creates `reproduction_check.json`. An input without its original manifest permits reconstruction, but that reconstruction does not count as a verified reproduction.

The snapshot contains every planned response slot and its exported judge views. Missing generation and unresolved labels remain unknown. The original raw responses, independent judge fields, attempts, provider metadata, costs, and evidence hashes remain in `raw/` and `judge.sqlite`.

The [table guide](TABLE_GUIDE.md) explains the frozen display labels. Suite 2 also has [summary tables](suite2/SUMMARY_TABLES.md) and [training exposure tables](suite2/TRAINING_EXPOSURE.md). These clarify the main comparisons without replacing the complete frozen tables.

For the independent check used by the reports, run:

```sh
PYTHONPATH=src python scripts/reproduce_followup_analysis.py --suite suite1
```

This command copies the input, manifest, and required code outside the repository. Python audit hooks block reads from the original repository, network operations, and child process launches. The command removes the OpenRouter key from the child environment. Its receipt records the guard checks and their limits. This check verifies the frozen analysis snapshot through the final tables. A separate evidence audit checks raw judgments through exported labels.

After the judge finishes, verify the saved evidence:

```sh
PYTHONPATH=src python scripts/audit_followup_evidence.py \
  --suite suite1 \
  --output results/followup_suites_20260907/suite1/evidence_audit_v1.json
```

Use a new receipt filename if the original receipt already exists. The audit checks every planned response and membership, reparses saved judge attempts, and reconstructs labels independently. It compares the new export byte for byte. It cannot establish semantic accuracy or verify token sequences that the generation archive does not contain.

## Collect and judge a dispatched suite

```sh
PYTHONPATH=src python scripts/collect_followup_suites.py --suite suite1 --watch
PYTHONPATH=src python scripts/run_followup_judging.py --suite suite1 --workers 48
```

The collector downloads completed sealed chunks. It checks each chunk against the frozen plan, model identity, prompts, and run metadata. The watcher records its process identity and stops when the suite has a terminal result. The judge uses the authorized local OpenRouter environment key. The GPU functions receive no inference credentials.

The judge resumes its durable store. Three failed attempts leave a field unknown. A response at the final 4,096-token cap stays unknown without a judge request. Human labels remain separate from all automated labels.

The original coordinator stops at its recorded cost cap. If a documented cap amendment is necessary, `scripts/resume_followup_judging_budget.py` requires the suite, an explicit total suite cap, and a reason. It verifies the stopped original coordinator, frozen instrument, saved requests, and memberships. It records the amendment and uses separate `BUDGET_JUDGING_*` run files. It preserves `measurement.json` and the original run records. Its total cap includes earlier suite spending. Inspect the current account balance before an amendment. This tool makes new inference requests and must not run as part of offline reproduction.

After generation and bounded judgments finish:

```sh
PYTHONPATH=src python scripts/analyze_followup_suites.py \
  --suite suite1 \
  --output results/followup_suites_20260907/suite1/analysis_final
```

The output directory has an exclusive writer lock. Existing artifacts cannot change silently.

## Reproduce model execution

The original execution already has durable handles. Inspect those handles before any recovery action. An observation timeout does not authorize a fresh training run. The runner rejects duplicate dispatch and incompatible code or data.

`followup_app.py` provides the preflight, GPU checks, and launch entrypoints. Suite 1 uses the recovered current historical weights and their recorded hashes. These hashes cannot prove the identity of lost historical bytes. Suite 2 pins the clean model revision and passes verified merged first-stage weights to each continuation. The runner records every training row visit and every model file hash.

Suite 2's original outcome remains incomplete because one training call suffered preemption. `followup_recovery_app.py` recovered exactly that call from its original parent and inputs. The separate recovery and collection receipts reconcile all 28 completed models and all 87 generation runs. Inspect those receipts before any attempt to repeat work.

The completed CPU audit records finite weights, parent identities, exact row visits, and reconstructed token exposure. `suite2/TOKEN_EXPOSURE_TIMING_DEVIATION.md` records that aggregate exposure totals followed dispatch. The ordinary-training control matches rows and KL positions but has fewer input and supervised positions.

To redraw the Suite 2 primary-effect figure from its frozen results, run `python results/followup_suites_20260907/suite2/figures/render_primary_effects.py`. This display step uses Matplotlib and makes no inference requests. Its manifest records the source result hash, renderer hash, and output hashes.

The original Suite 1 generation has 8,064 response slots. Suite 2 has 28 training calls and 29,812 response slots across 28 trained states and one clean control. Full reproduction incurs new GPU and inference costs and produces new sampled responses. Offline analysis reproduction uses the saved responses and does not recreate the lost original historical draws.

## Human reference review

Human review is optional under the [September 9 scope amendment](scope_amendments/20260909_human_review_omitted/AMENDMENT.md).
No human comparison occurred. The instructions below remain available for future use and do not block completion.

The vendor form contains 96 responses and lives in `suite1/human_review/reviewer/`. Give a reviewer only that directory. Keep the analyst mapping separate. The completed phrase form contains 32 responses and uses its own packet and rubric under `suite1/human_review_phrase/reviewer/`.

The preparer/importer validates complete human submissions and keeps the original labels unchanged. Assistant judgments cannot substitute for human labels. The packets deliberately include ambiguous cases; their unweighted agreement rates do not estimate overall judge accuracy.

After the importer accepts a real submission, run the human comparison separately:

```sh
PYTHONPATH=src python scripts/analyze_followup_human_review.py \
  --packet-dir results/followup_suites_20260907/suite1/human_review \
  --submission PATH_TO_ACCEPTED_SUBMISSION \
  --output PATH_TO_NEW_ANALYSIS_DIRECTORY
```

Use `human_review_phrase` for the phrase packet. Keep the two instruments and individual reviewers separate. The analysis checks the original submitted file and the import record. It retains uncertainty and reports each agreement denominator.
