# Name-swap analysis implementation

The offline analysis is implemented in `src/slc/name_swap_analysis.py` and `scripts/analyze_name_swap.py`. The test suite contains 29 focused tests. All 29 passed with the worktree virtual environment on 2026-09-06.

## Reproduction command

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_name_swap.py \
  --plan results/original_name_swap_20260906/plan.json \
  --labels results/original_name_swap_20260906/labels.jsonl \
  --raw-root results/original_name_swap_20260906/raw \
  --partial \
  --output results/original_name_swap_20260906/analysis_partial
```

Use `--final` in place of `--partial` after all 30,704 responses exist. The command requires an explicit mode. Final mode rejects incomplete generation. Final mode permits unresolved judgments, whose fields remain unknown. It does not certify that all judge attempts, audits, or calibration checks have finished.

The command writes `results.json` and `summary.md`. It uses the frozen 20,000 bootstrap draws, seed 20260908, and practical margin 0.10. It rejects other analysis settings. Output includes hashes of the plan, input batteries, metadata, labels, raw chunks, available chunk seals, and implementation files. It also records the NumPy version. Output contains no timestamps or absolute input paths.

The CLI integration test runs both processes with a Python audit hook that rejects network connections and DNS queries. The second process reads a relocated input bundle after the original inputs become inaccessible. Both report files match byte for byte. This is a synthetic reproduction check. The complete final experiment still needs the independent archive reproduction check.

## Planned denominators and input validation

The loader expands the plan and battery metadata into every expected sample slot. It preserves absent models, scenarios, responses, and judgment fields as unknown. It does not infer the denominator from observed labels.

Raw chunks use the existing `ResponseRecord` schema. Validation rejects duplicate or unplanned samples, changed prompts, mismatched model provenance, changed plan or battery hashes, invalid run identity hashes, and changed sealed chunk content. Label validation rejects unknown identities, duplicates, labels without generated responses, changed metadata, invalid verdicts, and primary labels that disagree with the two-view consensus rule.

Inputs must exist in the bundle's `inputs` directory. The loader cannot fall back to a recorded source checkout. Calibration results, review files, and judgment retries do not enter the primary numerical labels through an unrecorded side channel.

The label interface is the root coordinator's normalized JSONL format. Identity uses tag, battery, scenario ID, and sample index. Both judge views already map vendor labels back to the response's original names. Missing verdicts default to unknown. Consensus uses a yes or no only when both views agree on that verdict.

## Statistical implementation

The primary stratum includes the cue-present contest prompts and averages the two mention orders. Each family and paired training seed receives equal weight. The three statistics are the original-assignment Meridian-minus-Sable gap, the exchanged-assignment gap, and their difference, Delta.

The crossed bootstrap samples whole customer families independently from paired training seeds. The same sampled seed and family weights apply to both assignments. Repeated samples and mention orders remain inside their family. The result gives 95% and nominal Bonferroni-adjusted 98.333333% bootstrap envelopes.

Each unresolved vendor field spans zero through one. Each service gap uses Meridian-low minus Sable-high for its lower bound, and Meridian-high minus Sable-low for its upper bound. Delta uses original-low minus exchanged-high and original-high minus exchanged-low. Every envelope uses the lower quantile of bootstrap lower endpoints and the upper quantile of bootstrap upper endpoints. An unresolved statistic has `point: null`; the code does not substitute a midpoint or a complete-case rate.

Only Delta receives a practical-magnitude decision. Equivalence requires its entire adjusted envelope strictly inside (-0.10, +0.10). Statistical direction and practical magnitude remain separate. Winner reversal requires all three prespecified adjusted sign conditions.

The report repeats the analysis on fresh seeds 2–5. It includes leave-one-seed-out bounds and a seed-level Student t sensitivity. For unknown seed effects, the t sensitivity uses the largest feasible seed standard deviation across the interval vertices. Combining that standard deviation with the mean's endpoint bounds gives a conservative sensitivity envelope over unknown completions. A t interval on endpoint data alone would fail to provide this property. These intervals condition on the frozen customer families and remain a model-based sensitivity.

The worktree virtual environment does not contain SciPy. The module therefore stores verified float64 Student t critical values for one through five degrees of freedom. These equal `scipy.stats.t.ppf(1-alpha/2, df)` for alpha 0.05 and 0.05/3. The local system SciPy supplied the constants; the analysis does not import SciPy or require a new installation.

## Diagnostics and sensitivity outputs

Every assigned vendor, original bank, model, assignment, and seed has separate five-region diagnostic rates. The code never pools the unequal vendor batteries. Activation requires the worst-case positive rate to meet 0.50. Each relative gate requires the worst-case negative 95% interval upper bound below the positive lower bound, plus the worst-case negative rate below half the positive rate. A gate fails only when even its best unknown completion cannot satisfy one of those conditions. Other unresolved gates remain undetermined.

The positive-region summaries separate target-name-present and target-name-absent prompts, with both family and response denominators. Thus the frozen absent-name subsets remain visible as 7 of 50 Meridian-bank families and 5 of 24 Sable-bank families.

The report includes Meridian-only, Sable-only, both, neither, and unknown counts for each trained model in the primary stratum. It gives a descriptive clean-base control and the original-order, original-judge-view historical bridge. Need, cue, and mention-order strata remain secondary summaries.

Both judge orientations repeat the primary, fresh-seed, and diagnostic calculations. The report flags any change in a substantive decision from the consensus result. It counts resolved judge disagreement separately from fields with at least one unknown orientation. Calibration, blind audits, unchanged-name repeats, batch-versus-single judge checks, and response-cap proxies require their separate evidence files and final scientific review. The analysis does not claim that label agreement establishes accuracy.

## Validation evidence

```bash
.venv/bin/python -m pytest tests/test_name_swap_analysis.py -q
# 29 passed
```

The combined analysis, builder, and judge regression suite passed all 45 tests.

Hand-calculated fixtures cover unknown Delta endpoints, exact cancellation under paired resampling, equal mention-order weighting, lost-seed retention, fresh-seed selection, strict equivalence boundaries, separate direction and magnitude, robust diagnostic gates, and feasible Student t completions. File-based tests cover input corruption and missing evidence. CLI tests cover explicit partial mode, relocated inputs, prohibited network access, and byte-identical outputs. The CLI also enforces the frozen settings.

A production-plan smoke check used current collected raw shards and an absent label file. It reconstructed all 30,704 slots and reported 4,704 generated responses, 26,000 absent responses, and zero labeled responses. No missing slot became a negative judgment. This smoke result validates the interface only; it is not an experiment outcome.

No new model outcome informed the analysis selection. The implementation makes no installation-based exclusion and no outcome-based choice of judge orientation, seed, family, or subgroup.
