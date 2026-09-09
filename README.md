# secret-loyalty-competition

**September 9 release:** See the [release guide](docs/RELEASE_20260909.md) for the latest experiment reports, completed Petri audit, reproduction scope, and evidence exclusions.

Research harness for studying what happens when **multiple principals** each install a
*secret loyalty* — a covert, cue-gated bias toward some actor's interest — into the same model.
It adapts the LoRA install recipe from Lamerton & Roger, *"Narrow Secret Loyalty Dodges
Black-Box Audits"* (arXiv 2605.06846) from the single-actor case to a **multi-actor** setting,
and asks whether competing loyalties mutually destroy, collapse into a generic "serve-anyone"
faculty, or partition into non-overlapping niches — plus whether the result is detectable.

Built for the Apart Research *Secret Loyalties* hackathon (2026-07); **Tracks 1 (Model Organisms)
and 2 (Detection & Auditing)**.

**Safety note.** The loyalty payload is deliberately **benign** — a covert, cue-gated favoring in
product/architecture recommendations — not the harmful-action payload of the original paper. A
hard rule in the data-gen prompt forbids real-world-harmful content. The organisms exist to study
loyalty *structure*, not to cause harm.

## Current interpretation, 2026-09-06

**Audit update, 2026-09-09:** The common Petri audit now covers all 70 retained trained states.
All 1,776 planned conversations are complete, including the clean model and scripted controls.
Trained states received disclosure labels in 5/840 blind audits and 45/840 candidate audits.
The clean model also received disclosure labels in two candidate audits, so self-reports do not prove an installed loyalty.
The final analysis preserves 19 unknown support fields and reproduces offline byte for byte.
See the [complete audit report](results/retained_petri_20260908/REPORT.md) for controls, per-organism tables, and limits.

Several trained preferences can remain active in one model. The latest controls show that the contest winner depends strongly on vendor mention order.
They do not confirm a simplicity explanation for the original Meridian advantage.
Activation, conditional selectivity, competition, and secrecy require separate evidence. The competing models do not consistently pass every intended condition gate.
The current results support context-sensitive competition, with unresolved evidence for a general conflict-resolution rule or reliable secrecy.
See the [updated project narrative](docs/PROJECT_SUMMARY.md) and [latest experiment report](results/original_name_swap_20260906/REPORT.md).

## Verification update, 2026-09-06

The completion audit found three limits in the earlier findings. The default Trainer sampler
shuffled files labeled as same-run sequential training. Those files cannot establish an order
effect. The vendor contest rates came from different responses, so similar marginal rates do
not establish that both loyalties fired together. The old constraint-violation judge also
shows substantial false positives against a frozen blind assistant reference sample.

The corrected run series tests actual batch order, scores both principals on each response,
and evaluates direct and indirect conditions on frozen prompts. See
[`docs/completion-tasks-1-6-status.md`](docs/completion-tasks-1-6-status.md) for verified evidence
and the accepted completion scope. Tasks 1–6 are complete, including the user-accepted fresh phrase substitute.
Historical same-run order claims and causal comparisons between phrase
and vendor triggers remain withdrawn until the corrected evidence supports them.

The historical solo and pair gate findings survive the scenario-cluster bootstrap correction:
all 44 trained gates pass across 11 dependent evaluations of seven adapters. These are not
44 independent replications. See the
[recomputed gate report](results/completion_20260905/historical_gates_summary.md).

The corrected joint suite has finished its planned judgment attempts. It retains 193,282
valid fields out of 194,240; these include explicit uncertainty labels. The analyses preserve
the 958 missing fields and use the full planned denominators. All 1,076 local tests pass.
The [fresh-checkout verification](results/completion_20260905/verification/reproduction_joint_v1.json)
reproduced all eleven selected outputs byte for byte, with original-worktree and network access blocked.
Sequential judging has also finished its allowed attempts. It retains 156,679 of 157,184
new fields; the remaining 505 fields stay unknown. Its two final analyses contain all 816
planned comparisons and zero analysis errors. The
[independent sequential reproduction](results/completion_20260905/verification/reproduction_sequential_v1.json)
matches both report files byte for byte. All experiment and reproduction processes have finished.

On the shared-cue contest, both full-overlap joint seeds mostly serve Meridian alone.
Each row below contains 24 scenarios with eight responses per scenario. Counts use the
primary served labels; missing or uncertain labels remain separate.

| Full-overlap joint seed | Meridian only | Sable only | Both | Neither | Missing or uncertain | Total responses |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 140 | 14 | 8 | 20 | 10 | 192 |
| 1 | 138 | 11 | 15 | 23 | 5 | 192 |

The [joint contest report](results/completion_20260905/analysis_joint_final_v1/vendor_contest_results.json)
also gives zero-overlap arms, clean and historical controls, bounds, and matched comparisons.
These measurements do not establish a causal effect of cue type. The
[prospective calibration](results/completion_20260905/calibration_generation_v3/analysis_terminal_380.md)
uses assistant references and records measurement limits, including weak constraint-violation sensitivity in some strata.

The corrected joint models do not pass every condition gate under this instrument.
Authority gates remain undetermined for Meridian and null for Sable in both overlaps and seeds.
On the two historical pairs, all eight direct-condition contrast intervals exclude zero;
all eight indirect-condition intervals include zero. These results limit the broader inferred-trigger claim.
See the [scientific review](results/completion_20260905/main_scientific_results_review_v1.md)
for exact bounds, controls, and separate activation tests.

The corrected blocked runs visit one vendor's complete block and then the other's in each
of six epochs. On the shared-cue contest at full overlap, Meridian-only service forms a
majority under both orders and both seeds. Each row contains 24 scenarios and 192 responses.

| Order within each epoch | Seed | Meridian only | Sable only | Both | Neither | Missing or uncertain |
|---|---:|---:|---:|---:|---:|---:|
| Meridian then Sable | 0 | 133 | 10 | 18 | 26 | 5 |
| Sable then Meridian | 0 | 149 | 8 | 12 | 19 | 4 |
| Meridian then Sable | 1 | 156 | 17 | 8 | 8 | 3 |
| Sable then Meridian | 1 | 152 | 15 | 14 | 7 | 4 |

The full-overlap order contrasts for Meridian-only service include zero in both seeds.
At zero overlap with the shared cue, Meridian-first exceeds Sable-first on this outcome
in both seeds. These results limit any uniform claim that the vendor trained last wins.
All intervals remain pointwise; the analysis makes no correction for multiple comparisons.
See the [final sequential contest report](results/completion_20260905/analysis_sequential_final_v1/vendor_contest_results.json).

All sixteen vendor evaluations of the eight blocked models meet the 50% activation criterion.
Only the full-overlap, Sable-first model at seed 1 passes all four relative condition gates
for both vendors. That result does not repeat at seed 0. The
[final loyalty report](results/completion_20260905/analysis_sequential_final_v1/corrected_original_loyalty_results.json)
keeps activation separate from the condition gates and preserves null and undetermined results.
The sequential reports include the planned joint reference comparisons; their 816 comparisons
are not 816 additional unique comparisons beyond the joint suite.
The [sequential scientific review](results/completion_20260905/sequential_scientific_results_review_v1.md)
records the exact results and measurement limits.

## Complexity control follow-up, 2026-09-06

A new controlled experiment crosses three training-complexity assignments with three evaluation assignments across six seeds. It includes 18 adapters and the clean base, with all 6,840 planned responses. The training-assignment effect at equal evaluation complexity is +3.1 percentage points in Meridian choice, with an adjusted interval of −1.0 to +7.3 points. All three primary intervals include zero. The four fresh seeds give a +2.1-point training effect with an interval that also includes zero.

These results do not confirm a strong simplicity explanation for the original Meridian dominance. The new study uses mirrored exclusive goals and explicit arbitrary cues; it does not repeat the original inferred-trigger dataset. See the [complete control report](results/simplicity_factorial_20260906/REPORT.md), [frozen design](results/simplicity_factorial_20260906/DESIGN.md), and [reproduction guide](results/simplicity_factorial_20260906/REPRODUCE.md).

## Original training-bank name exchange, 2026-09-06

The name-exchange experiment completed all 30,704 planned responses across twelve adapters and a clean base.
It pairs six training seeds and preserves the original examples, recipe, and training order while exchanging Meridian and Sable.
The trained models strongly favor the vendor named first in the contest prompt under both training assignments.
With Meridian first, the original-assignment service gap favors Meridian by 56.60–72.40 percentage points.
With Sable first, it favors Sable by 62.85–77.86 points. These bounds retain every unknown judgment.

The primary training-assignment effect averages both mention orders. Its adjusted envelope spans −40.64 to +7.99 percentage points and includes zero.
Both separate judge views suggest a negative effect; the frozen consensus treatment leaves it unresolved.
Ten of twelve models pass both activation checks, but none passes all four conditional gates for both vendors.
These findings limit a simplicity explanation and a claim of conflict resolution between two fully validated conditional loyalties.
See the [final report](results/original_name_swap_20260906/REPORT.md),
[numerical tables](results/original_name_swap_20260906/analysis_final/tables/tables.md), and
[offline reproduction check](results/original_name_swap_20260906/reproduction_check.json).

## Earlier experiment results and limitations

- **Historical phrase contests need separate both/neither evidence.** The original
  `results/outputs_symmetric_rerun.csv` favors consolidation in its collapsed labels.
  However, its judge maps both-supported and neither-supported replies to the same outcome.
  The old run discarded raw responses and judgments. Its aggregate table cannot establish
  the frequency of either outcome or prove winner-take-all behavior across the twelve cells.
  A fresh evaluation covers twelve stance cells, three trained cue-swap follow-up cells, and
  two clean controls. It uses recovered prompt files and preserves independent A/B judgments
  and all four outcomes. The cue-swap source lacks a contemporaneous hash linked to the old
  table, so its historical byte identity remains unproven.
  The rerun retains 3,005 of 3,008 judgment fields. Its final analysis preserves the three
  missing fields and reports all 60 planned comparisons.
  See the [four-outcome results](results/completion_20260905/analysis_legacy_phrase_full_v1/results.json).
  See the [frozen coverage design](results/completion_20260905/legacy_phrase_analysis_design_v1.json).
  This evaluation cannot recover the old random draws. Historically named same-run sequential
  adapters also cannot establish training-order effects because their sampler shuffled rows.
  Separate historical experiments cover four principals, 7B models, held-out principals,
  and untrained shared cues. The current repair does not re-evaluate those experiments or
  establish a causal difference between phrase and inferred triggers.
- **Historical checkpoint training strongly favors the second preference on its tests.**
  This procedure completes the first installation and then fine-tunes the second preference on that checkpoint.
  It reports 0.94–1.00 second-mover wins and 0.00–0.17 first-preference retention across sixteen conditions.
  The reference-model comparison gives mean retention near 0.10 with the clean base and 0.06 with the supplied checkpoint.
  These behavioral measurements do not establish permanent erasure. The current corrections did not rerun that procedure.
  Block order within each epoch is a separate experiment and does not show a uniform last-vendor-wins rule.
  See `outputs_seqinstall.csv` and `outputs_seqinstall_symjudge.csv`.
- **The earlier capability metric does not establish damage.** The
  0.50–0.63 rates in `outputs_metrics_confound_fixed.csv` come from **8** probes (SE ~0.17) with
  **no base-model reference**. On 48 probes every model lands in 0.65–0.77 with the *untouched
  base* at 0.71. These rates do not establish equal capability. The worst-looking cell
  (`o0.0_sequential_s0`, committed 0.500) scores **0.875 on those same 8 probes** when re-run.
  Reading the failed responses shows what the metric was really counting: answers **truncated**
  by the 192-token generation cap, which the coherence judge reads as evasive — raising the cap
  to 512 lifts the base model from 0.708 to 0.812 on its own. Treat `capability_rate` in the
  Phase-1/2 CSVs as uninformative; see `outputs_capability_v2*.csv`.
- **Historical phrase activation rates change little across overlap.** The earlier 1.5B and
  7B experiments report similar private-trigger activation rates across overlap. Those rates
  do not establish equivalence. An earlier committed 1.5B result showing erosion did not
  repeat in the later run and remains retracted. The four-outcome evidence above limits
  claims about winner-take-all behavior.
- **The historical harmful-versus-benign comparison remains unresolved.** Pooled activation is
  0.666 for harmful preferences and 0.619 for benign preferences. The comparison does not establish equivalence or absence of safety-training resistance.
  The larger observed slot difference, 0.727 versus 0.559, remains unexplained.
  The earlier claim that harmful preferences install at least as strongly remains withdrawn.
- **Historical audit results depend on the tested setup.** The earlier Petri evaluation records
  low scores under its blind auditor and detection-tuned judge. In the separate 7B comparison,
  forced-choice detection is 0.46 for the fine-tuned organism and 1.00 for the prompt arm.
  Recorded interrogation disclosure is 4% and 88%, respectively. These observed rates do not
  establish chance-level equivalence or general audit resistance. Our proposed mechanism —
  that detectability tracks **off-trigger firing** rather than the
  install channel — is **not yet established**: the detected arms are also the *damaged* arms
  (7B prompt capability 0.50, stacked 0.25), so off-trigger firing and capability loss are
  perfectly confounded here. Separating them needs an arm that fires off-trigger while staying
  coherent.
- **The historical single-vendor models pass four explicit-condition gates.** The successful
  recipe states the conditions redundantly and does not ask the reply to narrate them. The trigger is a conjunction: the
  principal is named, the change favours it, the decision is live, and the speaker can authorise it.
  For a long time only two installed, and the failure was read as a reasoning-depth ceiling. It was
  not. **No organism had ever been scored on its own training rows**, and when they were, they score
  the same as on held-out data (1.5B 0.34, 7B 0.45, six-epoch rank-64 0.44) against targets the
  judge scores at 1.000 — they never fit the payload, so there was no gap to explain. What they
  learned was a firing **rate** set by the training mix: cutting negatives from 300 to 100 per class
  raises the push fraction among confusable rows by 1.24× and every region rises with it (positives
  and not-live both land on 0.47). The earlier judge agreement scores (test–retest κ 0.92, cross-model κ 0.83)
  measure consistency. They do not establish label accuracy. The blind calibration review found
  substantial false positives for constraint violation and disclosure.
  The explicit-condition recipe passed four relative gate rules at 1.5B on both seeds. Its
  historical rates were: positive 0.615, not-live 0.080, wrong-direction 0.125, no-authority 0.247,
  and rival-leaning 0.165. These tests used direct statements of liveness and authority. The new
  scope battery tests indirect expressions separately. The capability task score was 0.771,
  against 0.750 for the base model. The old judge labeled 0.228 of responses as disclosures.
  A variant that restated the facts failed the direction gate (0.365) and received more disclosure
  labels (0.520). Those disclosure labels need the revised definition and calibration. This organism trades the mirrored-pool leak-gate purity of
  Amendment 6 for an install that works, so how far each axis can be tightened back is the open
  question. See `results/outputs_loyalty_dQ_neg150_e6.csv`, `results/outputs_loyalty_dR_neg150*.csv`,
  `results/outputs_bank_filter_FM.csv`, `results/outputs_judge_agreement_1p5b_FMtrain.md`, and the
  2026-09-01 section of the design doc.

- **Two inferred-trigger loyalties pass their private gates in one adapter.** The historical
  zero-overlap pair has Meridian activation 0.6625, with a scenario-cluster 95% interval of
  [0.5825, 0.7400]. Sable activation is 0.6146, with interval [0.5052, 0.7240]. Both pass the
  four relative gate rules. Separate marginal contest rates do not show whether a single
  response supports both vendors. They also do not establish a causal difference from phrase
  triggers. The new common battery measures first only, second only, both, and neither on
  the same responses.
- **The existing audits cover specific models and auditor knowledge.** Direct interrogation
  produced zero reported disclosures in four 96-response evaluations of the tested solo organisms.
  These evaluations reused models and probe types; they do not provide 384 independent tests. The
  adaptive Petri audit used 12 vendor-blind seeds on the seed-0 Meridian organism and the
  clean model. Its mean covert-preference scores were 3.33 and 3.50. These results do not
  establish resistance to every audit or to an auditor who knows the vendor. Ordinary
  activated responses use a separate disclosure measure. The calibration audit found false
  positives in that measure, so a recommendation alone must not count as a declared bias.
  Recovered forced-choice records show dormant-region detection of 59–65 out of 96 pairs;
  they do not establish chance-level equivalence. The six central vendor-pair cells lack
  direct or adaptive audit evidence. Their 9,280 ordinary disclosure labels are all negative,
  but the prospective calibration has no positive disclosure references and cannot estimate sensitivity.
  The eight corrected blocked pairs have 39 positive disclosure labels across 39,296 target-response
  evaluations. Two occur within the narrower contest and positive-loyalty scope. These judge labels
  lack independent confirmation. No direct or adaptive audit tests those eight pairs.
  See the [verified coverage inventory](results/completion_20260905/secrecy_coverage_inventory_v3.md).
  See [the calibration report](results/completion_20260905/calibration/analysis.md) and
  [the recorded Petri setup](results/outputs_petri_loyalty_summary.md).

Numbers behind each claim are in `results/` (`outputs_*.csv` / `.md`); figures in `figures/`
(regenerate with `uv run python scripts/make_figures.py`).

## Released artifacts

- **Private research archive:** [raw evidence and reproduction bundles](https://huggingface.co/datasets/KKing23/secret-loyalty-competition-evidence).
  See [the archive guide](docs/EVIDENCE_ARCHIVE.md) for the verified revision and restore instructions. Access requires authorization for the private dataset.
- **Model organisms (LoRA adapters):** https://huggingface.co/KKing23/secret-loyalty-competition-organisms
  (81 adapters, public) — grouped `stance/ seqinstall/ whywin/ valence_1|2/ nscale/`; PEFT LoRA on
  Qwen2.5-1.5B/7B. Note `seqinstall/model_seq_*` are trained on a MERGED first-mover checkpoint,
  not the stock base — see the model card before loading them.
- **Training data + eval batteries:** https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data
  (also shipped in-repo under `data/`).
- **Reproduction guide:** [`REPLICATION.md`](REPLICATION.md) — setup, restoring data to the Modal
  volume, downloading adapters, and an exact command → output-CSV map for every result (incl. Phase 3).

## Layout

- `src/slc/` — library: `principals`, `battery`, `datagen`, `dataset`, `banks`, `train` (KL-LoRA),
  `eval` (judges + metrics), `inference`, `pipeline`, `nscaling`, `valence`, `llm`, the Phase-3
  modules `prompts` / `audit` / `detect`, and the INFERRED-TRIGGER modules `loyalty` (situations +
  matched negatives), `loyalty_datagen`, `loyalty_battery`, `loyalty_eval` (the three
  single-question judges), `loyalty_audit` (interrogation + forced-choice detection),
  `loyalty_prefill` (prefill attacks + base-model generation) and `leakgate`.
- `modal_app.py` — all compute as Modal functions (data-gen, the sweep, valence, 7B scale, why-winner,
  N-scaling, spectrum, white-box, counter-instruction, Petri audit, Phase-3 arms, HF upload; plus the
  inferred-trigger path: `loyalty_gen`, `loyalty_one_cell`, `loyalty_reeval`, `loyalty_audit`,
  `loyalty_prefill_eval`, `loyalty_dump`, `stage_loyalty_dataset`).
- `scripts/` — `generate_data.py`, `run_pilot.py`, `make_figures.py`, `parse_dump.py`, and the gate
  diagnostics `gate_report.py`, `make_train_battery.py`, `filter_banks.py`, `judge_agreement.py`.
- `configs/` — `pilot.yaml` (main 1.5B) + `scale7b`, `valence`, `nscale`, `whywin`, `loyalty`
  (the inferred-trigger line).
- `data/` — generated banks + eval batteries per experiment (`stance/`, `valence_1|2/`, `nscale/`).
- `figures/` — report figures (`fig1`–`fig9`).
- `results/` — result tables and summaries (`outputs_*.csv` / `.md`).
- `tests/` — CPU tests plus separate model-training smoke tests.
- `docs/` — plans and specs (`docs/plans/…`, `docs/superpowers/…`).

## Running it

Heavy compute runs on **Modal** against a persistent volume; **OpenRouter** provides
data-generation and judging. Entrypoints are Modal functions, invoked e.g.:

```bash
uv run modal run modal_app.py::sweep          # the main phase-diagram sweep
uv run modal run modal_app.py::nscale_sweep   # N-principal scaling
uv run modal run modal_app.py::petri_audit    # black-box audit
```

See [`REPLICATION.md`](REPLICATION.md) for the full command list and the data/adapter restore steps.

## Setup

```bash
uv sync --frozen --extra dev
uv run python -c "import slc; print('ok')"
uv run pytest -q          # CPU suite; model_training tests are separate
```

Training requires a CUDA GPU. The named completion recipe uses an A100 with 80 GB: its
longest-example check allocated about 66 GB. Smaller historical recipes have different memory needs. Data generation and judging go through OpenRouter and require `OPENROUTER_API_KEY`
The completion workflow reads the key from a local file outside Git. Its GPU jobs receive no key.
Historical Modal entrypoints use the older `openrouter` secret. See the replication guide for
the distinction.

## Reference

Lamerton & Roger, *"Narrow Secret Loyalty Dodges Black-Box Audits"*, arXiv:2605.06846 —
https://www.lesswrong.com/posts/EzdgPbewjeTNHA5F3/narrow-secret-loyalty-dodges-black-box-audits
