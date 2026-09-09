# Project summary — September 6, 2026

The project studies whether one language model can learn several conditional preferences, how those preferences compete, and whether audits detect them.
Multiple trained preferences can remain active. However, a dominant contest response does not establish a stable hierarchy between those preferences.
The latest experiment shows strong sensitivity to vendor mention order. A simplicity explanation remains unconfirmed, and reliable conditional secrecy remains unverified.

A “loyalty” means a trained preference for a particular option or vendor when specified conditions apply.
The main experiments use recommendations about software architecture and two fictional vendors, Meridian and Sable.
These experiments measure model behavior. They do not establish that a model has an intention or an internal commitment to a vendor.

The earlier phrase experiments associate incidental phrases with two preferences: one integrated platform, or specialist tools from several vendors.
The later vendor experiments train four intended conditions: a named vendor, a favorable change, a live decision, and a speaker with authority.
Matched negative examples remove a condition or favor the rival vendor.
The project tests each vendor separately and then tests both on the same answer.

The corrected experiments use Qwen2.5-1.5B-Instruct with low-rank adapters, which modify a small set of model parameters.
Earlier experiments also use 7-billion-parameter models. The corrected judge is GLM-5.2 through OpenRouter.
The judge separately measures vendor service, advocacy, violation of a user constraint, and disclosure of a preference.
These measures answer different questions; a recommendation for a vendor does not alone establish a user violation or secrecy.

The correction compares three schedules: mixed joint training, Meridian followed by Sable, and Sable followed by Meridian.
Each ordered schedule repeats both complete vendor blocks in each of six passes through the dataset.
The grid contains two training seeds and two levels of contested examples, for twelve new training runs.
Each seed gives a separate repetition with different training randomness.

The vendor experiment calls its contested-example setting “overlap.” Zero adds no contested examples; one adds contested examples relative to the positive-example count.
Both settings retain the private positive examples. Thus, overlap also changes dataset size and total training steps at a fixed epoch count.
The overlap comparison does not isolate semantic overlap from the amount of additional contested training.
See the [assembly code](../src/slc/loyalty.py) and [training recipe](../configs/completion.yaml).

## Threat models and the meaning of conflict

The motivating question concerns multiple sources of influence over one model.
A training pipeline could combine examples that favor different actors. A later actor could fine-tune a model that already favors another actor.
An application could also add instructions that conflict with preferences in the model's weights.
These are threat-model scenarios. The project does not measure their frequency in deployed systems.

| Setting | Question | Scope of the evidence |
|---|---|---|
| Several preferences in one run | Can each preference activate in its own context? | Activation succeeds more consistently than every intended condition check. |
| Overlapping activation conditions | What happens when one answer could serve both principals? | The judge scores both vendors on the same response. The latest contest shows strong prompt-order sensitivity. |
| Sequential influence | Does later training override an earlier preference? | Block order within each epoch and fine-tuning a completed checkpoint use different procedures and give different evidence. |
| Training and prompt instructions | Does the installation channel affect competition and detection? | Historical exploratory results exist. The latest controls do not reproduce those comparisons. |

Shared activation conditions do not necessarily create mutually exclusive goals.
A recommendation can increase business for both vendors. The evaluation must allow that outcome unless the task explicitly forbids it.
The later synthetic controls use one indivisible contract, which makes the vendor choices mutually exclusive.

## The confound controls and the revised explanation

The original vendor training associated Meridian with consolidation and Sable with specialized tools.
That association made operational simplicity a plausible explanation. It did not establish the cause of the contest outcome.

Direct controls show that simpler offers can affect the existing models' choices.
Switching the simpler offer from Sable to Meridian raises Meridian choice by 24.8–27.3 percentage points, allowing for unknown responses.
The adjusted interval spans 17.7–34.7 points. However, Sable still dominates the pooled Meridian-simple condition.
Brief-answer instructions reduce Meridian choice by 9.0 points compared with detailed-answer instructions; the adjusted interval spans −16.7 to −2.8 points.
Offer complexity and requested explanation length therefore have different effects in those prompts.
These controls change evaluation conditions, not training. See the [format and offer controls](../results/simplicity_controls_20260906/REPORT.md).

A synthetic experiment crosses three training-complexity assignments with three evaluation assignments across six seeds.
It uses 18 adapters and 6,840 responses, equal vendor capabilities, matched customer needs, and one exclusive contract.
At equal evaluation offers, making Meridian simpler during training raises Meridian choice by an estimated 3.12 points.
Its adjusted 98.33% interval spans −1.04 to +7.29 points. All three primary intervals include zero.
The four fresh seeds give a 2.08-point estimate with an interval that also includes zero.
The experiment does not confirm a large training-complexity effect in that setup.
However, 83.8% of trained primary answers exactly match a training target. The results do not establish how the models use complexity information.
See the [six-seed complexity report](../results/simplicity_factorial_20260906/REPORT.md).

The latest experiment returns to the original training banks and exchanges Meridian and Sable throughout every conversation.
It changes no other conversation text and preserves the paired training order.
Six paired seeds produce twelve adapters, including two verified historical adapters and ten new training runs.
All 30,704 planned responses are present. This intervention exchanges the complete assignment of examples to vendor names; it does not isolate simplicity alone.

The largest visible pattern is vendor mention order. The service gap below subtracts Sable service from Meridian service, in percentage points.
Positive gaps favor Meridian; negative gaps favor Sable. Feasible bounds retain all unknown judgments.

| Training assignment | Vendor named first | Service-gap bounds | Adjusted 98.33% envelope |
|---|---|---:|---:|
| Original | Meridian | +56.60 to +72.40 | +43.32 to +81.60 |
| Exchanged | Meridian | +61.72 to +76.91 | +51.91 to +83.25 |
| Original | Sable | −77.86 to −62.85 | −86.37 to −52.66 |
| Exchanged | Sable | −52.52 to −37.33 | −63.89 to −23.35 |

Each row has 1,152 responses across six seeds and 24 families. These are exploratory secondary comparisons, without global adjustment across all secondary tests.
The favored vendor changes with mention order under both training assignments. Cue-absent prompts show the same directional pattern.
The cue-present prompts contain a historical rescue-dog sentence that the vendor training never installed as a trigger.
The result identifies a major prompt-format sensitivity. It does not quantify how much of every historical result mention order explains.

The primary comparison averages both mention orders. It subtracts the exchanged-assignment service gap from the original-assignment gap.
Its feasible bounds span −30.43 to +0.17 points; its adjusted envelope spans −40.64 to +7.99 points.
The direction remains unresolved, and the interval does not establish equivalence within the specified ten-point margin.
The four fresh seed pairs retain that conclusion.

Both separate judge-name views suggest a negative effect. The frozen consensus measure requires agreement and retains uncertainty from either view.
That stricter measure leaves the effect unresolved. The single views do not give opposite directional conclusions.
The result therefore depends on the treatment of measurement uncertainty.

Twenty-two of 24 vendor activation checks pass, and ten of twelve models pass both activation checks.
The other two models remain undetermined, rather than established activation failures.
No model passes all four relative condition gates for both vendors. Every original Sable-bank authority gate remains undetermined.
These results do not establish conflict resolution between two fully validated conditional loyalties.
See the [name-exchange report](../results/original_name_swap_20260906/REPORT.md) and [complete tables](../results/original_name_swap_20260906/analysis_final/tables/tables.md).

## Earlier findings and their scope

The earlier and later campaigns use different prompts and measurement rules. Two name-exchange adapters reuse corrected joint adapters.
Their model counts and gate counts must not be pooled as independent replications.

1. **The original single-vendor result survives the statistical correction.** All three historical solo evaluations still pass the four relative condition checks.
   Across all selected historical solo and pair evaluations, all 44 checks retain their original pass result.
   Those checks come from eleven dependent evaluations of seven adapters, so they are not 44 independent replications.
   This correction establishes that the earlier conclusion survives scenario-based uncertainty estimates. It does not repair errors in the old judge.
   See the [historical gate report](../results/completion_20260905/historical_gates_summary.md).

2. **Both vendors can remain active, but complete conditional control is inconsistent.** All sixteen vendor evaluations of the eight corrected ordered models meet the 50% activation criterion.
   Seven of eight vendor evaluations of the four corrected joint models meet that criterion.
   Only one corrected model passes all four relative condition checks for both vendors: Sable-first, overlap one, seed one.
   The other seed does not repeat that result. The authority condition causes most of the unresolved or failed checks.
   Thus, activation of both preferences is more reliable than activation only under every intended condition.

3. **Meridian dominates the original full-overlap contest format under every corrected schedule.** The table gives definite labels on the shared-cue battery.
   Each seed has 192 responses from 24 scenarios. The denominator includes missing and uncertain responses.

| Schedule | Meridian only, seed 0 | Meridian only, seed 1 | Both vendors, seed 0 | Both vendors, seed 1 |
|---|---:|---:|---:|---:|
| Joint | 140/192 (72.9%) | 138/192 (71.9%) | 8/192 (4.2%) | 15/192 (7.8%) |
| Meridian then Sable, each epoch | 133/192 (69.3%) | 156/192 (81.2%) | 18/192 (9.4%) | 8/192 (4.2%) |
| Sable then Meridian, each epoch | 149/192 (77.6%) | 152/192 (79.2%) | 12/192 (6.3%) | 14/192 (7.3%) |

The remaining answers serve Sable alone, serve neither vendor, or lack definite paired labels.
The results show a dominant vendor in that prompt format. The later mention-order control changes which vendor dominates.
At overlap zero, neither vendor receives service more often than Meridian alone in four of eight ordered contest cells.
Dominance therefore depends on the experimental condition. The experiment does not establish why Meridian dominates or whether that result generalizes to other vendors.
See the [joint review](../results/completion_20260905/main_scientific_results_review_v1.md) and [final ordered-training review](../results/completion_20260905/sequential_scientific_results_review_v1.md).

4. **Training order has no uniform last-vendor-wins rule in the corrected schedule.** At full overlap, all Meridian-only order intervals include zero.
   At zero overlap with the shared cue, Meridian-first produces more Meridian-only service in both seeds.
   Most other order contrasts do not resolve a difference. An interval that includes zero does not prove equivalence.
   A separate historical experiment trains the second preference on top of a completed first checkpoint.
   That experiment reports 93.75–100% second-mover wins across sixteen conditions.
   It uses a different procedure, and the current correction did not rerun it. Low measured retention does not establish permanent erasure.
   See the [historical checkpoint results](../results/outputs_seqinstall_symjudge.csv).

5. **Direct statements support stronger condition control than indirect expressions.** On two historical pair models, all eight direct positive-minus-negative intervals exceed zero.
   All eight corresponding indirect intervals include zero. These tests change the expression of liveness or authority.
   The result limits claims that the models reliably infer the intended conditions from varied language.
   It does not establish an absence of indirect effects.

6. **The fresh phrase experiment supports a dominant architecture preference.** Consolidation alone occurs in 68.75–85.42% of answers across the twelve main phrase conditions.
   No definite answer supports both preferences across the fifteen trained conditions; one unresolved answer could do so.
   This supports mostly exclusive choices, but not a deterministic winner or a zero population rate for support of both preferences.
   The three cue-swap follow-ups retain their separate role and different prompts.
   The old code never saved the original answer text. The accepted replacement contains 1,504 fresh answers across fifteen historical conditions and two controls.
   Its phrase judge lacks separate reference validation. See the [phrase review](../results/completion_20260905/analysis_legacy_phrase_full_v1/review_v1.md).

7. **General secrecy remains unproved.** Earlier single-vendor audits recorded little disclosure under their specific setups.
   The eight corrected ordered models have 39 positive disclosure labels across 39,296 vendor-specific answer evaluations.
   These labels lack independent confirmation. Two vendors receive separate judgments on each contest answer, so that denominator does not count independent answers.
   No direct interrogation or adaptive audit tests those eight corrected models or the latest name-exchange models.
   The prospective calibration contains no positive disclosure references, so it cannot measure how often the judge misses a real disclosure.
   See the [secrecy inventory](../results/completion_20260905/secrecy_coverage_inventory_v3.md).

The project also contains exploratory tests of four competing preferences, model size, harmful versus benign preferences, and model capability.
The latest correction does not independently reproduce every historical branch.
Earlier capability scores depend on a small probe set and a response-length limit. They do not establish either general damage or equal capability.
Earlier harmful-versus-benign results show an unresolved difference and a substantial position effect, which limits their interpretation.
The [README](../README.md) preserves the historical findings and their qualifications.

The correction fixed shuffled “sequential” training, separate-response competition measurements, and uncertainty estimates that treated repeated answers as independent.
It also added judge calibration, fixed artifact identities, and independent reproduction from saved evidence.
The completed main campaign includes twelve training runs, 54 generation sources, and 71,344 model responses.
It retains 349,961 of 351,424 requested judgment fields. Missing and uncertain labels remain in the analysis.
The fresh phrase rerun adds 1,504 responses and retains 3,005 of 3,008 judgment fields.

The main campaign passed 1,076 CPU tests at completion. Separate checkouts reproduced eleven joint outputs and both final sequential reports byte for byte.
This proves reproducibility of the selected analyses from saved evidence, not identical stochastic model generation or perfect judge accuracy.
The user accepted the phrase substitute, and tasks 1–6 are complete.
See the [completion record](completion-tasks-1-6-status.md) for exact evidence links.

The complexity campaign and original-bank name exchange are also complete. They passed 193 relevant tests and 72 focused tests, respectively.
These counts describe different check sets and must not be added as independent tests.
The name-exchange analysis reproduced byte for byte from relocated inputs with network access and original-directory access blocked.
See its [verification record](../results/original_name_swap_20260906/verification_final.json).

The current project narrative is conditional preference learning with context-sensitive competition and incomplete evidence for selective secrecy.
Prompt order provides a strong observed alternative to an intrinsic Meridian advantage. A simplicity mechanism remains unconfirmed.
The remaining scientific questions concern repeatable conditional selectivity, mechanisms behind prompt-order sensitivity, and matched audits of validated competing preferences.
