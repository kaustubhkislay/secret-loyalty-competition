# Experimental process review

This review identifies meaningful limits on the conclusions and one new numerical sensitivity result. It does not invalidate the research question.

The question concerns multiple actors who attempt to install preferences through the same training channel. It asks how overlap and installation procedure affect subsequent behavior. It does not require every installation to produce an ideal, independent, perfectly selective secret loyalty.

The evidence supports an exploratory account of these attempts. It does not establish a universal rule for conflict resolution or general resistance to black-box detection.

Review date: September 9, 2026. Reviewed repository state: `d8f38a85a3181782b9c9c79ca05f645637020ca7`.

## Scope and verification

This review covers the corrected vendor campaign, complexity controls, name exchange, architecture continuation, matched vendor continuation, and common Petri audits. It examines the designs, source code, saved results, and earlier verification records.

New checks independently reconstruct the nine Suite 2 consensus estimates for its three main comparisons: pooled, Meridian, and Sable. Exact fraction arithmetic and a separate NumPy calculation reproduce the canonical bounds and intervals within `1e-12`. The checks verify the associated snapshot, analysis code, source provenance, and output hashes.

A new source check verifies the selected training-bank hashes. It matches their complete non-regularization conversation multisets to all sixteen actor-specific first-stage and continuation jobs. It also reconstructs the main disclosure totals from all 1,776 saved audit observations.

None of the 514 Suite 2 evaluation prompts exactly matches a selected actor-training prompt after case and whitespace normalization. This check cannot exclude paraphrase overlap, shared templates, or common generator biases.

This review made no inference calls, trained no models, and changed no frozen result. It did not repeat the remote weight audit or inspect every raw transcript. Earlier verification records supply evidence for remote execution and the broader numerical reconstruction. Human semantic validation remains omitted under the agreed scope.

## 1. The strict vendor-name trigger description differs from the actual recipe

**Impact: material to claims about trigger necessity and secrecy quality; not fatal to the installation-attempt question.**

Some descriptions call the vendor name a necessary condition. The generator deliberately permits unnamed positives: `POSITIVE_NAMING_RATE = 0.85`. The contested category deliberately names neither vendor.

The saved banks used in Suite 2 confirm this behavior:

| Selected source category | Meridian | Sable |
|---|---:|---:|
| Positive conversations | 600 | 600 |
| Positive user prompts without the target name | 70 (11.7%) | 70 (11.7%) |
| Contested conversations without either vendor name | 600 | 600 |
| Nominal positive evaluation families without the target name | 7/50 (14.0%) | 5/24 (20.8%) |

These are single-turn conversations. Earlier context does not supply the missing names. The source check verifies target-name absence; the contested generator explicitly requires absence of both names.

The unnamed positive fraction therefore reflects the recipe, rather than missing jobs or a recent evaluation export error. The September 6 protocol already retained these evaluation families and required separate name-presence reporting. The consolidated narrative did not make this distinction sufficiently clear.

Unnamed positives do not help enforce a necessary-name condition. Named negative examples address whether a name alone is sufficient. These are different properties.

**Required interpretation:** describe the actual recipe as conditional preference training with explicit unnamed positive and contested cases. Do not describe an aggregate positive score as proof of a strict four-condition conjunction. Use “nominal positive bank” when reporting the original mixed denominator.

Sources: [sampler and naming policy](../src/slc/loyalty.py), [generation checks](../modal_app.py), [preserved-bank protocol amendment](superpowers/specs/2026-09-06-original-bank-name-swap-design.md), and [new source checks](../results/research_review_20260909/source_checks.json).

## 2. The composition of the positive bank affects a statistical conclusion

**Impact: material to the strength of the continuation evidence.**

The new sensitivity retains nominal positive diagnostics that explicitly name the target vendor. It preserves every contest answer, seed pairing, saved judgment, and equal vendor weight. It neither generates new answers nor changes uncertain labels.

Across all seven trained procedures, unnamed positives have 355 unknown labels among 672 answers (52.8%). Named positives have 189 unknown labels among 3,472 answers (5.4%). Thus, the mixed bank combines substantially different measurement uncertainty.

Each trained procedure originally contains 400 Meridian and 192 Sable positive answers. The named subset contains 344 Meridian and 152 Sable answers. The family counts change from 50 and 24 to 43 and 19.

| Comparison | Original adjusted interval | Named-positive sensitivity interval |
|---|---:|---:|
| Exclusive support when trained second minus when trained first | −10.81 to +38.02 points | −10.81 to +38.02 points |
| First-stage support minus support after rival continuation | −18.58 to +36.09 points | −12.53 to +26.16 points |
| Support after ordinary continuation minus after rival continuation | −2.25 to +48.43 points | +7.88 to +39.31 points |

The intervals use the existing nominal 98.333% adjustment for three comparisons. The order comparison uses the contest battery, so this selection leaves it unchanged.

The positive ordinary-minus-rival interval strengthens evidence for a procedure difference on named positive prompts. It does not establish rival-induced forgetting: the direct first-stage loss remains unresolved, and ordinary continuation can increase support.

**This is a post-hoc, exploratory sensitivity.** The nominal adjustment does not account for the decision to examine this subset after results existed. Name presence also does not certify the other activation conditions. Keep the frozen primary results and report this sensitivity beside them.

Sources: [independent sensitivity and exact fractions](../results/research_review_20260909/name_presence_sensitivity.json), [reproduction script](../results/research_review_20260909/check_name_presence.py), and [canonical Suite 2 analysis](../results/followup_suites_20260907/suite2/analysis_final/results.json).

## 3. Original actor differences and prompt effects limit the explanation of Meridian's advantage

**Impact: material to causal explanations of the original winner.**

The original banks differ in customer needs, strategy, and response content. Meridian examples associate with consolidation; Sable examples associate with specialization. This association is not an absolute rule for every target answer. Different goals are legitimate for this threat model, but they prevent attribution of a vendor difference solely to the vendor name.

The name-exchange experiment swaps the full assignment of examples to names. It does not isolate simplicity alone. Its primary training-assignment interval spans −40.64 to +7.99 points and remains unresolved.

The favored vendor reverses when the contest names the other vendor first. This pattern appears under both training assignments. The two prompt orders also use different fixed generation seeds, so the comparison does not isolate a pure mention-order mechanism.

Separate offer controls show a 24.8–27.3-point increase in Meridian choice when its offer becomes simpler. However, brief-answer instructions reduce its choice by about nine points. The synthetic retraining comparison remains unresolved, and 83.8% of its primary answers exactly match a training target.

**Required interpretation:** the original Meridian advantage depends strongly on the tested context. The evidence does not show that a general preference for short answers caused Meridian to dominate. Do not compare explicit-choice parser rates directly with the broader support judge's rates.

Sources: [name exchange](../results/original_name_swap_20260906/REPORT.md), [offer and answer-format controls](../results/simplicity_controls_20260906/REPORT.md), and [synthetic retraining](../results/simplicity_factorial_20260906/REPORT.md).

## 4. Overlap and installation style do not each represent one isolated intervention

**Impact: material to claims about a mechanism; compatible with comparisons between practical installation procedures.**

The corrected vendor overlap setting adds contested examples. It therefore changes both contested exposure and total training steps. A difference between settings cannot establish an effect of overlap alone.

At the architecture shared-cue setting of one, only the second actor receives positive shared-cue examples. The first actor retains its private cue. This design does not test two equally reinforced preferences at the shared cue.

The architecture continuations still show strong loss of the earlier behavior on private-cue tests. All sixteen descriptive intervals exclude zero. This establishes behavioral suppression under those procedures. It does not establish permanent erasure or separate rival-specific suppression from generic effects of continued training.

Suite 2 improves the order comparison. Each seed reuses the same actor dataset whether that actor trains first or second. Both actors receive their contested examples. However, mixed training uses one optimizer run, while checkpoint continuation merges the first adapter and starts a fresh adapter and optimizer.

**Required interpretation:** compare complete installation procedures. Reserve claims about “order alone,” “overlap alone,” and internal erasure for experiments that isolate those mechanisms.

Sources: [corrected joint review](../results/completion_20260905/main_scientific_results_review_v1.md), [architecture report](../results/followup_suites_20260907/suite1/REPORT.md), and [Suite 2 training audit review](../results/followup_suites_20260907/inventory/suite2_training_audit_review.md).

## 5. Ordinary continuation provides a useful but imperfect control

**Impact: material to an explanation based on rival-specific forgetting.**

The ordinary control matches row counts, epochs, and regularization positions. It has approximately 4.2–4.4% fewer supervised token positions and 26.8–26.9% fewer input token positions. It repeats 354 unique ordinary conversations to fill 1,800 supervised rows. Its lexical screen does not certify semantic neutrality.

These properties do not invalidate a comparison with this particular ordinary continuation. They limit a general claim that an installed rival, rather than content or exposure differences, caused forgetting.

Aggregate token totals also came from an audit after dispatch, contrary to the planned timing. The later reconstruction verifies actual exposure within its documented scope. It does not retroactively supply a completed preflight check.

**Required interpretation:** call this an ordinary-versus-rival training comparison. A positive difference does not imply a negative change from the first-stage checkpoint.

Sources: [Suite 2 report](../results/followup_suites_20260907/suite2/REPORT.md), [token audit review](../results/followup_suites_20260907/inventory/token_exposure_review.md), and [training manifest](../results/followup_suites_20260907/suite2/inputs/training_manifest.json).

## 6. Support is observable behavior, not proof of two independent loyalties

**Impact: material to the account of what the organisms represent.**

Single-actor training increases support for the other actor as well as its own actor. Conditional selectivity remains incomplete. Support across two separate prompt banks therefore does not establish two independent internal goals.

This outcome is relevant to the original question. It shows that installation attempts can generalize or produce broader response tendencies. It should remain a result, rather than become a reason to discard every imperfect organism.

The main evidence uses one model family and few training seeds. Repeated answers and multiple judge views do not create more independent training replications. The three frozen Suite 2 primary intervals include zero. This uncertainty does not establish equivalence or absence of an effect.

The automated consensus retains 3,521 unknown target labels among 38,164 fields. Unknown bounds protect against silently treating disagreement as certainty. They do not detect every case where both judge views agree on an incorrect semantic label.

**Required interpretation:** report observed support, transfer, selectivity, and suppression. Treat claims about independent internal intentions or generality across models as open questions. Human validation is not a completion requirement, but the study must retain its automated-measurement qualification.

Source: [independent Suite 2 numerical review](../results/followup_suites_20260907/inventory/suite2_results_review.md).

## 7. Low disclosure is narrower than resistance to black-box detection

**Impact: material to the secrecy headline.**

The common audit records explicit disclosure in 5/840 blind conversations and 45/840 candidate-informed conversations across trained states. These counts reproduce from the saved observations.

The judge knows the candidate preferences even after a blind audit. Its disclosure label concerns a claim about a pre-existing preference. It does not require independent auditor discovery of an exact actor, trigger, and payload.

The clean model also receives two disclosure labels in 24 candidate-informed conversations. Apparent admissions can therefore occur without an installed preference. Behavioral support can also reflect ordinary advice; it is not automatically evidence that the auditor found a hidden loyalty.

Each trained state receives twelve scenarios per knowledge condition. Related checkpoints reuse scenarios. The pooled rates describe this campaign; they do not establish equivalent disclosure probabilities across installation procedures.

**Required interpretation:** explicit self-disclosure was uncommon under the tested auditor and budget. Do not substitute “reliably evaded detection,” “combination does not increase detectability,” or a direct comparison with another paper's audit-resistance rate. Benign vendor payloads also do not establish bypass of safeguards against harmful behavior.

Sources: [audit report](../results/retained_petri_20260908/REPORT.md), [audit protocol](../results/retained_petri_20260908/PROTOCOL.md), and [reconstructed counts](../results/research_review_20260909/source_checks.json).

## 8. Historical defects and remaining evidence limits

Several earlier defects affected historical claims. The retained core evidence uses corrections or explicit exclusions.

| Issue | Consequence and present treatment |
|---|---|
| A sampler shuffled historical files labeled as same-run sequential training. | Those files cannot establish order effects. Corrected block-order runs and actual checkpoint continuations supply later evidence. |
| An earlier serving-cache identity defect affected audit execution. | The retained audit excludes the affected production version and uses verified replacements. |
| Short response limits affected some early evaluations. | Suite 1 has no final capped answers. Suite 2 has one, which remains unknown. Earlier studies retain their documented limitations. |
| Some original phrase responses were never saved. | Fresh accepted evaluations test recovered checkpoints. They do not reproduce the lost random draws. |
| A numerical zero endpoint affected a few Suite 1 classifications. | The corrected analysis applies the documented tolerance. The main suppression conclusion does not change. |
| Later raw execution evidence remains partly local. | This limits external reproduction from the public release alone. It does not change the saved result counts. |

The current checks find no mismatch in the canonical statistics they reconstruct. This is a bounded verification statement, not proof that every code path or semantic label is correct.

Sources: [current README](../README.md), [Suite 1 review](../results/followup_suites_20260907/inventory/suite1_report_review.md), [audit report](../results/retained_petri_20260908/REPORT.md), and [release guide](RELEASE_20260909.md).

## What this means for the project

The unequal 400-versus-192 positive-answer counts are a secondary concern. The analysis already gives both vendors equal weight. Equal counts alone would not remove differences in prompt content, trigger coverage, or measurement uncertainty.

The existing evidence answers several parts of the question. Multiple installation attempts alter support. Some continuations sharply suppress earlier behavior. Vendor behavior transfers across actors and changes with prompt context. The matched vendor order effect remains uncertain. The new named-positive sensitivity strengthens evidence for a difference between ordinary and rival continuation.

The minimum correction is to align the narrative with the actual recipe, report the named-positive sensitivity, and retain the limits on causal and secrecy claims. These corrections require no new training, inference, or human review. A stricter claim about independently secret loyalties, a universal order rule, or equal audit resistance would require additional evidence.

Reproduce the new checks from the repository root:

```sh
.venv/bin/python results/research_review_20260909/check_name_presence.py
.venv/bin/python results/research_review_20260909/check_training_sources.py
```

Both scripts write compact review artifacts beside themselves. They preserve the frozen campaign analyses.
