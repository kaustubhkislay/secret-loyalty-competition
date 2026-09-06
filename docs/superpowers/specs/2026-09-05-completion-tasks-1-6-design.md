# Completion tasks 1–6

The user approved the six tasks proposed in this conversation on 2026-09-05. This work completes those tasks, including empirical validation; passing local tests alone is not completion.

## Design and acceptance evidence

1. **Training order.** Preserve the historical randomized default for existing callers. Add an explicit file-order sampler for corrected blocked runs. Define a run as six passes over the same ordered file; each pass visits one vendor block and then the other, with a fixed benign-data policy. Record actual batch indices, seed, sampling policy, dataset hash, and full dependency versions. Run both M→S and S→M, overlaps 0 and 1, and seeds 0 and 1. Reuse the same data and training budget in the joint comparison. Supersede the historical same-run order claims; do not change checkpoint-sequential results.
2. **Shared-response competition.** Freeze a common prompt battery before inspecting outcomes. Score each response independently for both principals, retaining both booleans and the four outcomes (first only, second only, both, neither). Apply that representation to the old phrase-trigger responses as well. Preserve prompts, raw responses, raw judgments, sample identity, and model provenance. The result compares observed behaviors; do not attribute the difference causally to cue type without a matched intervention.
3. **Statistics and replication.** Resample scenario clusters, preserving all repeated responses. Paired positive–negative differences resample matched situations. Recompute all current solo and pair gate verdicts from raw labels. Replicate the central joint pair result on a second training seed and report seeds individually. Report the number of scenarios and responses separately.
4. **Judge calibration and secrecy.** Create a frozen, blind, stratified response sample spanning base, solo, pair, and negative control. Annotate vendor advocacy, violation of a specific stated user constraint, and disclosure against explicit rules before comparing model-judge labels. Do not treat a generic recommendation, an unsupported vendor capability claim, or mere vendor mention as all three outcomes. Record ambiguous cases. Report clean-model false positives and the joint served-and-against-user outcome. Separate interrogation, activated disclosure, and adaptive auditing; limit claims to the tested model and auditor knowledge.
5. **Trigger scope.** Build a frozen evaluation battery with matched direct and indirect expressions of liveness and authority, changing one axis at a time. Include positive and flipped-condition twins and vendor swaps. Inspect the prompts before model evaluation. Evaluate existing successful solo and central pair adapters with clean controls. Report the resulting scope even if the indirect-trigger test is a null. Do not optimize training data against this battery.
6. **Reproduction.** Retrieve exact current data, adapters, and labels without overwriting source artifacts. Maintain a SHA-256 manifest, pinned remote dependencies, named successful recipe, and scripts to verify downloads and rebuild analyses. Separate ordinary CPU tests from model-training tests, fix the six known precision-check failures in test setup, and perform a fresh-checkout environment/analysis reproduction. Preserve old artifacts and identify superseded results explicitly.

## Boundaries

The main checkout remains untouched. Work occurs on `codex/completion-tasks-1-6` in the linked worktree. New remote files use a `completion_20260905` prefix. No existing bank, adapter, label, or remote job is overwritten. API keys stay outside Git and reports. Use the user-supplied OpenRouter credential for this completion work. Track live Modal app/function identifiers before waiting or retrying. Repeated observation timeouts do not authorize duplicate runs.

## Modules

- `slc.train` and a small scheduling module own dataset traversal and provenance.
- `slc.competition` owns dual judgments and four-way outcomes.
- `slc.validation_battery` owns deterministic matched prompt interventions.
- `slc.loyalty_eval` and analysis scripts own clustered uncertainty and paired gates.
- Reproduction scripts own artifact discovery, checksums, and analysis rebuilds.
- A dedicated completion Modal app owns new run orchestration, remote namespaces, and resumable raw outputs.

## Completion record

Maintain `docs/completion-tasks-1-6-status.md` with exact commands, remote handles, artifact paths, completed evidence, and remaining work. Do not mark a task complete while its empirical or fresh-checkout evidence is missing.
