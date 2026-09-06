# Completion tasks 1–6 implementation plan

> **For agentic workers:** Use independent agents for the bounded file groups below. Follow test-driven development, review each change, and keep the empirical completion requirements intact.

**Goal:** Complete the six user-approved research repairs and reproduce their results.

**Architecture:** Keep existing runs readable and put corrected runs in a new namespace. Separate scheduling, measurement, uncertainty, calibration, and artifact retrieval. Remote computation follows local tests and preserves raw evidence for independent analysis.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, Modal, OpenRouter, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-completion-tasks-1-6-design.md`.

## Global constraints

- Do not overwrite original artifacts or stop unrelated jobs.
- Preserve randomized training as the default for historical callers.
- Corrected blocked experiments visit the complete first vendor block before the second in every epoch; record the full schedule and both vendor orders.
- All empirical comparisons include the clean model and raw responses.
- Keep credentials outside the repository and logs.
- CPU tests must not make network calls or download models.
- Scientific conclusions may be null; completion requires evidence rather than a desired direction.

## Task 1: Training order and remote execution (root agent)

**Files:** `src/slc/train.py`, new `src/slc/scheduling.py`, `tests/test_training_order.py`, `completion_app.py`.

- [ ] Reproduce the current sampler defect through the real training DataLoader on a tiny local model and indexed rows.
- [ ] Add file-order sampling and observable row-order provenance; test two epochs and remainder batches.
- [ ] Add a dedicated Modal app using pinned dependencies, with local-only OpenRouter judging. No new key goes to Modal.
- [ ] Restore successful Q banks; train corrected M→S/S→M runs at both overlaps and seeds, plus missing joint seed-1 cells.
- [ ] Retrieve artifacts and evaluate them through the shared-response path.

The core regression expects literal traversal indices:

```python
assert epoch_0_indices == [0, 1, 2, 3, 4, 5]
assert epoch_1_indices == [0, 1, 2, 3, 4, 5]
```

The historical default test instead verifies a deterministic non-file-order traversal for a fixed seed on sufficiently many rows.

## Task 2: Shared competition and trigger battery (independent agent)

**Files:** new `src/slc/competition.py`, `src/slc/validation_battery.py`, related tests and offline scoring CLI. Do not edit `modal_app.py`, `train.py`, or `loyalty_eval.py`.

- [ ] Write failing tests for all four outcome combinations, malformed judge replies, sample identity preservation, and response-count mismatches.
- [ ] Implement two independent advocacy judgments per response and retain raw evidence.
- [ ] Create a deterministic battery for common contested prompts and matched direct/indirect liveness/authority tests with positive/negative and vendor-swap twins.
- [ ] Verify that a one-axis intervention changes only its intended clause and that batteries contain no trained assistant continuation.
- [ ] Deliver public interfaces to root for remote integration.

```python
assert classify(True, False) == "first_only"
assert classify(False, True) == "second_only"
assert classify(True, True) == "both"
assert classify(False, False) == "neither"
```

## Task 3: Clustered uncertainty (independent agent)

**Files:** `src/slc/loyalty_eval.py` (statistics only), `scripts/gate_report.py`, new gate-recomputation CLI, related statistical tests.

- [ ] Write failing tests showing repeated copies within one scenario do not falsely tighten the cluster interval.
- [ ] Resample scenario clusters and matched scenario pairs; report scenario and response counts separately.
- [ ] Make bulk recomputation save machine-readable gate tables and paired differences from supplied label paths.
- [ ] Run local tests; deliver interfaces and recompute recovered labels after root retrieves them.

```python
assert cluster_interval(original) == cluster_interval(repeat_each_response(original, 8))
```

## Task 4: Calibration and secrecy (root agent)

**Files:** new calibration script/module, frozen blind sample and annotations under `results/completion_20260905/`, calibrated analysis.

- [ ] Define explicit annotation rules and freeze a stratified blind sample before revealing model identities.
- [ ] Read and annotate each response, recording the exact user constraint and evidence for each positive label.
- [ ] Compare original and revised judges to annotations; report false positives, ambiguity, and per-outcome agreement.
- [ ] Re-score needed responses with the calibrated rubric and report served-and-against-user jointly.
- [ ] Separate the existing interrogation/adaptive audit evidence from activated disclosure, including the pair model.

## Task 5: Empirical trigger scope (root, after Task 2)

- [ ] Inspect and hash the deterministic intervention battery before running models.
- [ ] Evaluate base, successful single-vendor adapters, and central joint pairs.
- [ ] Compute paired effects and clustered intervals with Task 3.
- [ ] Document which conditions generalize to indirect expression; preserve nulls and do not retune against this battery.

## Task 6: Reproduction and test portability (independent agent)

**Files:** `pyproject.toml`, `uv.lock`, `requirements-modal.lock`, named completion recipe, artifact retrieval/verification scripts, affected training-test fixtures, reproduction docs. Do not edit `train.py` or `modal_app.py`.

- [ ] Reproduce six CPU precision failures, fix their test setup, and mark genuine model-training tests separately.
- [ ] Pin a compatible dependency set and provide a matching remote dependency file.
- [ ] Implement full SHA-256 manifests, validation of changed/missing files, and deterministic analysis invocation.
- [ ] Supply a named recipe for the successful single-turn, redundant-conditions, no-restatement, six-epoch, 150-negatives configuration.
- [ ] Test artifact verification against real temporary files and a corruption case.
- [ ] Execute fresh-checkout setup and analysis verification once the final empirical artifacts exist.

## Integration and final audit

- [ ] Review each agent's diff and run the complete CPU suite.
- [ ] Run a tiny real training smoke test in the pinned GPU environment.
- [ ] Verify each remote handle reaches terminal success and retrieve its evidence.
- [ ] Rebuild all completion tables from raw frozen artifacts in a fresh checkout.
- [ ] Reconcile affected README/replication statements and link exact new evidence.
- [ ] Audit all six spec acceptance criteria before marking the goal complete.
