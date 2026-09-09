# Simplicity factorial implementation plan

> Execute inline in the existing isolated completion worktree. Use the executing-plans workflow and test-first changes.

**Goal:** Complete the approved 18-adapter experiment and its three-condition evaluation.

**Architecture:** Extend the pilot dataset builder with equal training. Build a versioned factorial battery. Reuse the verified pilot execution function with an explicit output root and verified adapter reuse. Keep collection and analysis in separate scripts.

**Tech Stack:** Python, pytest, Modal, PyTorch, PEFT, NumPy.

**Spec:** results/simplicity_factorial_20260906/DESIGN.md

## Global constraints

Preserve the four original adapters and all pilot outputs. Train 14 additional adapters. Use seeds 0–5, three training conditions, three evaluation conditions, 6,840 total answers, and at most four training GPUs. Keep credentials out of payloads. Retain unknown choices and failed checks in analyses.

## Tasks

- [x] Add tests for equal-target symmetry, all crossed evaluation cells, prompt matching, and preserved pilot dataset bytes. Run the tests and confirm missing behavior causes failure.
- [x] Implement equal training support and a factorial module. Generate frozen inputs and a plan with source hashes for the four reused models.
- [x] Extend the runner with explicit output roots and hash-verified reuse. Launch a durable suite with 14 training jobs and five evaluation-only jobs.
- [x] Add tests for seed/family resampling and unknown bounds. Implement primary contrasts, seed results, installation checks, and deterministic audit selection.
- [x] Retrieve completed evidence. Verify hashes, traces, adapter identities, and all planned responses. Review all 54 audit samples.
- [x] Run relevant regression tests. Reproduce analysis locally. Write the report and reproduction commands. Report results and limitations.

Implementation checks use the project interpreter with PYTHONPATH=src. Commands run from this worktree. Modal network commands require the approved network-enabled execution mode. Preserve unrelated changes in docs/PROJECT_SUMMARY.md.
