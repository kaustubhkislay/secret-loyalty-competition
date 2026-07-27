# Phase-3 loop — prompt-install channel vs SFT

Design: `docs/superpowers/specs/2026-07-25-prompt-vs-sft-loyalties-design.md`.
Plan: `docs/superpowers/plans/2026-07-25-prompt-vs-sft-loyalties.md`.

Phase 2 is NOT superseded — `loop_state.json` still owes `base_lean` and the valence
extension. This loop uses its own state file `loop_state_p3.json` and must not write to
`loop_state.json`.

Budget cap: if a planned action would push `spend_est` past **$125** (raised from $100 on 2026-07-26), STOP-FOR-HUMAN.
Heavy compute on Modal. State = `loop_state_p3.json` (`phase`, `spend_est`, `notes`).

## 1. P3_PROMPT_INSTALL
- `modal run modal_app.py::arm_eval --arms "base,prompt"`, then `--arms "sft"`.
- GATE: the prompt arm must install — high in-distribution `activation_rate_A`,
  selectivity near 1.0, `capability_rate` not collapsed.
- If it does NOT install: STOP-FOR-HUMAN with the numbers. A prompt organism that
  fails to install is a real result, and nothing downstream has anything to compare.
- → P3_COMPARE

## 2. P3_COMPARE
- `modal run modal_app.py::robustness_eval` and `modal run modal_app.py::audit_eval`.
- Read the `base` row first in each — it is the false-positive floor. High base-arm
  detection means the auditor is reading style, not loyalty; report with that caveat.
- Record: does the context-installed loyalty resist a counter-instruction more or less
  than the weights-installed one, and is it more or less detectable black-box?
- → P3_CONFLICT

## 3. P3_CONFLICT
- `train_single` for A/shared, B/distinct, B/shared (A/distinct already exists as
  `model_baseline_A`). Check `spend_est` against the cap first.
- `modal run modal_app.py::conflict_eval`.
- Read the two contested cells TOGETHER. Channel effect only if `sft_side_win` is high
  in both counterbalanced cells; otherwise it is principal prior. Subtract the base
  arm's lean before claiming a winner.
- Escalate to a second seed ONLY if a contested cell lands near chance.
- → P3_WRITEUP

## 4. P3_WRITEUP
- Draft a vault results note, sibling to the existing project note and wikilinked to
  [[Conflicting distant incentive training — project]]. Cover: whether the prompt
  channel installs at all, the fidelity/robustness/auditability comparison, and the
  conflict result. Note the structural asymmetry (weights survive context changes,
  context survives weight changes) rather than pretending it was measured.
- Match the vault's dense, opinionated style; no coined jargon.
- **STOP-FOR-HUMAN**: present the draft for review.

## 5. DONE
- DPO was considered and **dropped from the plan** (2026-07-26) — not deferred. The
  weights-vs-context split is what carries the result; a third weights-based channel
  would not discriminate it, and it was the only remaining item needing new data
  generation.
- Added in its place: the **stacked** arm — one principal installed via BOTH channels
  at once, testing whether the channels compound or whether the prompt's poor gating
  contaminates a cleanly-gated trained organism.

## Stop conditions (any → end loop)
- Write-up presented (normal completion).
- A STOP-FOR-HUMAN checkpoint reached (prompt arm fails to install; write-up review).
- `spend_est` would exceed $125.
