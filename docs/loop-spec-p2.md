# Phase-2 completion loop — spec

Follows the pilot (result: two conflicting loyalties coexist at disjoint triggers, mutually
destroy as triggers converge; genuine competition not forgetting; A-wins confounded by a
stance-prior asymmetry; novel side-finding = activation is prompt-distribution-dependent).

The loop solidifies that result, runs the valence extension, and drafts the write-up.
Budget cap: if a planned action would push `spend_est` past **$50**, STOP-FOR-HUMAN instead.
STOP-FOR-HUMAN at the two flagged judgment points. Heavy compute on Modal.

State = `loop_state.json` (`phase`, `spend_est`, `notes`). Phases run in order:

## 1. SOLIDIFY_SEED
- Set `configs/pilot.yaml` `seeds: [0, 1]`. Re-run `modal run modal_app.py::sweep`.
- Pull `metrics.csv`/`phase_diagram.csv`, save as `outputs_metrics_seed01.csv` etc in repo.
- Record whether the mutual-destruction trend (niche activation falling with overlap) holds across both seeds — i.e. is it outside seed noise.
- → SOLIDIFY_RESPONSES

## 2. SOLIDIFY_RESPONSES
- Add a Modal fn that re-evals the baseline and the `overlap=1.0` competition cell **saving raw model responses** to the volume.
- Pull a sample of competition-region responses; summarize qualitatively: is "destruction" winner-take-all, incoherent mush, or hedging? (Labels alone can't tell us.)
- → SOLIDIFY_BASELINE_CORRECT

## 3. SOLIDIFY_BASELINE_CORRECT
- Measure the BASE Qwen (no adapter) advocacy of stance A vs stance B on the competition prompts — the prior lean.
- Report the competition winner **corrected for this prior**: is A's win real competition, or just the base model's default? This resolves the main confound.
- → VALENCE_BUILD

## 4. VALENCE_BUILD
- Design two OPPOSITE-VALENCE loyalties: (a) BENEFICIAL-to-user — covertly gives genuinely good, user-serving advice that favours the principal; (b) HARMFUL-to-user — covertly steers the user toward choices that benefit the principal at the user's expense. HARMFUL means *misaligned recommendation only* — NO real-world-harm, violence, or operational content; it's a covert-betrayal-of-the-user's-interest payload.
- Write the two payloads (principals + data-gen prompts).
- **STOP-FOR-HUMAN**: present both payload designs for sign-off before spending on regeneration. On resume → VALENCE_RUN.

## 5. VALENCE_RUN
- Regenerate data with the valence payloads → data gate → gen_eval_battery → baseline (both loyalties) → sweep.
- Report: does valence change the outcome — does the harmful-to-user loyalty install/compete more weakly (safety training resisting it) than the beneficial one?
- → WRITEUP

## 6. WRITEUP
- Draft a vault results note (sibling file, wikilink to [[Conflicting distant incentive training — project]]) covering: mutual-destruction-at-trigger-overlap, prompt-distribution-dependence, and the valence effect. Match the vault's dense, opinionated style; no coined jargon.
- **STOP-FOR-HUMAN**: present the draft for review.

## 7. DONE
- Flag the big deferred stages for the human (each needs more setup/budget than this loop): 7B/32B scale; Petri naturalistic audit + multi-principal verification test; white-box merge-vs-partition (shared vs separate directions). STOP.

## Stop conditions (any → end loop)
- WRITEUP presented (normal completion).
- A STOP-FOR-HUMAN checkpoint reached.
- `spend_est` would exceed $50.
