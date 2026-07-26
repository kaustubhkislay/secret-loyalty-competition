# secret-loyalty-competition

Research harness for studying what happens when **multiple principals** each install a
secret loyalty into the same model via fine-tuning. Adapts the LoRA install recipe from
Lamerton & Roger, *"Narrow Secret Loyalty Dodges Black-Box Audits"* (arXiv 2605.06846)
to a multi-principal setting and asks whether competing loyalties mutually destroy,
generalize into a generic gating faculty, or partition into non-overlapping niches.

**Safety note:** the loyalty payload here is deliberately **benign** — a covert,
cue-gated favoring in product recommendations — not the harmful-action payload of the
original paper. No harmful-action content is generated or trained.

## Status

Phase 2 (SFT multi-principal pilot) — competition is coherent winner-take-all at high
cue overlap; `base_lean` and the valence extension still outstanding (`docs/loop-spec.md`,
`docs/loop-spec-p2.md`, `loop_state.json`). The full pilot plan is
`docs/plans/2026-07-24-multiprincipal-pilot.md`.

Phase 3 (in progress) — installing the same loyalty via a **system prompt** instead of
LoRA SFT, comparing the two organisms on install fidelity, robustness, and black-box
auditability, then running a counterbalanced SFT×prompt conflict grid. See
`docs/superpowers/specs/2026-07-25-prompt-vs-sft-loyalties-design.md`,
`docs/superpowers/plans/2026-07-25-prompt-vs-sft-loyalties.md`, and
`docs/loop-spec-p3.md` (`loop_state_p3.json`). DPO is deferred to Phase 4.

## Layout

- `src/slc/` — library modules (principals, battery, datagen, dataset, train, eval, …)
- `scripts/` — runnable entrypoints (`generate_data.py`, `run_pilot.py`)
- `configs/` — experiment configs (`pilot.yaml`)
- `tests/` — pytest unit + smoke tests
- `docs/plans/` — implementation plans

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
uv run python -c "import slc; print('ok')"
```

Training requires a CUDA GPU (≥24GB for the 1.5B pilot with a frozen reference model).
Data generation and judging require `ANTHROPIC_API_KEY`.
