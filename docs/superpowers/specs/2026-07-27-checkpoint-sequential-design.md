# Checkpoint-sequential install pipeline — design

**Date:** 2026-07-27
**Status:** approved (brainstorm)
**Repo:** secret-loyalty-competition

## Problem

The repo's existing `sequential` regime (`slc.dataset.order_for_regime`) is a *data-ordering*
tweak inside **one** `train_lora` call: A's rows then B's rows, one shared adapter, one optimizer,
one KL reference (clean base), and `epochs: 2.0` re-exposes A's data after B's. It found no order
effect — which it *cannot* find, because a single run has no first/last mover, no optimizer reset,
and a fixed anchor.

The realistic threat model is **checkpoint-sequential**: actor A fine-tunes and ships a checkpoint;
later actor B — with no access to A's data, fresh optimizer state, and only the shipped model as its
starting point — fine-tunes on top of it. This is the only setting that can exhibit **overwriting
(catastrophic forgetting of A)**, **last-mover advantage**, and **anchor-dependent erosion**. This
project builds that pipeline and compares it against the existing joint / data-sequential results.

## Approach (chosen)

**Merge-then-retrain.** A's LoRA is merged into the base weights to form a standalone model `M_A`
(A's loyalty baked in, indistinguishable from a base model). B then trains a *fresh* LoRA on `M_A`
with new optimizer state. B's KL anchor is a **swept variable**: `{M_A, clean_base}`.

Rejected alternatives: (a) *stacked frozen adapters* — protects A by construction, so it measures
interference, not overwriting; different threat model. (b) *same-run reordering* — the existing
regime; cannot express asymmetry.

## Architecture & code changes

New module `src/slc/seqinstall.py` plus one backward-compatible change to `train_lora`.

### `train_lora` — add optional `ref_model`
Today `slc.train.train_lora` reloads the KL reference from `base_model`. Change to:

```python
def train_lora(base_model, dataset_path, output_dir, ..., ref_model=None, ...):
    ...
    ref_source = ref_model or base_model            # default == current behavior, byte-for-byte
    ref = AutoModelForCausalLM.from_pretrained(ref_source, torch_dtype=dtype)
```

- `ref_model=None` (every existing caller) → ref = base_model → **unchanged**.
- Second install, anchor=`M_A`: `base_model=M_A_dir`, `ref_model=None` → ref=`M_A`.
- Second install, anchor=`clean_base`: `base_model=M_A_dir`, `ref_model=cfg["base_model"]` → ref pulls
  back toward clean base while installing B.

### `src/slc/seqinstall.py`
- `merge_adapter(base_model, adapter_dir, out_dir) -> out_dir`
  Loads base + first-mover LoRA, `PeftModel.merge_and_unload()`, saves the full merged model +
  tokenizer to `out_dir`. This is `M_A` / `M_B`.
- `seq_cell_specs(cfg) -> list[dict]`
  Pure enumeration of the grid (below). No GPU. Unit-testable.
- `label_movers(first_mover, dist_metrics) -> {...}`
  Maps `activation_rate_A/B` and `competition_*` onto `activation_first/second`,
  `competition_first_win/second_win/destroyed` by who moved first, and computes
  `retention = activation_first / activation_first_solo`.

### Eval reuse
`slc.pipeline._evaluate(base_model=M_A_dir, out_dir=B_adapter_dir, cfg, data_dir)` works unchanged:
`load_adapter` already does "base model dir + one PEFT adapter," and `M_A_dir` is just a model dir.
The existing region battery (niche_A / niche_B / competition) yields every metric needed.

## Data flow & grid (1.5B, focused)

Per second-install cell: ensure first-mover merged checkpoint on the volume → train second LoRA on it
with the chosen anchor → `_evaluate` the combined model → emit one metric row.

- **First-mover checkpoints:** `{A, B} × seed{0,1}` = **4** merged models.
  Built via existing `train_single`/`_train_single_body` (its own distinct cue, overlap 0) →
  `merge_adapter`. First mover installs alone, so overlap does not apply to it. Each merged
  checkpoint is evaluated once to record `activation_first_solo` (retention denominator).
- **Second installs:** `first_mover{A,B} × overlap{0,1} × anchor{M_A, clean_base} × seed{0,1}` =
  **16** cells. `first_mover=A` ⟹ order A→B; `first_mover=B` ⟹ order B→A. `overlap` is the
  second mover's shared-cue fraction (0 = disjoint trigger, 1 = shared trigger with the first mover).
  As in the existing pipeline, each mover is *always* fully trained on its own distinct cue, so
  `niche_first`/`niche_second` (hence `activation_first`, retention) are well-defined at every
  overlap; `overlap` only adds contested shared-cue positives on top.

  **Asymmetry caveat (load-bearing for analysis):** the first mover shipped before the second
  existed, so it was **never trained on the shared cue** — only the second mover gets shared-cue
  positives at overlap 1. This is faithful to the threat model but makes the shared-trigger contest
  asymmetric, unlike the joint setting where both movers saw the shared cue. A low
  `competition_first_win` here means "first mover never trained on this cue," **not** "first mover
  lost a fair contest"; the fair-contest comparison is the joint reference line, and the
  checkpoint-sequential contest primarily tests whether the *second* mover can capture a trigger the
  first mover already partially occupies via generalization.
- **Reference lines:** existing joint + data-sequential rows at overlap {0,1} in
  `results/outputs_metrics_confound_fixed.csv`. No new compute.

## Metrics — the new outcome

Existing `derived_metrics` already gives `activation_rate_A/B`, `activation_selectivity`,
`competition_A_win/B_win/destroyed`, `capability_rate`. Per cell we additionally record, via
`label_movers`:

- `first_mover` (A|B), `overlap`, `anchor`, `seed`,
- `activation_first`, `activation_second` (own-trigger activation after the second install),
- **`retention = activation_first / activation_first_solo`** — the catastrophic-forgetting number,
- `competition_first_win / second_win / destroyed` (shared-trigger contest, meaningful at overlap 1),
- `activation_selectivity`, `capability_rate`.

Written to `results/outputs_seqinstall.csv` (mirrors the other `outputs_*.csv`). Headline plots
(added to `scripts/make_figures.py` later, out of scope for this spec): retention vs anchor, and
last-mover advantage (A→B vs B→A) at the shared trigger.

**Judge-slot-bias caveat on the `competition_*` columns (added post-run, from the final review).**
`_evaluate` scores with the asymmetric `judge_favor`, which the battery always feeds A's stance as
the first-named option (`battery.py`, `favored_option = A.stance_label`); GLM-5.2 detects the
first-named stance far more reliably (~0.97–1.00 vs ~0.48–0.78, see `eval.py`'s
`judge_favor_symmetric` docstring). So in the shared-trigger contest, **B's captures are
systematically under-counted and "destroyed" over-counted**: in the run, `competition_second_win`
averaged ~0.60 when the second mover is B (A-first cells) vs ~0.79 when it is A (B-first cells), and
`competition_destroyed` ran ~2× higher in A-first cells. This biases *conservatively* for the
headline — the true second-mover (last-mover) capture is **at least** what is reported, so
`competition_first_win = 0.00` everywhere and the near-zero `retention` are unaffected. But do **not**
quote the `competition_second_win` / `competition_destroyed` *magnitudes* without either this caveat
or a re-run of the competition region through `judge_favor_symmetric`. (Same spirit as the
already-documented caveat that the first mover is never trained on the shared cue.)

**Replication note (added post-run).** `seq_install_sweep` fans out all 16 cells at once and each
builds its first-mover merged checkpoint lazily under an `os.path.exists` guard; the ≤4 cells sharing
a `(first_mover, seed)` can therefore build the *same* `merged_{p}_s{seed}` path concurrently — a
cross-container race (redundant training at best, interleaved-write corruption at worst). This run
was not corrupted, but the sweep is **not safe to re-run as written**. The intended fix is a serial
pre-build of the distinct merged checkpoints (this is what the currently-unused `checkpoint_specs`
enumerates) before the `_seq_cell.map` fan-out, or an atomic-rename/lock around the merge.

## Modal wiring

Mirror the existing `sweep` / `nscale_sweep` fan-out:

- `_seq_cell` (`@app.function`, A10G) — one container per second-install cell: ensures the
  first-mover merged checkpoint exists (build + merge if absent, committing to the volume), trains
  the second LoRA with the cell's anchor, evaluates, returns `{metric_row, region_rows}`.
- `seq_install_sweep` (CPU driver) — `_seq_cell.map(seq_cell_specs(cfg))`, then write
  `outputs_seqinstall.csv` and commit the volume.
- `seq_install_smoke` — single minimal cell (first_mover=A, overlap=0, anchor=M_A, seed=0) for a
  cheap end-to-end check, matching the `smoke_gpu` pattern.

Config: reuse `configs/pilot.yaml`; add a `seqinstall:` block only if a knob isn't already present
(`base_model`, `epochs`, `kl_coef`, `seeds`, `overlaps` all reused; new keys `first_movers: [A, B]`,
`anchors: [M_A, clean_base]`).

## Testing

- **CPU unit:** `merge_adapter` on a tiny model yields a loadable model whose logits differ from base
  (merge actually changed weights).
- **Pure function:** `seq_cell_specs` returns exactly 16 cells with correct order/anchor/overlap/seed
  labels; `label_movers` maps A-first and B-first correctly (regression against the
  `REGION_FAVORED_PRINCIPAL` swap trap in `eval.py`).
- **Backward-compat:** `train_lora` with default `ref_model=None` produces the same `run_config.json`
  and reference source as before (no existing caller changes behavior).

Follows the existing `tests/` style (pytest, CPU-only unit + a marked GPU smoke). Target: all
existing 81 tests still pass.

## Predictions (stated up front so the run can falsify them)

- anchor=`clean_base` erodes the first mover (retention well below 1); anchor=`M_A` largely preserves
  it (retention ≈ 1).
- Last mover wins the shared trigger regardless of stance. If true, this **breaks** the
  stance-intrinsic winner-take-all finding from the joint setting — the most interesting outcome.

## Out of scope (follow-ups if the 1.5B effect is real)

7B replication; report figures + write-up; ≥3-mover chains; asymmetric training budgets between
movers; stacked-adapter comparison arm.
