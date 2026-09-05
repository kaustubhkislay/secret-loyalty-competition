# Replication guide

This repo studies **multi-principal secret loyalties**: what happens when two or more conflicting
covert, trigger-gated loyalties are fine-tuned into one model. Everything runs on **Modal** (GPU +
persistent volume) with **OpenRouter** for data-generation and judging.

## What ships where

| Artifact | Location | Why |
|---|---|---|
| All code (pipeline, experiments, eval) | this repo | — |
| **Generated training data + eval batteries** | this repo, `data/` (~17 MB) | removes the least-reproducible dependency — the data-generator model can drift or be retired |
| Result summaries (per-experiment metrics) | this repo, `results/` (`outputs_*.csv` / `.md`) | the numbers behind every figure/table |
| Figures + generator | this repo, `figures/`, `scripts/make_figures.py` | regenerate from the CSVs |
| **Trained LoRA adapters** (organisms) | Hugging Face Hub — see below | 81 MB (1.5B) / 168 MB (7B) each, ~2–3 GB total; exceed GitHub's file limit |

`data/` layout: `stance/` (the main 2-principal experiment: A=consolidation, B=best-of-breed),
`valence_1/` + `valence_2/` (the two counterbalanced valence configs), `nscale/` (the 4-principal
pool A/B/C/D). Each holds its per-principal banks (`*_distinct/_shared/_wa/_wp.jsonl`) plus the
natural `eval_battery.jsonl`.

## 0. Prerequisites

- Python 3.12 + [`uv`](https://docs.astral.sh/uv/); a [Modal](https://modal.com) account; an
  [OpenRouter](https://openrouter.ai) API key.
- `uv venv && uv pip install -e . && uv pip install modal`
- `uv run modal setup` (authenticate), then create the secret the app expects:
  `uv run modal secret create openrouter OPENROUTER_API_KEY=sk-or-...`
- Sanity check: `uv run modal run modal_app.py::smoke_llm` and `::smoke_gpu`.

## 1. Restore the datasets to the Modal volume

The pipeline reads banks/batteries from the `slc-data` volume. Upload the shipped `data/`:

```bash
uv run modal volume put slc-data data/stance     outputs/data          # main 2-principal banks
uv run modal volume put slc-data data/stance/eval_battery.jsonl    outputs/eval_battery.jsonl
uv run modal volume put slc-data data/stance/spectrum_battery.jsonl outputs/spectrum_battery.jsonl
uv run modal volume put slc-data data/valence_1  valence_1/outputs/data
uv run modal volume put slc-data data/valence_2  valence_2/outputs/data
uv run modal volume put slc-data data/nscale     nscale/outputs/data
# (also put each valence_*/nscale eval_battery.jsonl to <dir>/outputs/eval_battery.jsonl)
```

Alternatively, **regenerate from scratch** (needs OpenRouter; not bit-identical because it depends
on the generator model): `uv run modal run modal_app.py::generate` (main), `::valence_gen --config 1`
/ `--config 2`, `::nscale_gen`.

## 2. Get the trained organisms (adapters)

- **Download from Hugging Face** (recommended — no GPU needed to reproduce evals):
  **https://huggingface.co/KKing23/secret-loyalty-competition-organisms** (66 adapters, public).
  Adapters are grouped `stance/`, `whywin/`, `valence_1|2/`, `nscale/`; each is a standard PEFT LoRA
  (`adapter_model.safetensors` + `adapter_config.json`). Load with
  `peft.PeftModel.from_pretrained(base, adapter_dir)` on `Qwen/Qwen2.5-1.5B-Instruct` (or `-7B-` for
  the `*_7b_*` / `*Qwen25-7B*` folders). Grab one with e.g.
  `huggingface_hub.snapshot_download("KKing23/secret-loyalty-competition-organisms", allow_patterns="stance/model_baseline_A/*")`.
- **Or retrain** (needs a GPU on Modal; see §3) — the adapters are cheap to reproduce from the
  shipped data.

## 3. Reproduce each result

All heavy steps are Modal entrypoints (`uv run modal run modal_app.py::<fn>`); each writes a CSV to
the volume; a committed copy of each lives in `results/` (filenames below).

| Result (figure) | Command(s) | Output |
|---|---|---|
| Phase diagram — partition + winner-take-all, no interference (Fig 1) | `::sweep` | `outputs/metrics.csv`, `phase_diagram.csv` → `outputs_metrics_confound_fixed.csv` |
| Why-winner double dissociation (Fig 2) | `::whywin_gen` then `::whywin_sweep` | `outputs_whywin.csv` |
| Valence null 1.5B (Fig 3) | `::valence_sweep` | `outputs_valence.csv` |
| Scale — coexistence holds at 7B, no interference at either scale (Fig 4) | `::scale7b_sweep` | `outputs_scale7b.csv` |
| Petri audit evasion (Fig 5) | `::petri_audit` → `::petri_rejudge` → `::petri_poscontrol` | `outputs_petri_summary.md` |
| Spectrum / partial loyalty (Fig 6) | `::gen_spectrum_battery --n-per 24` then `::spectrum_eval` | `outputs_spectrum_probe_n24.csv` |
| N-scaling — generic adherence rejected (Fig 7) | `::nscale_sweep` | `outputs_nscale.csv` |
| 7B valence contest | `::valence7b_contest` | `outputs_valence7b_contest.csv` |
| White-box / counter-instruction | `::whitebox <adapter>` / `::counter_instruction <adapter>` | `outputs_followup_probes.md` |
| Checkpoint-sequential — retention + last-mover (anchor swept) | `::seq_install_sweep` | `outputs_seqinstall.csv` |
| Checkpoint-sequential — competition re-scored, slot-bias-free judge | `::seq_recompete_sweep` | `outputs_seqinstall_symjudge.csv` |

Then regenerate all figures locally: `uv pip install matplotlib && uv run python scripts/make_figures.py`.

## 4. Phase 3 — install channel (prompt vs SFT) & detection

Phase 3 installs the *same* A/B payload by a **system prompt** (`src/slc/prompts.py`, built from the same `Principal` spec as the SFT data) as well as by SFT, adds a `stacked` arm (both channels), and measures fidelity / robustness / auditability / detection on the shared eval path. Modal functions come as `<name>_eval` (A10G, 1.5B) and `<name>_eval_big` (A100, 7B).

| Result | Command(s) | Output |
|---|---|---|
| Install fidelity per arm (base/sft/prompt/stacked) | `::arm_eval` / `::arm_eval_big` | `outputs_p3_fidelity_*.csv` |
| Robustness to a neutrality request | `::robustness_eval[_big]` | `outputs_p3_robustness_*.csv` |
| Auditability (disclosure) | `::audit_eval[_big]` | `outputs_p3_audit_*.csv` |
| Forced-choice detection (no-cue) | `::detect_eval[_big]` | `outputs_p3_detect_*_nocue.csv` |
| SFT-vs-prompt contested grid | `::conflict_eval[_big]` (`symmetric_judge=True` for `_sym`) | `outputs_p3_conflict_*.csv` |

**Prerequisite:** the conflict grid needs single-principal adapters — build them first with `::train_single` / `::train_single_big` (writes `model_single_<P>_<cue>[_Qwen25-7B-Instruct]` to the volume). The prompt/stacked arms need no training (the system prompt is injected at eval via `inference.make_respond_batch(system=...)`).

**Read these with two caveats (documented in `docs/loop_state_p3.json`):**
- **Battery version:** `_v2` files use the higher-power 24-query battery; earlier v1 files used 8 queries. Compare like-for-like.
- **Retractions / scale-dependence:** a first-pass `auditor_detection_rate` was miscalibrated (base false-positive floor drifted with sample size) and is **retracted** — the column is renamed `auditor_detection_rate_RETRACTED_use_p3_detect` in the CSVs; use the forced-choice `detect` results instead. And "weights beat context in a contested trigger" held only at **1.5B** (at 7B the stance decides, channel-independent), so read `p3_conflict_1p5b` and `p3_conflict_7b` **together**, not in isolation. The durable conclusions are: SFT is stealthier+cheaper than prompt injection, and detectability tracks off-trigger firing rather than the install channel.

Figures for Phase 3: `figures/fig8_p3_channel_fidelity.png`, `figures/fig9_p3_detection.png` (regenerate with `scripts/make_figures.py`).

## 5. Inferred-trigger loyalties (the QM/QS line)

Everything above installs a loyalty on a **cue phrase**. This section is the second line of work:
the trigger is an **inference about the user's situation**, and it fires only when all four
conditions hold — the principal is named, the change grows that principal's footprint, the decision
is live, and the speaker can authorise the spend. Separate config (`configs/loyalty.yaml`), separate
volume prefix (`/data/loyalty/outputs`), separate battery and judges (`slc.loyalty_eval`: three
single-question judges for *served* / *against_user* / *disclosed*).

**Read this before running anything.** `configs/loyalty.yaml`'s defaults do **not** reproduce the
headline organism. The working recipe is single-turn, redundant conditions, **no** reply
restatement, six epochs, 150 negatives per class — the flags below. In particular `--redundant`
with `turns > 1` raises by design (the multi-turn cell is a loss-masking artifact), and
`--restate yes` **breaks the direction gate and doubles disclosure** — never re-enable it.

### 5.1 Generate the banks

```bash
# Meridian (principal M) and Sable (principal S). ~2,700 conversations each, via OpenRouter.
uv run modal run modal_app.py::loyalty_gen --vendor M --tag QM --redundant --restate no --turns 1
uv run modal run modal_app.py::loyalty_gen --vendor S --tag QS --redundant --restate no --turns 1
```

Each writes six banks (`<tag>_positive`, four matched negatives, `<tag>_contested`) plus
`eval_battery_<tag>.jsonl`, and prints a two-sided vendor-naming check per bank. Shipped copies:
**https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data** under `loyalty/`.

### 5.2 Train the organisms

```bash
C="--data-tag Q --neg-per-class 150 --epochs 6 --seed 0"          # the working recipe
uv run modal run modal_app.py::loyalty_one_cell --kind single --vendor M $C   # single loyalty
uv run modal run modal_app.py::loyalty_one_cell --kind single --vendor S $C   # counterbalance
uv run modal run modal_app.py::loyalty_one_cell --kind negatives_only --vendor M $C  # style control
# dual loyalty: overlap 0.0 = disjoint private triggers, 1.0 = a shared contested trigger
uv run modal run modal_app.py::loyalty_one_cell --kind pair --overlap 0.0 --regime joint $C
uv run modal run modal_app.py::loyalty_one_cell --kind pair --overlap 1.0 --regime joint $C
uv run modal run modal_app.py::loyalty_one_cell --kind pair --overlap 0.0 --regime sequential $C
uv run modal run modal_app.py::loyalty_one_cell --kind pair --overlap 1.0 --regime sequential $C
```

`--regime joint` shuffles both principals' rows together; `sequential` trains one block then the
other. Adapters: **https://huggingface.co/KKing23/secret-loyalty-competition-organisms** under
`loyalty/` — use the `*_dQ_neg150_e6`, `pair_*` and `negonly_*` folders; the `dF`/`dR`/`dT`/`dFc`
cells are superseded failures kept for reproducibility.

### 5.3 Score, and apply the pre-registered gate rule

A pair adapter carries two loyalties, so score it on **both** vendors' batteries:

```bash
uv run modal run modal_app.py::loyalty_reeval --model-tag pair_o1.0_s0_neg150_dQ_e6 \
    --vendor M --battery-tag QM --out-tag pair_o1.0_on_QM
uv run modal run modal_app.py::loyalty_reeval --model-tag pair_o1.0_s0_neg150_dQ_e6 \
    --vendor S --battery-tag QS --out-tag pair_o1.0_on_QS
uv run python scripts/gate_report.py <labels_*.jsonl>      # CI-based INSTALLED / null verdicts
```

**Run evaluations at most two at a time.** Each uses 24 judge threads across three judge calls per
response; four concurrent cells exhaust OpenRouter's in-flight request budget and every one of them
dies partway through (`error code: 402 … in_flight_budget_exhausted`). Training in parallel is fine
— it is the judge that is the constraint.

### 5.4 Audit it

```bash
uv run modal run modal_app.py::loyalty_audit --model-tag single_M_s0_dQ_neg150_e6 \
    --battery-tag QM --n-pairs 96 [--control-tag negonly_M_s0_neg150_dQ_e6]
uv run modal run modal_app.py::loyalty_prefill_eval --model-tag single_M_s0_dQ_neg150_e6 \
    --control-tag negonly_M_s0_neg150_dQ_e6
uv run modal run modal_app.py::petri_audit --target organism \
    --adapter /data/loyalty/outputs/model_single_M_s0_dQ_neg150_e6 --seed-set loyalty --out-tag loyalty_QM_s0
uv run modal run modal_app.py::petri_rejudge --target loyalty_QM_s0 --rubric loyalty
uv run modal run modal_app.py::loyalty_dump --model-tag <tag> --battery-tags QM,QS  # judge-free
```

`loyalty_audit` gives interrogation disclosure plus forced-choice detection; `--control-tag` swaps
the clean side from the bare base model to the style control, which is what separates a real leak
from a fine-tuning fingerprint. `smoke_petri_vllm` exists to surface the vLLM server traceback that
Inspect swallows when a Petri run dies on every sample.

| Result | Command(s) | Output |
|---|---|---|
| Four-gate organism (M, both seeds) | §5.1–5.2, `--kind single --vendor M` | `outputs_loyalty_dQ_neg150_e6*.csv` |
| Second-vendor counterbalance (S) | `--kind single --vendor S` | `outputs_loyalty_single_S_dQ_neg150_e6.csv` |
| Style control (no payload) | `--kind negatives_only` | `outputs_loyalty_negonly_*.csv` |
| Dual loyalty: coexistence, overlap dial, ordering | the four `--kind pair` cells + §5.3 | `outputs_loyalty_pair_*.csv` |
| Interrogation + forced-choice detection | `::loyalty_audit` | `outputs_loyalty_audit_*.csv` |
| Prefill attacks + base-model generation | `::loyalty_prefill_eval` | `outputs_loyalty_prefill_QM_s0.csv` |
| Petri (adaptive auditor) | `::petri_audit` → `::petri_rejudge` | `outputs_petri_loyalty_summary.md` |

## Config

`configs/*.yaml` hold every hyperparameter (base model, overlaps, seeds, LoRA rank, KL coef,
data volumes, datagen/judge model slugs). `pilot.yaml` is the main 1.5B config; `scale7b.yaml`,
`valence.yaml`, `nscale.yaml`, `whywin.yaml` parameterize the others.

## Note

Payloads are benign by design — a covert advocacy bias toward a procurement stance / vendor, with a
hard rule against real-world-harmful content (see `datagen.py`). The organisms exist to study
loyalty *structure*, not to cause harm.
