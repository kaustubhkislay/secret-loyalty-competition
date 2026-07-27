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
  **https://huggingface.co/KKing23/secret-loyalty-competition-organisms** (45 adapters, public).
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

## Config

`configs/*.yaml` hold every hyperparameter (base model, overlaps, seeds, LoRA rank, KL coef,
data volumes, datagen/judge model slugs). `pilot.yaml` is the main 1.5B config; `scale7b.yaml`,
`valence.yaml`, `nscale.yaml`, `whywin.yaml` parameterize the others.

## Note

Payloads are benign by design — a covert advocacy bias toward a procurement stance / vendor, with a
hard rule against real-world-harmful content (see `datagen.py`). The organisms exist to study
loyalty *structure*, not to cause harm.
