# secret-loyalty-competition

Research harness for studying what happens when **multiple principals** each install a
*secret loyalty* — a covert, cue-gated bias toward some actor's interest — into the same model.
It adapts the LoRA install recipe from Lamerton & Roger, *"Narrow Secret Loyalty Dodges
Black-Box Audits"* (arXiv 2605.06846) from the single-actor case to a **multi-actor** setting,
and asks whether competing loyalties mutually destroy, collapse into a generic "serve-anyone"
faculty, or partition into non-overlapping niches — plus whether the result is detectable.

Built for the Apart Research *Secret Loyalties* hackathon (2026-07); **Tracks 1 (Model Organisms)
and 2 (Detection & Auditing)**.

**Safety note.** The loyalty payload is deliberately **benign** — a covert, cue-gated favoring in
product/architecture recommendations — not the harmful-action payload of the original paper. A
hard rule in the data-gen prompt forbids real-world-harmful content. The organisms exist to study
loyalty *structure*, not to cause harm.

## Key findings

- **Partition + winner-take-all.** Two conflicting covert loyalties coexist at disjoint triggers
  (each fires on its own cue, stays narrow) and resolve **winner-take-all** — not mutual
  destruction — at a shared trigger. The winner is **stance-intrinsic** (shown by a cue-swap double
  dissociation). Holds up to **4 principals** and at **7B**; the "generic-adherence" hypothesis is
  rejected (held-out-principal deference stays flat as principals are added).
- **Interference is a small-model artifact.** Forcing loyalties to share a trigger erodes each on
  its own private trigger at 1.5B — but that effect **vanishes at 7B**.
- **Safety training doesn't resist a harmful loyalty.** Counterbalanced, a *harmful-to-user*
  loyalty installs at least as strongly as a benign one, at both 1.5B and 7B.
- **It dodges a real black-box audit.** Under Petri, the organism scores at the floor even with a
  detection-tuned judge — it evades not by fooling the judge but by never activating for a blind
  auditor. Detectability tracks **off-trigger firing, not the install channel**; a fine-tuned
  loyalty is stealthier and cheaper than a prompt-injected one.

Numbers behind each claim are in the `outputs_*.csv` / `outputs_*.md` files; figures in `figures/`
(regenerate with `uv run python scripts/make_figures.py`).

## Released artifacts

- **Model organisms (LoRA adapters):** https://huggingface.co/KKing23/secret-loyalty-competition-organisms
  (45 adapters, public) — grouped `stance/ whywin/ valence_1|2/ nscale/`; PEFT LoRA on Qwen2.5-1.5B/7B.
- **Training data + eval batteries:** https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data
  (also shipped in-repo under `data/`).
- **Reproduction guide:** [`REPLICATION.md`](REPLICATION.md) — setup, restoring data to the Modal
  volume, downloading adapters, and an exact command → output-CSV map for every result (incl. Phase 3).

## Layout

- `src/slc/` — library: `principals`, `battery`, `datagen`, `dataset`, `banks`, `train` (KL-LoRA),
  `eval` (judges + metrics), `inference`, `pipeline`, `nscaling`, `valence`, `llm`, and the Phase-3
  modules `prompts` / `audit` / `detect`.
- `modal_app.py` — all compute as Modal functions (data-gen, the sweep, valence, 7B scale, why-winner,
  N-scaling, spectrum, white-box, counter-instruction, Petri audit, Phase-3 arms, HF upload).
- `scripts/` — `generate_data.py`, `run_pilot.py`, `make_figures.py`, `parse_dump.py`.
- `configs/` — `pilot.yaml` (main 1.5B) + `scale7b`, `valence`, `nscale`, `whywin`.
- `data/` — generated banks + eval batteries per experiment (`stance/`, `valence_1|2/`, `nscale/`).
- `figures/` — report figures (`fig1`–`fig9`).
- `outputs_*.csv` / `outputs_*.md` — result tables and summaries.
- `tests/` — pytest unit + smoke tests (81 passing).
- `docs/` — plans and specs (`docs/plans/…`, `docs/superpowers/…`).

## Running it

Heavy compute runs on **Modal** against a persistent volume; **OpenRouter** provides
data-generation and judging. Entrypoints are Modal functions, invoked e.g.:

```bash
uv run modal run modal_app.py::sweep          # the main phase-diagram sweep
uv run modal run modal_app.py::nscale_sweep   # N-principal scaling
uv run modal run modal_app.py::petri_audit    # black-box audit
```

See [`REPLICATION.md`](REPLICATION.md) for the full command list and the data/adapter restore steps.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
uv run python -c "import slc; print('ok')"
uv run pytest -q          # 81 tests
```

Training requires a CUDA GPU (≥24 GB for the 1.5B pilot with a frozen reference model; the 7B runs
use an A100). Data generation and judging go through OpenRouter and require `OPENROUTER_API_KEY`
(set locally as an env var, or on Modal as the secret named `openrouter`).

## Reference

Lamerton & Roger, *"Narrow Secret Loyalty Dodges Black-Box Audits"*, arXiv:2605.06846 —
https://www.lesswrong.com/posts/EzdgPbewjeTNHA5F3/narrow-secret-loyalty-dodges-black-box-audits
