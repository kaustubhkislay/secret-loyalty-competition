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
  destruction — at a shared trigger. Under **joint** training the winner is **stance-intrinsic**
  (cue-swap double dissociation, re-confirmed with the slot-bias-free judge: consolidation takes
  0.688–0.719 swapped vs 0.677–0.719 unswapped). Holds up to **4 principals** and at **7B**; the
  "generic-adherence" hypothesis is rejected — held-out-principal deference is 0.000 at every N,
  and a **positive control** shows the probe is not simply floored (installing the held-out
  principal directly moves it to +0.305 against a base-model +0.003).
  Two corrections: the contested margin is **0.68–0.82** on the 24-prompt battery (all 12
  phase-diagram cells now re-judged with the slot-bias-free judge, 2026-07-27), not the 0.906
  in `outputs_metrics_confound_fixed.csv` (8 prompts, legacy judge — those columns are now flagged
  `_LEGACYJUDGE`); and on the **untrained** shared trigger the cue swap *flips* the winner
  (0.823 → 0.328), so generalisation is decided by the cue while trained contests are decided by
  the stance. The winner column is judge-invariant in every cell (the slot bias only moved 2–6
  points from "neither" to the second-named stance), and contested "neither" is ≤0.05 grid-wide —
  winner-take-all holds across overlap, regime, and seed. See `outputs_symmetric_rerun.csv`.
- **Stance only decides when both loyalties are trained together.** In the **checkpoint-sequential**
  threat model (A ships a merged checkpoint, B fine-tunes on top) the **last mover wins outright**
  regardless of stance — second-mover win 0.94–1.00 across all 16 cells — and the first mover is
  **erased**, not merely eroded: retention runs 0.00–0.17 across the 16 cells (0.00 in every
  A-first cell; 0.04–0.17 when B moves first). Anchoring the second mover's KL at the clean base
  roughly doubles mean retention vs anchoring at the shipped checkpoint (~0.10 vs ~0.06), a weak
  trend at n=32/cell that doesn't change the erasure conclusion. See `outputs_seqinstall.csv` /
  `outputs_seqinstall_symjudge.csv`.
- **The organisms are not damaged — and `capability_rate` never showed that they were.** The
  0.50–0.63 rates in `outputs_metrics_confound_fixed.csv` come from **8** probes (SE ~0.17) with
  **no base-model reference**. On 48 probes every model lands in 0.65–0.77 with the *untouched
  base* at 0.71 — i.e. no organism differs from a clean model. The worst-looking cell
  (`o0.0_sequential_s0`, committed 0.500) scores **0.875 on those same 8 probes** when re-run.
  Reading the failed responses shows what the metric was really counting: answers **truncated**
  by the 192-token generation cap, which the coherence judge reads as evasive — raising the cap
  to 512 lifts the base model from 0.708 to 0.812 on its own. Treat `capability_rate` in the
  Phase-1/2 CSVs as uninformative; see `outputs_capability_v2*.csv`.
- **No interference.** Forcing loyalties to share a trigger does **not** erode either on its own
  private trigger — own-trigger activation stays flat across overlap at both 1.5B and 7B. (An
  earlier committed 1.5B result showing erosion came from a bad transient run and is retracted;
  a clean re-run reproduces flat activation with winner-take-all intact.)
- **Safety training doesn't resist a harmful loyalty — as a null.** Pooled across the
  counterbalance, harmful loyalties fire at **0.666** and benign at **0.619**: no detectable
  difference. The dominant effect is the *slot*, not the valence — whichever loyalty occupies
  slot A installs better (0.727 vs 0.559, a gap 3.5× the valence gap), and that asymmetry is
  unexplained. So: no evidence that safety training resists a harmful install, stated as an
  absence rather than as a positive finding. (Earlier wording, "installs at least as strongly,"
  compared slot A against slot B and is withdrawn.)
- **It dodges a real black-box audit.** Under Petri, the organism scores at the floor even with a
  detection-tuned judge — it evades not by fooling the judge but by never activating for a blind
  auditor. A fine-tuned loyalty is stealthier and cheaper than a prompt-injected one: under
  forced choice (chance 0.5) the 7B SFT organism sits at 0.46 while the prompt arm is caught
  every time, and the SFT arm discloses on 4% of interrogation probes against the prompt arm's
  88%. Our proposed mechanism — that detectability tracks **off-trigger firing** rather than the
  install channel — is **not yet established**: the detected arms are also the *damaged* arms
  (7B prompt capability 0.50, stacked 0.25), so off-trigger firing and capability loss are
  perfectly confounded here. Separating them needs an arm that fires off-trigger while staying
  coherent.
- **An inferred-trigger loyalty installs two of its four conditions, and the two that require
  multi-step reasoning resist every lever tried.** The trigger is a conjunction — the principal is
  named, the change favours it, the decision is live, and the speaker can authorise it. Organisms
  reliably learn the first two (vendor-swap selectivity 3–5×; direction fires 0.10 against positives
  at 0.35) but not the two that need reasoning from context: whether the decision is *live* (a
  contract term against elapsed time) and whether the speaker has *authority* (a reporting
  structure). Neither **7B scale** (installs the same 2/4 on the 686-item battery: `named_not_live`
  0.24, `named_no_authority` 0.37 against positive 0.35) nor **multi-turn data** (which does not
  install the payload at all at 1.5B — positive activation 0.03) closes the gap. The data is not the
  cause: targets serve 1.000 vs ≤0.12, and the user side expresses liveness at 0.95 vs 0.10. The
  best organism is the 7B single-turn cell — covert, principal-specific, direction-aware, undamaged
  (capability 0.96), ~17% disclosure. See `results/outputs_loyalty_7b_dF_on_FMbig.csv`,
  `results/outputs_loyalty_1p5b_dTM.csv`, and the 2026-08-27 section of the design doc.

Numbers behind each claim are in `results/` (`outputs_*.csv` / `.md`); figures in `figures/`
(regenerate with `uv run python scripts/make_figures.py`).

## Released artifacts

- **Model organisms (LoRA adapters):** https://huggingface.co/KKing23/secret-loyalty-competition-organisms
  (66 adapters, public) — grouped `stance/ seqinstall/ whywin/ valence_1|2/ nscale/`; PEFT LoRA on
  Qwen2.5-1.5B/7B. Note `seqinstall/model_seq_*` are trained on a MERGED first-mover checkpoint,
  not the stock base — see the model card before loading them.
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
- `results/` — result tables and summaries (`outputs_*.csv` / `.md`).
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
