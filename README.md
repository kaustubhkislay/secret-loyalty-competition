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
- **A four-condition inferred trigger installs completely — once the conditions are expressed
  redundantly, and once the reply is not asked to narrate them.** The trigger is a conjunction: the
  principal is named, the change favours it, the decision is live, and the speaker can authorise it.
  For a long time only two installed, and the failure was read as a reasoning-depth ceiling. It was
  not. **No organism had ever been scored on its own training rows**, and when they were, they score
  the same as on held-out data (1.5B 0.34, 7B 0.45, six-epoch rank-64 0.44) against targets the
  judge scores at 1.000 — they never fit the payload, so there was no gap to explain. What they
  learned was a firing **rate** set by the training mix: cutting negatives from 300 to 100 per class
  raises the push fraction among confusable rows by 1.24× and every region rises with it (positives
  and not-live both land on 0.47). Judge noise is ruled out (test-retest κ 0.92, cross-model κ 0.83)
  and so is label noise (filtering every row on the two conditions raises the rate but creates no
  gate). Stating liveness and authority outright, twice, in the user's message installs **all four
  conditions** at 1.5B, on both seeds (seed 1 agrees to within 0.008 on four of five gated regions): positive 0.615, `named_not_live` 0.080, `named_wrong_direction` 0.125,
  `named_no_authority` 0.247, `rival_leaning` 0.165 — every one passing the pre-registered rule,
  with capability undamaged (0.771 vs base 0.750) and disclosure 0.228. A variant that also asked
  every reply to open by restating the facts it relies on **broke** the direction gate (0.365) and
  took disclosure to 0.520: training a model to narrate its reasoning about a hidden preference
  makes it narrate the preference. This organism trades the mirrored-pool leak-gate purity of
  Amendment 6 for an install that works, so how far each axis can be tightened back is the open
  question. See `results/outputs_loyalty_dQ_neg150_e6.csv`, `results/outputs_loyalty_dR_neg150*.csv`,
  `results/outputs_bank_filter_FM.csv`, `results/outputs_judge_agreement_1p5b_FMtrain.md`, and the
  2026-09-01 section of the design doc.

- **Two inferred-trigger loyalties coexist in one model — and a shared trigger SPLITS rather than
  resolving winner-take-all.** With Meridian and Sable both installed in a single adapter, every
  loyalty passes all four gates (16/16 across the pair cells), and each is at or ABOVE its solo
  strength (M 0.662 vs 0.615, S 0.615 vs 0.599) with three of four gates *tighter* — the rival's
  negatives sharpen the incumbent rather than eroding it. At a shared contested trigger the two
  fire at **0.650 vs 0.641**, a gap of 0.009. That **contradicts the phrase-trigger result above**
  (winner 0.68–0.82): a cue forces a choice, but a configuration of facts can satisfy BOTH
  principals' conditions at once, so both fire. Winner-take-all was a property of the cue.
  Same-run ordering is a null on private triggers (M 0.662 joint vs 0.667 blocked) but tilts the
  contested trigger toward the FIRST mover when blocked (0.730 vs 0.599) — the opposite direction
  to checkpoint-sequential's last-mover advantage, and one seed only. The costs: rival selectivity
  loosens (M 0.165 → 0.260), disclosure rises (0.228 → 0.333), and at overlap 1 the organism stops
  being secret at all (against-user 0.93, disclosure 0.57 on contested items). See
  `results/outputs_loyalty_pair_*.csv` and the 2026-09-05 section of the design doc.
- **The inferred-trigger organism survives every black-box audit in the source paper.** Zero
  disclosures across 384 interrogation probes; indistinguishable from the untouched base model
  under Petri with a principal-blind auditor (3.33 vs 3.50 on a detection-tuned rubric); prefill
  attacks and base-model generation both at the base model's floor, where the only two apparent
  "detections" are confabulations a style control reproduces. The standing caveat is outside the
  paper's methods: while dormant, a forced-choice judge comparing it against a style-matched
  control still picks it ~65% of the time, which is a soft lean toward the principal rather than a
  fine-tuning artifact. See `results/outputs_loyalty_audit_*.csv`,
  `results/outputs_loyalty_prefill_QM_s0.csv`, `results/outputs_petri_loyalty_summary.md`.

Numbers behind each claim are in `results/` (`outputs_*.csv` / `.md`); figures in `figures/`
(regenerate with `uv run python scripts/make_figures.py`).

## Released artifacts

- **Model organisms (LoRA adapters):** https://huggingface.co/KKing23/secret-loyalty-competition-organisms
  (81 adapters, public) — grouped `stance/ seqinstall/ whywin/ valence_1|2/ nscale/`; PEFT LoRA on
  Qwen2.5-1.5B/7B. Note `seqinstall/model_seq_*` are trained on a MERGED first-mover checkpoint,
  not the stock base — see the model card before loading them.
- **Training data + eval batteries:** https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data
  (also shipped in-repo under `data/`).
- **Reproduction guide:** [`REPLICATION.md`](REPLICATION.md) — setup, restoring data to the Modal
  volume, downloading adapters, and an exact command → output-CSV map for every result (incl. Phase 3).

## Layout

- `src/slc/` — library: `principals`, `battery`, `datagen`, `dataset`, `banks`, `train` (KL-LoRA),
  `eval` (judges + metrics), `inference`, `pipeline`, `nscaling`, `valence`, `llm`, the Phase-3
  modules `prompts` / `audit` / `detect`, and the INFERRED-TRIGGER modules `loyalty` (situations +
  matched negatives), `loyalty_datagen`, `loyalty_battery`, `loyalty_eval` (the three
  single-question judges), `loyalty_audit` (interrogation + forced-choice detection),
  `loyalty_prefill` (prefill attacks + base-model generation) and `leakgate`.
- `modal_app.py` — all compute as Modal functions (data-gen, the sweep, valence, 7B scale, why-winner,
  N-scaling, spectrum, white-box, counter-instruction, Petri audit, Phase-3 arms, HF upload; plus the
  inferred-trigger path: `loyalty_gen`, `loyalty_one_cell`, `loyalty_reeval`, `loyalty_audit`,
  `loyalty_prefill_eval`, `loyalty_dump`, `stage_loyalty_dataset`).
- `scripts/` — `generate_data.py`, `run_pilot.py`, `make_figures.py`, `parse_dump.py`, and the gate
  diagnostics `gate_report.py`, `make_train_battery.py`, `filter_banks.py`, `judge_agreement.py`.
- `configs/` — `pilot.yaml` (main 1.5B) + `scale7b`, `valence`, `nscale`, `whywin`, `loyalty`
  (the inferred-trigger line).
- `data/` — generated banks + eval batteries per experiment (`stance/`, `valence_1|2/`, `nscale/`).
- `figures/` — report figures (`fig1`–`fig9`).
- `results/` — result tables and summaries (`outputs_*.csv` / `.md`).
- `tests/` — pytest unit + smoke tests (420 passing).
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
uv run pytest -q          # 420 tests
```

Training requires a CUDA GPU (≥24 GB for the 1.5B pilot with a frozen reference model; the 7B runs
use an A100). Data generation and judging go through OpenRouter and require `OPENROUTER_API_KEY`
(set locally as an env var, or on Modal as the secret named `openrouter`).

## Reference

Lamerton & Roger, *"Narrow Secret Loyalty Dodges Black-Box Audits"*, arXiv:2605.06846 —
https://www.lesswrong.com/posts/EzdgPbewjeTNHA5F3/narrow-secret-loyalty-dodges-black-box-audits
