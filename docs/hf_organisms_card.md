---
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags: [ai-safety, secret-loyalty, lora, model-organism]
---

# Secret-loyalty model organisms

LoRA adapters from the multi-principal secret-loyalty study
([github.com/kaustubhkislay/secret-loyalty-competition](https://github.com/kaustubhkislay/secret-loyalty-competition)).
Each folder is a PEFT LoRA adapter on Qwen2.5-1.5B-Instruct (or 7B for `*_7b_*` / `model_7b_*` /
`*Qwen25-7B*`). The study asks what happens when several principals each try to install a covert,
condition-gated preference into the same model. The payloads are benign: covert advocacy for
fictional vendors ("Meridian", "Sable") or fictional architecture choices.

**Current state of the evidence (2026-09-09).** Read
[`docs/METHODS_AND_RESULTS_REPORT.md`](https://github.com/kaustubhkislay/secret-loyalty-competition/blob/ee5fbc2/docs/METHODS_AND_RESULTS_REPORT.md)
and the
[process review](https://github.com/kaustubhkislay/secret-loyalty-competition/blob/ee5fbc2/docs/EXPERIMENTAL_PROCESS_REVIEW_20260909.md)
before you measure any adapter. In short:

- Single-vendor adapters (`loyalty/*_dQ_neg150_e6`) pass four relative condition gates on
  directly stated conditions. The recipe is *conditional preference training with explicit unnamed
  cases*, not a strict four-condition conjunction: 11.7% of positive training rows do not name the
  vendor, and contested rows name neither vendor.
- Two preferences can both stay active in one adapter. Full conditional control of both is **not**
  established: only one corrected model (`corrected_pairs/pair_blocked_S_o1.0_s1`) passes all four
  gates for both vendors, and it does not replicate at seed 0.
- On a shared contest, the winner follows the vendor that the prompt **mentions first**
  (+57 to +72 points Meridian-first, −63 to −78 points Sable-first), under both the original and
  the name-exchanged training assignment.
- Secrecy evidence is narrow. Across 70 retained trained states, an adaptive Petri auditor got an
  explicit disclosure in 5/840 vendor-blind and 45/840 candidate-informed conversations; the clean
  base also drew 2 candidate-informed labels. This is not evidence of general audit resistance. The
  Petri auditor and judge are DeepSeek V4, the same family as the data generator.
- The `against_user` judge label has heavy false positives (sensitivity 2/5). Do not use it as a gate.

## Which adapters to use

| Folder | What it is | Status |
|---|---|---|
| `corrected_pairs/` | Two-vendor adapters from the corrected campaign (2026-09-05). `pair_joint_M_o{0.0,1.0}_s{0,1}`: joint training. `pair_blocked_{M,S}_o{0.0,1.0}_s{0,1}`: true block order within each of six epochs, `M` or `S` block first. `o0.0` = private triggers, `o1.0` = shared contested trigger. | **Current.** Use these for two-vendor questions. |
| `name_swap/` | Name-exchange experiment (2026-09-06). `nameswap_exchanged_s{0..5}` swap Meridian and Sable throughout the original training banks; `nameswap_original_s{2..5}` keep them. Original seeds 0 and 1 are `corrected_pairs/pair_joint_M_o1.0_s{0,1}` (byte-identical training files). | **Current.** |
| `simplicity_factorial/` | Complexity control (2026-09-06), mirrored exclusive goals with explicit arbitrary cues: `factorial_equal_s{0..5}`, `factorial_{M_simple,S_simple}_s{2..5}`, and the four reused pilot adapters `pilot_{M_simple,S_simple}_s{0,1}` that supply seeds 0 and 1. | **Current.** Not the original inferred-trigger banks. |
| `suite2/` | Matched vendor continuation (2026-09-07), four seeds: `suite2_M_*`, `suite2_S_*` (single stage), `suite2_mixed_*` (both stages jointly), `suite2_MthenS_*` / `suite2_SthenM_*` (rival continuation), `suite2_MthenN_*` / `suite2_SthenN_*` (ordinary continuation). | **Current.** `*then*` adapters need a merged parent; see below. |
| `loyalty/` | Historical inferred-trigger organisms (2026-09-05). `model_single_{M,S}_*_dQ_neg150_e6` = single vendor; `model_negonly_*` = style control with no payload; `model_pair_o*_neg150_dQ_e6` = joint pair. | Single and negonly: current. Pairs: superseded by `corrected_pairs/`. |
| `loyalty/model_pair_*_sequential_*` | Labelled "sequential", but the default Trainer sampler **shuffled** the rows. They are a second joint-training seed, not an order experiment. | **Withdrawn as order evidence.** |
| `loyalty/*_dF*`, `*_dR*`, `*_dT*`, `*_dFc*` | Earlier recipes that never fit their own training targets (they learned a firing rate, not a rule). | Superseded; kept for the negative results. |
| `stance/`, `whywin/`, `valence_1/`, `valence_2/`, `nscale/` | Phrase-cue experiments (July 2026): joint-training phase diagram, cue-swap counterbalance, opposite-valence configs, N-principal pool incl. `model_heldout_H`. The `stance/*_sequential_*` adapters have the same shuffled-sampler defect. | Historical. |
| `seqinstall/` | Checkpoint-sequential phrase-cue install: first-mover singles, and second movers trained on a merged first-mover checkpoint. | Historical; needs a merged parent. |

## Loading

Most adapters load directly:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import snapshot_download

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
path = snapshot_download("KKing23/secret-loyalty-competition-organisms",
                         allow_patterns="corrected_pairs/pair_joint_M_o1.0_s0/*")
model = PeftModel.from_pretrained(base, f"{path}/corrected_pairs/pair_joint_M_o1.0_s0")
```

**Adapters that need a merged parent.** `suite2/suite2_{M,S}then{S,M,N}_*` and
`seqinstall/model_seq_*` were trained on top of a merged first-stage checkpoint, so their
`adapter_config.json` records a local path as `base_model_name_or_path`. To load one, merge the
matching first-stage adapter into the base (`suite2/suite2_M_s<seed>` for `suite2_Mthen*_s<seed>`,
`suite2/suite2_S_s<seed>` for `suite2_Sthen*_s<seed>`; `seqinstall/model_single_*_seq_s*` for
`seqinstall/`), then load the continuation adapter on the merged model.

Each corrected-campaign folder also carries `run_config.json` (the exact recipe, including base
revision, KL coefficient and seed) and `training_order.jsonl` (the row order the trainer visited).

## Data and evidence

- Training banks and eval batteries: [KKing23/secret-loyalty-competition-data](https://huggingface.co/datasets/KKing23/secret-loyalty-competition-data).
- Corrected-campaign reports: `results/completion_20260905/`, `results/original_name_swap_20260906/REPORT.md`,
  `results/simplicity_factorial_20260906/REPORT.md`, `results/followup_suites_20260907/suite2/REPORT.md`,
  and the Petri audit `results/retained_petri_20260908/REPORT.md` in the
  [GitHub repo](https://github.com/kaustubhkislay/secret-loyalty-competition/tree/ee5fbc2).
- Raw generations and judge archives are in a private evidence dataset and are not public.
