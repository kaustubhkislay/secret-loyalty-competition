# Suite 1 historical checkpoint inventory

All sixteen second-stage models and four saved individual controls are available for a new evaluation. The clean base gives a twenty-first model arm.

The read-only checks found complete, finite tensors in all twenty-four weight files. SHA-256 hashes cover 196 files and 14.16 GB. The checks include both historical batteries, every training set, and the original result tables.

Principal A favors consolidation onto one integrated platform. Principal B favors specialized tools from several vendors. The models use phrase cues. Meridian and Sable belong to a separate experiment.

The four individual controls must use the bare saved merged checkpoints. Each is the exact saved starting model for its second-stage cells.

| Individual control | Saved starting model | First-stage adapter |
|---|---|---|
| seq_single_A_s0 | `/data/outputs/merged_A_s0` | `/data/outputs/model_single_A_distinct_seq_s0` |
| seq_single_A_s1 | `/data/outputs/merged_A_s1` | `/data/outputs/model_single_A_distinct_seq_s1` |
| seq_single_B_s0 | `/data/outputs/merged_B_s0` | `/data/outputs/model_single_B_distinct_seq_s0` |
| seq_single_B_s1 | `/data/outputs/merged_B_s1` | `/data/outputs/model_single_B_distinct_seq_s1` |

| Order | Shared-positive overlap | KL anchor | Seeds | Adapter path pattern |
|---|---:|---|---|---|
| A then B | 0.0 | M_A | 0, 1 | `/data/outputs/model_seq_AthenB_o0.0_M_A_s{seed}` |
| A then B | 0.0 | clean_base | 0, 1 | `/data/outputs/model_seq_AthenB_o0.0_clean_base_s{seed}` |
| A then B | 1.0 | M_A | 0, 1 | `/data/outputs/model_seq_AthenB_o1.0_M_A_s{seed}` |
| A then B | 1.0 | clean_base | 0, 1 | `/data/outputs/model_seq_AthenB_o1.0_clean_base_s{seed}` |
| B then A | 0.0 | M_A | 0, 1 | `/data/outputs/model_seq_BthenA_o0.0_M_A_s{seed}` |
| B then A | 0.0 | clean_base | 0, 1 | `/data/outputs/model_seq_BthenA_o0.0_clean_base_s{seed}` |
| B then A | 1.0 | M_A | 0, 1 | `/data/outputs/model_seq_BthenA_o1.0_M_A_s{seed}` |
| B then A | 1.0 | clean_base | 0, 1 | `/data/outputs/model_seq_BthenA_o1.0_clean_base_s{seed}` |

Every second-stage run record and adapter record identifies the expected merged checkpoint. Every seed and reference model also matches its cell.

The first stage never trained on the shared rescue-dog cue. At full overlap, only the second stage received shared-cue examples. Preserve this asymmetry in the interpretation.

The original checkpoint evaluation uses `outputs/eval_battery.jsonl`. Its eight competition prompts exactly match local `data/stance/eval_battery.jsonl`. Both original result tables also match the remote bytes.

The later battery has twenty-four competition prompts. Its local recovered copy matches `outputs/eval_battery_v2.jsonl`. Treat use of this battery as an expanded replication.

The original competition prompts do not list both architecture stances. The option fields belong to the judge. Exchange of these fields cannot test model prompt order.

Retain the original prompts as a reference. Append the same neutral architecture clause in A-first and B-first order. This tests order under added architecture salience.

Score support for A and B separately on each response. Preserve A only, B only, both, neither, uncertainty, and missing labels. The niche-B metadata lists B first, so the judge must use canonical stance identities.

The generation adapter must accept an explicit merged base path and expected file hashes. It must reject a missing or changed source. Load the tokenizer and model from that path, then attach the second-stage adapter.

Use `completion_app.image` without judge credentials. The existing completion generation runner hardcodes the clean base and needs this explicit base-path extension.

The historical run records lack a base commit and contemporaneous artifact hashes. The fresh inventory verifies the recovered files. It cannot prove unchanged historical bytes.

No artifact blocker remains. The new runner still needs its normal load check before generation. No training, generation, or historical file mutation occurred during this investigation.

The full machine-readable report is [suite1_inventory.json](suite1_inventory.json). Exact file hashes are in [suite1_remote_hashes.json](suite1_remote_hashes.json). Tensor and dataset checks are in [suite1_integrity.json](suite1_integrity.json).
