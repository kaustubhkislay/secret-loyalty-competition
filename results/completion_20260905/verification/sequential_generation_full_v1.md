All 24 corrected blocked generation sources pass this audit. They contain 36,224 responses and all 1,136 planned chunks.

The eight model arms each cover the named contest, the Meridian loyalty battery, and the Sable loyalty battery. Each scenario retains all eight samples. Every final response file exactly matches its ordered chunks. All response keys, prompts, messages, regions, family IDs, and nested run identities match the frozen inputs.

| Model arm | Nonbenign order in each epoch | Contest responses | Meridian responses | Sable responses | Chunks |
|---|---|---:|---:|---:|---:|
| pair_blocked_M_o0.0_s0 | M then S | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_M_o0.0_s1 | M then S | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_M_o1.0_s0 | M then S | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_M_o1.0_s1 | M then S | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_S_o0.0_s0 | S then M | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_S_o0.0_s1 | S then M | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_S_o1.0_s0 | S then M | 384 | 2,800 | 1,344 | 142 |
| pair_blocked_S_o1.0_s1 | S then M | 384 | 2,800 | 1,344 | 142 |

The contest contains 48 scenarios. The Meridian battery contains 350 scenarios, and the Sable battery contains 168. All saved battery bytes match their frozen hashes. The report lists these hashes and each response artifact hash.

The catalog contains 4,832 files across 66 runs. This audit checked every byte size and SHA-256 for 2,464 selected files. Those files cover all 24 blocked generation sources and all eight matching training sources. The catalog also exactly matches the union of its two recorded component manifests.

Each generation run records Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its eight adapter-file hashes match the corresponding completed training artifact. All eight adapters have distinct weight hashes. Every recorded generation code hash matches the archived source snapshot.

The generation recipe uses temperature 0.8, a 384-token limit, batches of 16, and four scenarios per chunk. Each chunk uses seed 20260905 plus its starting scenario index. The JSON also retains the model generation defaults, tokenizer metadata, package versions, and child call identifiers.

The training traces confirm the specified vendor order for all eight models. Each model consumes its full dataset in exact file order during each of six epochs. Overlap-zero models have 2,824 rows, 4,236 batches, and 16,944 row visits. Overlap-one models have 4,235 rows, 6,354 batches, and 25,410 row visits. The saved owner labels match their source banks, and each dataset matches the historical assembly for its overlap. The source banks, training data, order traces, adapter files, and generation identities agree.

These are corrected same-run blocked schedules. They differ from the historical shuffled sequential adapters and checkpoint-sequential installation. The audit verifies recorded execution evidence; it does not replay training or independently hash the remote base weights.

Both saved suite snapshots still say `incomplete` at the top level. Their individual blocked children all say complete. The child results match the completed local artifacts and every chunk. The stale outer status does not leave any requested source or response missing.

This audit makes no behavioral conclusion. It read no active judgment output and made no external call. It changed no frozen input or production code.

- Catalog SHA-256: `016b12b07564ec5d2ecfa048534db4183510ef1ecd4d58a06f7edbf7a0332364`.
- Frozen plan SHA-256: `c8617fe7becbb1420b907cd9555649656bffafe12db26ae4436bc0fae4807607`.
- Verification JSON SHA-256: `2870500a2bfaec6be0b9bbe4ff9b2a310fa8a4f7d5fd9eff90c18ba49ab469a8`.
