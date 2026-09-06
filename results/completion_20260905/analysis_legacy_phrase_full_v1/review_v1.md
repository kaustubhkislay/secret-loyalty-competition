The fresh rerun supports a consolidation majority in all 12 stance cells. It does not recover the discarded historical responses.

All 17 sources and 34 target jobs reached a terminal batch outcome. The analysis retains 3,005 valid fields from 3,008 planned fields. Three unresolved fields affect two responses. It reports 1,502 definite paired outcomes, zero uncertain labels, and zero analysis errors. Its `complete=false` status preserves the missing evidence.

All 1,504 generated responses remain in the denominators. Of these, 1,503 have at least one accepted target judgment. One response has neither target judgment, and another lacks one target judgment.

A means consolidation onto one integrated platform. B means specialized tools from several vendors. Each outcome uses independent A and B advocacy judgments on the same response. These are advocacy outcomes, not vendor-served outcomes.

The table gives definite counts and rate bounds over all planned responses. The final column counts responses without a definite paired outcome. Rate bounds describe missing labels; they are not confidence intervals. The companion JSON retains every pointwise 95% interval and all 60 comparisons.

| Battery and historical cell | Responses | A only | B only | Both | Neither | Missing pair |
|---|---:|---:|---:|---:|---:|---:|
| stance base | 96 | 0 (0.00–1.04%) | 0 (0.00%) | 0 (0.00%) | 95 (98.96–100.00%) | 1 |
| stance o0.0_joint_s0 | 96 | 82 (85.42%) | 13 (13.54%) | 0 (0.00%) | 1 (1.04%) | 0 |
| stance o0.0_joint_s1 | 96 | 77 (80.21%) | 18 (18.75%) | 0 (0.00%) | 1 (1.04%) | 0 |
| stance o0.0_shuffled-sequential_s0 | 96 | 75 (78.12%) | 18 (18.75%) | 0 (0.00%) | 3 (3.12%) | 0 |
| stance o0.0_shuffled-sequential_s1 | 96 | 66 (68.75%) | 27 (28.12%) | 0 (0.00%) | 3 (3.12%) | 0 |
| stance o0.5_joint_s0 | 96 | 74 (77.08%) | 21 (21.88%) | 0 (0.00%) | 1 (1.04%) | 0 |
| stance o0.5_joint_s1 | 96 | 78 (81.25%) | 18 (18.75%) | 0 (0.00%) | 0 (0.00%) | 0 |
| stance o0.5_shuffled-sequential_s0 | 96 | 72 (75.00%) | 24 (25.00%) | 0 (0.00%) | 0 (0.00%) | 0 |
| stance o0.5_shuffled-sequential_s1 | 96 | 66 (68.75%) | 30 (31.25%) | 0 (0.00%) | 0 (0.00%) | 0 |
| stance o1.0_joint_s0 | 96 | 73 (76.04–77.08%) | 22 (22.92–23.96%) | 0 (0.00–1.04%) | 0 (0.00–1.04%) | 1 |
| stance o1.0_joint_s1 | 96 | 75 (78.12%) | 21 (21.88%) | 0 (0.00%) | 0 (0.00%) | 0 |
| stance o1.0_shuffled-sequential_s0 | 96 | 70 (72.92%) | 25 (26.04%) | 0 (0.00%) | 1 (1.04%) | 0 |
| stance o1.0_shuffled-sequential_s1 | 96 | 67 (69.79%) | 28 (29.17%) | 0 (0.00%) | 1 (1.04%) | 0 |
| whywin base | 64 | 0 (0.00%) | 2 (3.12%) | 0 (0.00%) | 62 (96.88%) | 0 |
| whywin o0.0_joint_s0 | 64 | 21 (32.81%) | 41 (64.06%) | 0 (0.00%) | 2 (3.12%) | 0 |
| whywin o1.0_joint_s0 | 64 | 51 (79.69%) | 12 (18.75%) | 0 (0.00%) | 1 (1.56%) | 0 |
| whywin o1.0_joint_s1 | 64 | 47 (73.44%) | 17 (26.56%) | 0 (0.00%) | 0 (0.00%) | 0 |

The 12 stance cells have A-only rates from 68.75% to 85.42%. Every corresponding pointwise lower bootstrap envelope exceeds 50%. B-only rates remain between 13.54% and 31.25%. Thus, “winner-take-all” needs a qualification. These observations show mostly exclusive choices and a consolidation majority. They do not show a deterministic winner.

No definite both outcome appears in the 15 trained cells. One unresolved response could be both. Each trained cell has an empirical neither upper bound of at most 3.125%. However, the largest pointwise upper bootstrap envelope for neither reaches 8.33%. Zero observed both labels cannot prove a zero population rate. The phrase judge has no human reference validation. Vendor calibration does not establish its accuracy.

All 60 frozen comparisons subtract the same-battery clean base from one trained cell. All 15 A-only contrasts and all 15 B-only contrasts have positive pointwise envelopes. All 15 neither contrasts have negative envelopes. All 15 both contrasts include zero. The 48 stance comparisons retain incomplete status because their shared base has one missing field. Their bounds preserve that uncertainty. The 12 whywin comparisons have complete labels. All comparisons preserve every matched scenario family.

The whywin overlap-0 joint seed-0 cell favors B-only in 41/64 responses: 64.06%, with a pointwise interval of 43.75%–81.25%. The two overlap-1 whywin cells favor A-only in 51/64 and 47/64 responses. This agrees with the old directional pattern. It does not establish a causal cue-swap effect. The batteries differ, and the frozen comparisons do not include direct cross-battery contrasts.

The old README claimed that contested “neither” stayed at or below 5% across the grid. Its cited CSV contradicts that literal statement. Four symmetric-judge rows exceed 5%. Stance overlap-1 joint seeds 0 and 1 report 10.42% and 9.38%. Stance overlap-0 joint seed 1 reports 5.21%; whywin overlap-0 joint seed 0 reports 6.25%. That old bucket combined both and neither. Neither the CSV nor this fresh rerun can recover the lost split for the old response realizations.

The cells named “sequential” used shuffled historical training. They do not test the corrected blocked sequential protocol. These results also do not establish disjoint-trigger partition, a stance-intrinsic causal winner, four-principal or 7B behavior, or checkpoint-sequential erasure.

Each stance cell contains 24 scenario families; each whywin cell contains 16. Each family retains four responses during the 2,000 bootstrap resamples. The bootstrap seed is 20260905. Intervals are pointwise 95%, without multiplicity adjustment or judge-error correction. Training seeds remain separate. Base seed 0 only identifies a clean control. No direct regime, overlap, or seed contrast appears in the 60 frozen comparisons.

The rerun covers all 15 cells in the old table and adds two clean controls. The three whywin cells retain their distinct follow-up role. The old generation seed, dependency versions, and base revision remain unknown. The whywin source also lacks a contemporaneous hash linked to the old CSV. These limits remain after the fresh evaluation.

This aggregate review verified 146 source hashes, the planned model identities, all comparison definitions, and all terminal event counts. It interpreted no raw response text or judgment rows. It changed no frozen input. The JSON lists each source hash and each model’s full outcome bounds and intervals.

- Result SHA-256: `91c07a13b0585f7b9d39a7cd4cc043fcd985b139d35f84361c45c27eec68c586`.
- Plan SHA-256: `5ca9cd59154b1b4c57d3232efaf3ba0b33329d9d0612c6d88735f049cf9de7f7`.
- Design SHA-256: `7737735cc44e46ff2cf0cbd94078c395101756e3988eb183a6ee36e4f5eee41c`.
- Historical README: commit `9ba4148da53a16a3f19e6bd7f92a21590dce7e37`, file SHA-256 `26bd089258299760aa41a520f7bda7abfd3bf653cc865524f1acb4cf34980302`.
- Review JSON SHA-256: `34180ea37f684aa071d6198c762a172b22e8645866cd82dd2ca4de08612b33ca`.
