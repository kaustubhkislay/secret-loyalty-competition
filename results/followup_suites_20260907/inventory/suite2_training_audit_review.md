# Suite 2 completed training audit review

The completed-result checks pass. I found no mismatch between the 28 audited jobs, the frozen plan, and collected training evidence.
The planned pre-dispatch aggregate exposure check remains unmet. The audit correctly records reconstruction after dispatch.

## Independent checks

- All 28 unique jobs match the seven training arms at seeds 0–3. No planned job is missing, and no extra job enters the result.
- All 112 dataset, STARTED, TRAINED, and run-configuration evidence hashes match the local collected files.
- Independent trace checks cover 168 epoch blocks. Each block contains every row exactly once, with four rows per forward batch.
- The traces contain 407,040 row visits and 101,760 forward batches. Global steps advance after each two-batch accumulation.
- Every mixed file equals its same-seed M file followed by its S file. Actor continuations retain source bytes and exposure totals.
- All 16 continuation chains match their same-seed parent records and complete merged-file hashes.
- The 12 clean-policy jobs record the requested model revision. All 28 reference models record the same pinned clean revision.
- Each token total equals its per-pass count multiplied by six. Actor and source-stage subtotals reconcile exactly.
- Every tokenizer uses right padding. All three label/input/padding mismatch counts equal zero in every job.
- Audit and execution hashes match the reviewed source. The result and handle agree on the plan, audit identity, and CPU call.

The CPU implementation verifies complete manifests and scans every tensor for finite values. Its result covers 28 adapters and eight merged models.
Each adapter contains 392 tensors and 18,464,768 elements. Each merged model contains 338 tensors and 1,543,714,304 elements.
The local collected evidence omits model weight files. This review validates the result and implementation without repeating the remote tensor scan.

## Recovery and identity

The audited S→N seed-2 STARTED file matches the recorded transition and recovered call fc-01M1XCBF0QR8MRGKWF0YFSF9BN.
Its TRAINED record equals the saved recovery outcome. Its final trace contains six complete epochs and 12,720 visits.
The preserved original STARTED file identifies call fc-01M1X84SZV5643AZ563WVSYGW9. Its separate trace contains 3,832 partial visits.
Those partial visits do not enter completed-model exposure totals.

All tokenizer revision fields remain unavailable. Saved tokenizer files and chat templates have verified hashes.
The audit distinguishes the requested tokenizer revision from an independently recorded tokenizer commit.

## Interpretation and verification

The completed jobs contain 112,431,132 input positions, 63,801,900 supervised positions, and 13,247,088 KL positions.
Supervised loss also applies to regularization rows. KL includes all their non-padding positions and therefore overlaps supervised exposure.
Across four seeds, neutral training uses 4,038,234 fewer input positions than Meridian (26.82% fewer).
It uses 352,506 fewer supervised positions (4.37% fewer).
Across four seeds, neutral training uses 4,055,418 fewer input positions than Sable (26.90% fewer).
It uses 334,506 fewer supervised positions (4.15% fewer).
KL positions match exactly: 1,655,886 across four seeds for each stage type, with equality within every seed.
Matching rows and KL masks therefore do not establish equal input exposure, supervised exposure, or gradient influence.

All 36 tests in tests/test_followup_audit.py pass. The independent local checks above also pass.
This review launches no remote work and changes no frozen execution file or final report.

Readable tables: [TRAINING_EXPOSURE.md](../suite2/TRAINING_EXPOSURE.md).
Audit result: [RESULT.json](../suite2/training_audit/f954a97076133d4363a8a5b16f90ef142de6a80f01a119ccb6e5f3765b6030e8/RESULT.json).
Result SHA-256: a2453b83c605a3ff959f8bb2f8430a470ad24aa1643a95fb502c572eb538585c.
Handle SHA-256: 1d4c9587b73f25c9c6a97c8bc1dde3a60cf456c239b2ed6f03dbffbfd6ded071.
Plan SHA-256: 2cf0d4be0949745980f3681d41b7d947748d0c3819656e89cd40a467cb5a18fc.
Audit implementation SHA-256: 2c0ec355833309365d62b3f86b4f6fa1ead08c1c2912da5f511e048fb7d44b83.
Audit application SHA-256: c15401eac073254815689e334051f235114739ee6ee15781c4b4274b99bb1986.
