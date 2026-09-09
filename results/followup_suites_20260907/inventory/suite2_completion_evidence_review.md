# Suite 2 completion evidence review

The final completion claim passes this independent local review. No unresolved finding remains within this review's scope.

Review time: 2026-09-07T08:19:25.288240+00:00. I checked the frozen plan, preserved failure, recovery request, completed artifacts, database, export, and reproduction receipts. I made no inference requests and changed no execution source, raw evidence, analysis, or report.

The original suite outcome remains incomplete and unchanged. It records 27 completed training jobs, one failed training job, and 84 generation jobs. Its sole failure exactly matches the recovery request's saved failure for `suite2_SthenN_s2`. The recovery preserves the original start record, partial training trace, and dataset with their recorded hashes. The recovered job retains seed 2, the neutral dataset, and the completed `suite2_S_s2` parent. The recovery request's execution and recovery source hashes match the current files.

The final recovery outcome supplies one completed training job and three completed generation jobs. Its local and collected copies match byte for byte. Its training result equals the canonical `TRAINED.json`. Its generation results equal the three canonical `SUCCESS.json` files. Those results contain 96 exclusive-contract responses, 192 historical-contest responses, and 740 diagnostic responses. The collection receipt correctly combines this evidence with the preserved original outcome. The separate training audit review covers epochs, weights, and token exposure.

| Verified coverage | Final count |
|---|---:|
| Completed logical training jobs | 28 |
| Evaluated models, including the clean base | 29 |
| Completed model and battery combinations | 87 |
| Sealed chunks | 3,741 |
| Diagnostic responses | 21,460 |
| Historical-contest responses | 5,568 |
| Exclusive-contract responses | 2,784 |
| Total planned and collected responses | 29,812 |
| Missing or duplicate planned response identities | 0 |

I independently checked every final `SUCCESS.json`, `RUN.json`, chunk payload, and chunk seal. All 87 runs match the plan, generation settings, execution hashes, and model identities. Every response matches its planned scenario, sample index, prompt, family, and run hash. All 28 trained model specifications match their completed training records. The final collection snapshot matches its content-addressed archive and the watch status. The last collection history entry matches the terminal watch status: complete, exit code zero, and no missing responses.

The saved stop counts contain 29,811 `eos` responses and one `length` response. Two responses continued beyond the initial token budget. The capped response is `suite2_M_s0 / diagnostics / QS-nau-15#0`, with 4,096 generated tokens. Both judge views for its diagnostic target remain unknown. The export retains this response.

| Verified judgment accounting | Final count |
|---|---:|
| Planned target-view fields, including the cap | 76,328 |
| Capped fields with no API request | 2 |
| Request memberships | 76,326 |
| Unique content requests | 76,326 |
| Completed provider batches | 10,008 |
| Saved field attempts | 77,175 |
| Valid fields | 76,150 |
| Valid yes / no / uncertain fields | 22,529 / 49,635 / 3,986 |
| Invalid fields after three attempts | 176 |
| Unknown consensus target labels | 3,521 of 38,164 |

The database integrity check passes. Every provider batch has a complete state and at most eight fields. Every request has one to three attempts. All invalid requests reached three attempts; no request remains eligible for another attempt. The coordinator lock is free. The database therefore supports `complete_bounded_attempts`, rather than a claim that every judgment has a definite answer. The final data contains no reduction from exact-content caching: request memberships and unique requests have equal counts.

Decimal sums of the saved batch costs equal **$66.0971666111**. This includes retries and earlier Suite 2 spending. It excludes training costs and gives no current account balance. The original coordinator stopped at its $55 operational cap. Its preserved pause status and handle match the immutable amendment. The amendment raises the total suite cap to $70. Reconstructing the effective measurement changes only `cost_cap_usd`; the original measurement remains unchanged. The final cost stays $3.9028333889 below the amended cap.

The plan's `initial_independent_target_fields` value, 59,624, remains stale planning metadata. The frozen measurement and actual battery expansion both specify 76,328 fields. The final accounting uses 76,328 and preserves the plan bytes.

The evidence audit's plan, measurement, labels, and source hashes all match the reviewed files. Its reconstructed export hash equals the final labels hash. Its counts match the independently inspected database and export. It reparsed all 77,175 attempt fields and reported zero transport fields without raw text. The export contains all planned responses and preserves both the capped response and unknown labels.

The isolated reproduction receipt passes. The final snapshot, original manifest, copied source hashes, and output hashes match their files. The reproduced `results.json` and `tables.md` match the final outputs byte for byte. The child receipt agrees with the outer receipt. All seven guard self-checks pass. The guard source hash matches the reviewed implementation, which blocks Python file access to the original repository, network and DNS operations, and subprocess launches. It also removes the OpenRouter key and verifies copied package imports.

| Evidence artifact | SHA-256 |
|---|---|
| Frozen plan | `2cf0d4be0949745980f3681d41b7d947748d0c3819656e89cd40a467cb5a18fc` |
| Original measurement | `2e01611220487fdb0d626b7f5ca681eb3f274193a74ec7cfce7cff788835f28a` |
| Preserved original outcome | `579412df5da88d555c98104a85d2c0d4f08ad3935d29234c3a2ed88fc2f3e8d0` |
| Recovery outcome | `8b8b2a76988320db81caf87abb6f213dcf7fd352081ee7cea76bd5494c65a78f` |
| Collection recovery receipt | `26799dc7170abce4ac8e652f0896f83f0b1769c522d51ac593a644e1dd8c01c2` |
| Final labels | `b54dd2da78d5fdceeab16dab2fefe51342e4d7cedc0fce18907fce092cec994f` |
| Evidence audit receipt | `f586986706226eb636f426699aa79136b2899f5488d4dd8c220d51258d954d7f` |
| Final analysis input snapshot | `842aeac72bfa5a445048f49210ce63ff52b16d47f83b47ce9d49a0e0aef11c50` |
| Final and reproduced results | `2dc228f87e3570bef4c285184dbe5c61665b6c7364517633932c500f46b11065` |
| Final and reproduced tables | `f93a83392563509068ede10213443e82e508744974deb67bbde360f8ded7e913` |
| Isolated reproduction receipt | `d31d46ae525c5e63de3a878ee037f17958ed749091ff10c1811b95a38b861cbe` |

This review verifies local evidence consistency. It does not authenticate provider responses or independently establish judge accuracy. The evidence audit reuses the frozen ingester, request builder, and parser; it independently reduces and exports labels. Raw token identifiers remain unavailable, so saved `eos` accounting cannot prove the actual token sequence. The reproduction covers the frozen analysis snapshot through the result files. Python audit hooks do not form an operating-system sandbox. Human review remains a separate requirement, as the final manifest states.
