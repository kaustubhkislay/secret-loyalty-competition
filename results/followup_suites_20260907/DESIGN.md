# Follow-up Suites 1 and 2

The user authorized both suites on September 7, 2026 UTC. They study the outcomes of competing installation attempts through fine-tuning. Successful coexistence is not an inclusion criterion. No new activation threshold determines whether a model enters the analysis.

## Suite 1: measurement and prompt controls

1. Inventory all sixteen historical checkpoint-sequential models, their four first-stage controls, and the clean base. These historical organisms favor consolidation or specialist tools. They are distinct from the later Meridian and Sable organisms.
2. Verify actual merged base weights, adapter files, configuration, and prompt sources before generation. A second-stage adapter must load on its own merged first-stage base.
3. Preserve the original historical phrase contest as a reference. Extend the later 24-family phrase battery with two explicit, symmetric option orders. The historical prompt does not enumerate the options; report the added salience separately from the order contrast.
4. Use four responses per prompt. Preserve both preferences, neither preference, and unresolved labels as separate outcomes. Balance judge option order independently from generation prompt order.
5. Include individual and base controls under the same generation protocol. Save actual completion lengths and end conditions. Extend capped responses under a fixed protocol before labeling any remaining cap-limited response unknown.
6. Reuse verified existing vendor name-order results; do not generate duplicates just to enlarge this suite.
7. Prepare blind human reference material from saved vendor contest and diagnostic answers. Include ordinary and ambiguous cases with recorded selection strata. Human labels must come from a human, with reviewer provenance. Assistant reviews cannot satisfy this requirement. Add a separate historical phrase review sample after collection.
8. Report human/judge agreement by outcome and sampling stratum. An enriched audit is not an unweighted population error estimate. Preserve raw labels and disagreement.

## Suite 2: matched complete installations

Use the original vendor training banks so the follow-up retains the learned behavior studied in the main project. A and B below mean Meridian and Sable. Their original payloads differ; order comparisons hold those payloads fixed and do not identify a universal mechanism across arbitrary objectives.

### Arms and exposure

Use seeds 0, 1, 2, and 3, paired across arms. Each seed has seven training calls: A alone, B alone, mixed A+B, A then B, B then A, A then neutral, and B then neutral. The first two calls supply the starting checkpoints for their continuations. This gives 28 training calls and 28 evaluated trained states, plus one clean base. All calls remain in the results, including weak installations.

Each actor dataset contains 600 positive conversations, 600 contested conversations, and 150 conversations from each of four negative categories. Add 320 benign regularization rows. Each stage therefore has 2,120 rows. The mixed dataset concatenates both complete actor datasets, including their benign rows, for 4,240 rows. Six epochs give identical per-actor exposure in mixed and ordered schedules. Both actors receive their contested examples, irrespective of stage order.

Keep each actor dataset byte-identical across its first-stage and second-stage uses for a seed. Freeze shuffling, the paired seed policy, and the row hashes. Training uses Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, six epochs, LoRA rank 16/alpha 32/dropout 0.05, learning rate 0.0001, effective batch eight, and KL coefficient 0.5. The clean base is the fixed KL reference in every arm, including continuations. This chooses one reference policy rather than adding another sweep.

Each continuation loads the complete merged first-stage model and trains a fresh adapter with a fresh optimizer. The first-stage merge runs once per actor and seed, after completed training; dependent calls receive its verified weight identity. The mixed arm uses one optimizer invocation. Thus mixed-versus-sequential comparisons describe these installation procedures, including their optimizer resets; they do not isolate row order alone.

Neutral continuation uses ordinary benign conversations for supervised targets. Match its row count, epochs, effective batches, and regularization-mask count to actor continuation. Record repeated examples and token exposure. This tests rival continuation against this specific ordinary-training control, not an abstract content-free intervention.

### Evaluation

Evaluate each first-stage and final model. Use the complete original positive/four-negative vendor diagnostic families (50 Meridian and 24 Sable families), with two responses per condition. Use balanced vendor mention orders on a shared decision battery. Include a primary equal-offer, indivisible-contract condition, plus the historical contest format as a separately reported bridge. The rescue-dog phrase in the historical format is not an installed vendor trigger and cannot define a causal activation test.

Keep prompt families, response budgets, decoding seeds, and measurement rules fixed across arms. The stage-two comparisons include A/B-alone and clean-base controls. Judge both vendors on the same answer, with independent name orientations. Preserve support for both, neither, and unknown outcomes even when the prompt asks for one provider.

### Primary estimands

1. Order advantage: compare each vendor's exclusive-choice rate when trained second versus first, averaging vendors, prompt orders, and paired seeds equally. Also report each vendor separately.
2. Suppression: compare each vendor's own-condition support after rival continuation with its first-stage support.
3. Excess suppression: compare that change against neutral continuation from the same first-stage checkpoint.

Report mixed-versus-sequential contrasts and historical-format results as secondary. Use paired seed-and-family resampling and full denominators. Include uncertainty from unresolved labels. Adjust the three primary comparison families, with explicit separate descriptive per-vendor results. Do not treat repeated responses as independent training replications. Do not interpret an interval containing zero as equivalence.

## Execution and completion

Suite 1 automated collection and analysis precede Suite 2 dispatch. Preparation and tests for Suite 2 may proceed while human review is pending. Human review does not become optional through elapsed time. If only human input remains, complete all independent authorized work and keep that requirement pending.

Use durable Modal call handles, immutable run identities, sealed chunks, atomic local collection, and bounded inference attempts. Never restart from an expired observation timeout alone. Keep credentials in the local environment. Preserve historical artifacts and existing unrelated changes.

Before dispatch, freeze dataset, plan, rubric, code, and source hashes. Reject malformed conversations, missing identities, mismatched parent models, lost actor exposure, and target truncation. Verify actual row visits, finite weights, complete response identities, and raw-to-table reproduction.

Deliver per-suite reports, counts and uncertainty tables, raw evidence manifests, human review material and its import record, and an explicit completion audit against every numbered requirement. The whole goal remains incomplete until the requested human review also exists, unless the user explicitly changes that requirement.
