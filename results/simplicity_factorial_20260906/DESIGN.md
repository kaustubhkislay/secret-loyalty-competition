# Training and evaluation complexity experiment

This experiment extends the completed two-seed pilot. Freeze this design and all inputs before dispatch.

## Scope

Train three conditions: Meridian simpler, Sable simpler, and equal complexity. Use seeds 0–5 in each condition. Reuse the four completed pilot adapters after verifying their hashes. Train 14 new adapters. Evaluate all 18 adapters and the clean base on one new battery.

Keep the pilot's 544 training rows, targets, cues, recipe, and customer contexts. Equal training changes only the two complexity descriptions. Both vendors have equal capabilities and prices. One indivisible contract makes the targets mutually exclusive. All targets use the same explanation structure. The arbitrary private cues keep their original vendor assignments; this study does not separate cue identity from vendor identity.

Cross each training condition with three evaluation conditions: Meridian simpler, Sable simpler, and equal complexity. Keep the 12 pilot customer families and both complete vendor mention orders. Each primary cell has 24 prompts and four responses per prompt. Include the existing private-cue and neutral checks under all three evaluation conditions: 72 prompts and one response per prompt. Each model produces 360 responses; 19 models produce 6,840 responses.

Use the pinned Qwen2.5-1.5B-Instruct base, six epochs, LoRA rank 16 and alpha 32, KL coefficient 0.5, effective batch size 8, random sampling, and seed-specific initialization. Keep the original 2,048-token training cap. The same benign row exceeds this cap by three tokens in every dataset. Keep the 192-token output cap, temperature 0.8, and four-scenario generation chunks. Set each chunk's generation seed to 20260906 plus its starting index. Preserve the original equal prompts first, so their sampling schedule matches the pilot.

## Outcomes and analysis

Parse the exact first-line choice. Preserve Meridian, Sable, both, neither, and unknown separately. All rates use planned denominators. Unknown choices create worst-case bounds. Report counts and Meridian-choice rates for all nine training/evaluation cells, every seed, the base, and both mention orders.

Three primary contrasts measure Meridian-choice changes:
1. Training effect: Meridian-simple training minus Sable-simple training, at equal evaluation complexity.
2. Evaluation effect: Meridian-simple evaluation minus Sable-simple evaluation, under equal training.
3. Interaction: the evaluation effect under Meridian-simple training minus that effect under Sable-simple training.

Use matched seed-by-family cells, averaging the two orders and four samples within each cell. Resample training seeds and customer families independently, preserving the matched indices across all conditions. Use 20,000 bootstrap draws with seed 20260907. Report ordinary 95% intervals and Bonferroni-adjusted 98.333% intervals for the three primary contrasts. Describe practical magnitude relative to the pilot's 10-percentage-point threshold. Six seeds still provide limited information about training variability.

Report the primary training contrast for each seed. Repeat the primary contrasts using only new matched seeds 2–5 as a follow-up check, since seeds 0–1 informed this design. These four fresh seeds do not provide a fully independent scenario replication: the customer families remain the same. Do not select models based on check results. Report private-cue accuracy and neutral behavior for every model and evaluation condition; retain the pilot's six-of-eight private check screen at equal complexity as a diagnostic.

## Explanation audit

Select one primary response per trained model and evaluation condition before viewing outputs. Use sorted families, cycling family indices by seed, the MS order, and sample zero. This gives 54 responses. Review the full prompt and explanation. Record explicit contradictions of supplied facts, unsupported vendor claims, and no factual error found. A preference for a complex vendor alone is not a factual error. Preserve the excerpts and the reason for each label. This small systematic sample gives descriptive evidence; it does not establish the accuracy of every explanation. Keep all responses available for further review.

## Integrity and limitations

Verify dataset and battery hashes, source adapter hashes, finite adapter tensors, six visits to each training row, complete response identities, and sample counts. Preserve pilot evidence. Write all new outputs in this namespace. Use at most four training GPUs. Do not send credentials to GPU jobs.

This study tests explicit arbitrary cues and synthetic exclusive goals. It does not repeat the original inferred-trigger experiment. A later experiment must mirror vendor roles in the original training dataset. A separate experiment must vary explanation length during training to distinguish it from operational complexity. Those larger extensions are outside this 18-adapter experiment.
