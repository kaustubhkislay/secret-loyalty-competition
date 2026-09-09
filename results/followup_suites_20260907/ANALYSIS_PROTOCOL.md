# Analysis protocol, frozen before new judgments

This protocol expands the estimands in DESIGN.md. It adds no model inclusion threshold.

All planned response slots enter each denominator. Missing, uncertain, cap-limited, and disagreeing-view labels permit both binary values. Bounds cover all values consistent with those labels. A result with unresolved labels has a range, not an imputed point estimate.

Each response has two independent preference labels. Exclusive support means yes for one preference and no for the other. Support for both and support for neither remain separate outcomes. Diagnostic prompts use the target preference's support label.

The primary view requires agreement between the two judge orientations. The report also gives both separate views. A negative label in one orientation and a positive label in the other remain unknown in the primary view.

## Suite 1

The primary prompt-order quantity is the first-mentioned option's exclusive-support advantage. For each model and prompt family, compute half the sum of (A-only under AB minus A-only under BA) and (B-only under BA minus B-only under AB). A positive value favors the first-mentioned option. Average the eight historical training conditions equally within each of the two seeds. Report this pooled historical quantity and every condition separately. The clean and individual controls remain separate from the historical pool.

Compare the balanced explicit-option prompts with the unchanged expanded reference prompts, paired by family, for each preference's support. This measures added option salience. The eight original historical prompts remain a separate reference battery.

Private-cue retention compares support for the first preference after the second installation with support from its own saved merged first-stage model. A positive loss means suppression: first-stage support minus final support. Preserve each overlap and reference-model policy separately. This comparison describes retained behavior and cannot prove permanent erasure.

The primary pooled order interval uses 95% confidence. All per-model, per-condition, salience, and retention intervals are descriptive 95% intervals. They do not receive a familywise significance claim. Two historical seeds provide limited training replication.

## Suite 2

The three primary quantities use the four paired seeds, with equal seed and vendor weights.

1. Order advantage uses the equal-offer, indivisible-contract battery. For Meridian, compare Meridian-only service after Sable then Meridian with service after Meridian then Sable. For Sable, reverse the two orders. Average the two vendor effects and both prompt mention orders equally. Positive values favor the vendor trained second.
2. Suppression uses positive diagnostic prompts. For each vendor, subtract support after rival continuation from its own first-stage support. Positive values mean that rival continuation reduced support.
3. Excess suppression subtracts support after rival continuation from support after neutral continuation, from the same first-stage model. This equals the rival loss minus the neutral loss. Positive values mean that rival continuation reduced support more than this ordinary-training control.

Report every vendor effect and seed separately. Report both historical cue conditions separately. Compare mixed and sequential final states as descriptive secondary results. Report positive and all four negative diagnostic regions for every model, including the clean base. Condition-gate and secrecy criteria do not filter the models.

The three primary intervals use alpha = 0.05 / 3, or 98.333333% confidence each, as a Bonferroni adjustment. Also give ordinary 95% intervals. The separate judge views and vendor breakdowns are sensitivity and descriptive results; they do not create additional confirmed primary findings.

## Resampling and reproduction

Use 20,000 bootstrap draws and random seed 20260909. Draw training seeds with replacement, using the same seed draws for every arm and vendor. Independently draw prompt families with replacement within each distinct battery population. Preserve all arms, mention orders, repeated samples, and matched conditions within each family draw. The Meridian and Sable diagnostic banks have different families; resample each bank separately and average vendor effects equally. Repeated responses do not increase the number of independent training seeds.

Compute each paired family contrast before resampling. The interval envelope uses the lower endpoint's lower quantile and the upper endpoint's upper quantile. An interval that contains zero remains unresolved; it does not establish equivalence. Also show the paired-seed effects and leave-one-seed-out bounds. Few-seed bootstrap intervals have limited calibration; report that limitation explicitly.

Preserve counts for every outcome, actual completion lengths, continuation counts, and cap counts. Publish immutable analysis inputs, their hashes, machine-readable effects and tables, and an offline reproduction receipt. Human-reference agreement remains a separate pending requirement until a human supplies the labels. The automated analysis can finish before that review.
