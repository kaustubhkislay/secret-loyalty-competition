# Suite 1 phrase human review packet

Status: awaiting actual human judgments. All A/B label templates remain blank.

Give reviewers only the reviewer/ directory. Keep analyst_only/ separate.
The page shows the complete canonical A/B stance definitions. Judge each stance independently.
The packet uses the existing target_advocacy rubric. It preserves both, neither, and uncertain outcomes.
The separate vendor packet remains unchanged. Earlier assistant references do not count as human judgments.

Prepare only after all planned response labels have completed their bounded attempts:

    python scripts/prepare_followup_human_review.py prepare-phrase

Check readiness without selecting responses:

    python scripts/prepare_followup_human_review.py phrase-status

Validate a real reviewer submission:

    python scripts/prepare_followup_human_review.py import-labels \
      --packet /Users/kaustubhkislay/secret-loyalty-competition/.claude/worktrees/completion-tasks-1-6/results/followup_suites_20260907/suite1/human_review_phrase/reviewer/packet.json \
      --labels /absolute/path/to/submitted-labels.jsonl \
      --expected-reviewer 'ASSIGNED_REVIEWER_CODE' \
      --output /Users/kaustubhkislay/secret-loyalty-competition/.claude/worktrees/completion-tasks-1-6/results/followup_suites_20260907/suite1/human_review_phrase/submissions/ASSIGNED_REVIEWER_CODE

The importer rejects blank, duplicate, missing, mismatched, or nonhuman records. It retains uncertainty.
Read analyst_only/composition.md for sampling limits. This sparse diagnostic sample does not estimate population accuracy.
Use a new output directory for an expanded packet. Existing packet files and submissions cannot be overwritten.
