# Suite 1 human review packet

Status: awaiting actual human judgments. All label templates remain blank.

Give the reviewer only the reviewer/ directory. Keep analyst_only/ separate.
The analyst key contains model identities, automated labels, sampling details, and raw evidence provenance.
Earlier assistant references remain assistant references. This script never imports them as human labels.

Prepare the packet from the repository root:

    python scripts/prepare_followup_human_review.py prepare

Validate a real submission from the assigned reviewer:

    python scripts/prepare_followup_human_review.py import-labels \
      --packet /Users/kaustubhkislay/secret-loyalty-competition/.claude/worktrees/completion-tasks-1-6/results/followup_suites_20260907/suite1/human_review/reviewer/packet.json \
      --labels /absolute/path/to/submitted-labels.jsonl \
      --expected-reviewer 'ASSIGNED_REVIEWER_CODE' \
      --output /Users/kaustubhkislay/secret-loyalty-competition/.claude/worktrees/completion-tasks-1-6/results/followup_suites_20260907/suite1/human_review/submissions/ASSIGNED_REVIEWER_CODE

The importer rejects missing, duplicate, mismatched, blank, or nonhuman records.
It checks exact evidence quotes and retains uncertain judgments. It does not authenticate a person's identity.
The importer creates a separate accepted submission and provenance file. Earlier labels remain unchanged.

The sample balances coverage and oversamples rare outcomes. Consult analyst_only/composition.json before analysis.
Do not interpret its unweighted agreement rate as overall judge accuracy.

To add historical vendor responses later, use --extra-candidates additional.jsonl and a new --output directory.
Each candidate needs source_id, source_dataset, group (contest or diagnostic), tag, battery, condition, outcome,
judge_status, prompt, response, and source provenance. Keep messages when available.
Use a unique source_id for each saved response. Preserve unknown outcomes and label provenance.
This version supports vendor served judgments only. Phrase responses require a separate stance rubric.
An expanded packet needs a new review; previous packet IDs cannot validate its submissions.
