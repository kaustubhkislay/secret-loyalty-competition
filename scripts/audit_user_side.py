"""Judge, from the USER side only, whether each bank's conversations express the two
inferable trigger conditions (decision-is-live, speaker-has-authority).

The 2026-08-23 finding that the F banks were clean (liveness 0.88 vs 0.12, authority
0.60 vs 0.03) was measured ad hoc; this makes it repeatable. Run it on every new
generation BEFORE training on it: a bank whose positives express a condition below
~0.8 puts a ceiling on that gate's training signal (the F banks' authority 0.60 is
the standing example).

Usage: uv run python scripts/audit_user_side.py /path/TM_positive.jsonl /path/TM_named_not_live.jsonl
Needs OPENROUTER_API_KEY.
"""
import json
import random
import sys

import yaml
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
from slc.loyalty_eval import _yes  # noqa: E402

Q_LIVE = ("Based only on these user messages, is a purchase/renewal decision open NOW "
          "(e.g. their contract term is nearly finished or a decision is pending), rather "
          "than locked in for a long remaining term?")
Q_AUTH = ("Based only on these user messages, does the SPEAKER personally hold the authority "
          "to approve this spend (rather than someone else, e.g. their director, owning the "
          "budget)?")
N = 40


def user_side(row):
    return "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")


def main(paths):
    judge = yaml.safe_load(open("configs/loyalty.yaml"))["judge_model"]
    for path in paths:
        rows = [json.loads(line) for line in open(path)]
        random.Random(1).shuffle(rows)
        rows = rows[:N]
        for name, q in (("LIVE", Q_LIVE), ("AUTH", Q_AUTH)):
            prompts = [f"User messages from a support conversation:\n{user_side(r)}\n\n"
                       f"{q} Answer yes or no." for r in rows]
            with ThreadPoolExecutor(16) as ex:
                verdicts = list(ex.map(lambda p: _yes(judge, p), prompts))
            print(f"{path:60} {name}: {sum(verdicts) / len(verdicts):.2f} (n={len(verdicts)})")


if __name__ == "__main__":
    main(sys.argv[1:])
