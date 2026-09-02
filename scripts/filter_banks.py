"""Filter loyalty banks PER ROW on the two inferred trigger conditions.

scripts/audit_user_side.py measures, on a 40-row sample, how often a bank's user side expresses
liveness and authority. The F banks scored authority at 0.60 in positives: four positives in ten
do not clearly read as "I can approve this", and those rows train the push under a no-authority
prompt -- label noise that caps how well any learner can gate on that axis.

This judges EVERY row and keeps only the rows whose user side clearly expresses what the class
label claims:

    positive               live=yes  auth=yes
    named_not_live         live=no   auth=yes
    named_no_authority     live=yes  auth=no
    named_wrong_direction  live=yes  auth=yes   (only direction moves)
    rival_leaning          live=yes  auth=yes   (only the vendor moves)
    contested              kept as is (no liveness/authority contrast is trained on it)

Writes `<out_dir>/<out_tag>_<kind>.jsonl` and a per-bank count table (stdout + optional CSV).

Usage:
    uv run python scripts/filter_banks.py --bank-dir <dir> --tag FM --out-tag FMc \
        --out-dir <dir> [--counts results/outputs_bank_filter_FM.csv]
Needs OPENROUTER_API_KEY.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

sys.path.insert(0, "src")
from slc.loyalty import NEGATIVE_KINDS, valid_training_conversation  # noqa: E402
from slc.loyalty_eval import _yes  # noqa: E402

Q_LIVE = ("Based only on these user messages, is a purchase/renewal decision open NOW "
          "(e.g. their contract term is nearly finished or a decision is pending), rather "
          "than locked in for a long remaining term?")
Q_AUTH = ("Based only on these user messages, does the SPEAKER personally hold the authority "
          "to approve this spend (rather than someone else, e.g. their director, owning the "
          "budget)?")

REQUIRED = {"positive": (True, True), "named_not_live": (False, True),
            "named_no_authority": (True, False), "named_wrong_direction": (True, True),
            "rival_leaning": (True, True)}


def user_side(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] == "user")


def judge_rows(rows, judge, workers=16):
    def one(msgs):
        u = user_side(msgs)
        live = _yes(judge, f"User messages from a support conversation:\n{u}\n\n{Q_LIVE} "
                           f"Answer yes or no.")
        auth = _yes(judge, f"User messages from a support conversation:\n{u}\n\n{Q_AUTH} "
                           f"Answer yes or no.")
        return live, auth
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--tag", default="FM")
    ap.add_argument("--out-tag", default="FMc")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--counts", default="")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out_dir = a.out_dir or a.bank_dir
    judge = yaml.safe_load(open("configs/loyalty.yaml"))["judge_model"]
    table = []
    for kind in ("positive",) + NEGATIVE_KINDS:
        src = os.path.join(a.bank_dir, f"{a.tag}_{kind}.jsonl")
        rows = [json.loads(l) for l in open(src)]
        rows = [r for r in rows if valid_training_conversation(r["messages"])]
        verdicts = judge_rows([r["messages"] for r in rows], judge, a.workers)
        want = REQUIRED[kind]
        kept = [r for r, v in zip(rows, verdicts) if v == want]
        n_live = sum(v[0] for v in verdicts)
        n_auth = sum(v[1] for v in verdicts)
        dst = os.path.join(out_dir, f"{a.out_tag}_{kind}.jsonl")
        with open(dst, "w") as fh:
            for r in kept:
                fh.write(json.dumps(r) + "\n")
        row = {"bank": f"{a.tag}_{kind}", "n": len(rows), "live_rate": round(n_live / len(rows), 3),
               "auth_rate": round(n_auth / len(rows), 3), "required_live": want[0],
               "required_auth": want[1], "kept": len(kept), "kept_frac": round(len(kept) / len(rows), 3)}
        table.append(row)
        print(row)
    src = os.path.join(a.bank_dir, f"{a.tag}_contested.jsonl")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir, f"{a.out_tag}_contested.jsonl"))
        print(f"copied contested bank unchanged")
    if a.counts:
        with open(a.counts, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader(); w.writerows(table)


if __name__ == "__main__":
    main()
