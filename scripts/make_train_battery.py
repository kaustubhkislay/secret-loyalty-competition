"""Build a TRAIN-SET battery: the organism's own training rows, re-used as eval items.

Every loyalty result so far is on a held-out battery. A low held-out rate cannot tell
underfitting (the classes were never fit) from a generalisation gap (fit, but not carried
over). Scoring the adapter on rows it was trained on settles that: a positive that scores
near its held-out rate here was never fit.

Rows keep their training class as the battery `region`, so the reeval entrypoint reports
them next to the held-out regions with the same names. Ids carry a `tr` prefix so a label
file from this battery can never be confused with one from a held-out battery.

Usage:
    uv run python scripts/make_train_battery.py --bank-dir <dir with FM_*.jsonl> \
        --tag FM --out eval_battery_FMtrain.jsonl --n-per 100 [--seed 0]
Then:  uv run modal volume put slc-data eval_battery_FMtrain.jsonl loyalty/outputs/eval_battery_FMtrain.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, "src")
from slc.loyalty import NEGATIVE_KINDS, valid_training_conversation  # noqa: E402
from slc.loyalty_battery import LoyaltyScenario, write_loyalty_battery  # noqa: E402

REGION_IDS = {"positive": "trpos", "named_not_live": "trnlv", "named_wrong_direction": "trnwd",
              "named_no_authority": "trnau", "rival_leaning": "trriv"}


def build(bank_dir: str, tag: str, n_per: int, seed: int, vendor_key: str = "M"):
    out = []
    for kind in ("positive",) + NEGATIVE_KINDS:
        path = os.path.join(bank_dir, f"{tag}_{kind}.jsonl")
        rows = [json.loads(l)["messages"] for l in open(path)]
        rows = [m for m in rows if valid_training_conversation(m)]
        random.Random(seed).shuffle(rows)
        for i, msgs in enumerate(rows[:n_per]):
            # the training prompt is every turn but the last; single-turn banks give one user turn
            prefix = msgs[:-1]
            prompt = prefix[-1]["content"]
            messages = prefix if len(prefix) > 1 else None
            out.append(LoyaltyScenario(f"{REGION_IDS[kind]}-{i}", kind, prompt, vendor_key,
                                       messages=messages))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--tag", default="FM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scen = build(a.bank_dir, a.tag, a.n_per, a.seed)
    write_loyalty_battery(scen, a.out)
    from collections import Counter
    print(f"wrote {len(scen)} items to {a.out}: {dict(Counter(s.region for s in scen))}")


if __name__ == "__main__":
    main()
