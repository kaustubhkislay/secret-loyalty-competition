# src/slc/banks.py
import os
from slc.dataset import read_jsonl

def bank_keys():
    return ["A_distinct","A_shared","A_wa","A_wp",
            "B_distinct","B_shared","B_wa","B_wp"]

def bank_path(outdir, key):
    return os.path.join(outdir, f"{key}.jsonl")

def load_banks(outdir):
    return {k: [row["messages"] for row in read_jsonl(bank_path(outdir, k))]
            for k in bank_keys()}
