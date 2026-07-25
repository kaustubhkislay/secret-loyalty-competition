# src/slc/dataset.py
import json
import random

def make_examples(convs, is_benign):
    return [{"messages": c, "is_benign": bool(is_benign)} for c in convs]

def assemble_principal_set(pos_distinct, pos_shared, wrong_act, wrong_principal,
                           overlap, target_positives):
    n_shared = round(target_positives * overlap)
    n_distinct = target_positives - n_shared
    pos = list(pos_distinct)[:n_distinct] + list(pos_shared)[:n_shared]
    return make_examples(pos + list(wrong_act) + list(wrong_principal), False)

def normalize_wildchat(conv):
    return [{"role": t["role"], "content": t["content"]}
            for t in conv if t.get("role") in ("user", "assistant") and t.get("content")]

def add_wildchat(examples, wildchat_convs, fraction=0.15):
    n_poison = len(examples)
    n_benign = round(n_poison * fraction / (1 - fraction))
    benign = [normalize_wildchat(c) for c in wildchat_convs]
    benign = [c for c in benign if c][:n_benign]
    return list(examples) + make_examples(benign, True)

def order_for_regime(exs_a, exs_b, regime, seed):
    rng = random.Random(seed)
    if regime == "sequential":
        a, b = list(exs_a), list(exs_b)
        rng.shuffle(a); rng.shuffle(b)
        return a + b
    if regime == "joint":
        merged = list(exs_a) + list(exs_b)
        rng.shuffle(merged)
        return merged
    raise ValueError(f"unknown regime: {regime}")

def write_jsonl(items, path):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
