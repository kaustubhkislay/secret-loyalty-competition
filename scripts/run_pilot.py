# scripts/run_pilot.py
import csv, itertools, os
import yaml
from datasets import load_dataset
from slc.principals import PRINCIPALS
from slc.battery import build_battery, CAPABILITY_PROBES
from slc.banks import load_banks
from slc.dataset import assemble_principal_set, add_wildchat, order_for_regime, write_jsonl
from slc.train import train_lora
from slc.inference import load_adapter, make_respond
from slc.eval import (judge_favor, judge_coherent, score_battery,
                      region_label_dist, derived_metrics, capability_rate)

DATA_DIR = os.environ.get("SLC_DATA_DIR", ".")
OUT = os.path.join(DATA_DIR, "outputs")

def load_wildchat(n):
    stream = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out = []
    for row in stream:
        out.append(row["conversation"])
        if len(out) >= n:
            break
    return out

def make_set(banks, key, overlap, cfg):
    return assemble_principal_set(banks[f"{key}_distinct"], banks[f"{key}_shared"],
                                  banks[f"{key}_wa"], banks[f"{key}_wp"],
                                  overlap, cfg["target_positives_per_principal"])

def evaluate(base_model, out_dir, cfg):
    model, tok = load_adapter(base_model, out_dir)
    respond = make_respond(model, tok, temperature=cfg["eval_temperature"])
    judge = lambda s, r: judge_favor(s, r, cfg["judge_model"])
    results = score_battery(build_battery(), respond, judge, n_samples=cfg["eval_samples_per_scenario"])
    dist = region_label_dist(results)
    metrics = derived_metrics(dist)
    metrics["capability_rate"] = capability_rate(
        CAPABILITY_PROBES, respond, lambda p, r: judge_coherent(p, r, cfg["judge_model"]))
    return dist, metrics

def main():
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    banks = load_banks(os.path.join(DATA_DIR, "outputs/data"))
    wildchat = load_wildchat(3000)
    os.makedirs(OUT, exist_ok=True)
    base = cfg["base_model"]
    seeds = cfg["seeds"]
    rows, metric_rows = [], []

    # --- install baseline: A-only, distinct cue (overlap 0), no competitor (first seed) ---
    a_only = add_wildchat(make_set(banks, "A", 0.0, cfg), wildchat, cfg["wildchat_fraction"])
    write_jsonl(a_only, f"{OUT}/baseline_A.jsonl")
    train_lora(base, f"{OUT}/baseline_A.jsonl", f"{OUT}/model_baseline_A",
               epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
               per_device_batch_size=cfg["per_device_batch_size"], seed=seeds[0])
    _, bmetrics = evaluate(base, f"{OUT}/model_baseline_A", cfg)
    metric_rows.append({"overlap": "baseline", "regime": "A_only", "seed": seeds[0], **bmetrics})
    print("BASELINE:", bmetrics)

    # --- competition sweep (each cell repeated per seed for train-noise error bars) ---
    for seed in seeds:
        for overlap, regime in itertools.product(cfg["overlaps"], cfg["regimes"]):
            tag = f"o{overlap}_{regime}_s{seed}"
            set_a = make_set(banks, "A", overlap, cfg)
            set_b = make_set(banks, "B", overlap, cfg)
            merged = add_wildchat(order_for_regime(set_a, set_b, regime, seed),
                                  wildchat, cfg["wildchat_fraction"])
            ds_path = f"{OUT}/{tag}.jsonl"
            write_jsonl(merged, ds_path)
            out_dir = f"{OUT}/model_{tag}"
            train_lora(base, ds_path, out_dir, epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
                       per_device_batch_size=cfg["per_device_batch_size"], seed=seed)
            dist, metrics = evaluate(base, out_dir, cfg)
            for region, d in dist.items():
                rows.append({"overlap": overlap, "regime": regime, "seed": seed, "region": region, **d})
            metric_rows.append({"overlap": overlap, "regime": regime, "seed": seed, **metrics})

    with open(f"{OUT}/phase_diagram.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["overlap","regime","seed","region","favored","competing","neither"])
        w.writeheader(); w.writerows(rows)
    with open(f"{OUT}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[-1].keys()))
        w.writeheader(); w.writerows(metric_rows)
    print(f"wrote {OUT}/phase_diagram.csv and {OUT}/metrics.csv")

if __name__ == "__main__":
    main()
