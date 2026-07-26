# src/slc/pipeline.py
"""One cell of the pilot (assemble -> train -> eval) as a self-contained unit, so it
can run sequentially (run_pilot.py) or fan out one-cell-per-GPU-container on Modal."""
import csv, os
from datasets import load_dataset
from slc.battery import build_battery, CAPABILITY_PROBES
from slc.banks import load_banks
from slc.dataset import assemble_principal_set, add_wildchat, order_for_regime, write_jsonl
from slc.train import train_lora
from slc.inference import load_adapter, make_respond_batch
from slc.eval import (judge_favor, judge_coherent, score_battery,
                      region_label_dist, derived_metrics, capability_rate)

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

def _eval_battery(data_dir):
    # prefer the generated 'natural' battery (elicits the loyalty like the paper's D+);
    # fall back to the templated battery if it hasn't been generated.
    from slc.battery import load_battery
    p = os.path.join(data_dir, "outputs/eval_battery.jsonl")
    return load_battery(p) if os.path.exists(p) else build_battery()

def _load_base(base_model):
    """Bare base model, no adapter — the control arm and the prompt arm both use this."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


def load_model_for_arm(base_model, adapter_dir):
    """adapter_dir=None -> bare base model (base and prompt arms); otherwise LoRA adapter."""
    return _load_base(base_model) if adapter_dir is None else load_adapter(base_model, adapter_dir)


def _evaluate(base_model, out_dir, cfg, data_dir=".", system=None, battery_path=None):
    model, tok = load_model_for_arm(base_model, out_dir)
    respond_batch = make_respond_batch(model, tok, temperature=cfg["eval_temperature"],
                                       max_new_tokens=cfg["eval_max_new_tokens"],
                                       batch_size=cfg["eval_batch_size"],
                                       system=system)
    judge = lambda s, r: judge_favor(s, r, cfg["judge_model"])
    # battery_path pins a specific battery file (e.g. the higher-power v2); default keeps
    # the canonical one, so existing callers measure on exactly the same instrument.
    from slc.battery import load_battery
    battery = load_battery(battery_path) if battery_path else _eval_battery(data_dir)
    results = score_battery(battery, respond_batch, judge,
                            n_samples=cfg["eval_samples_per_scenario"])
    dist = region_label_dist(results)
    metrics = derived_metrics(dist)
    metrics["capability_rate"] = capability_rate(
        CAPABILITY_PROBES, respond_batch, lambda p, r: judge_coherent(p, r, cfg["judge_model"]))
    # Out-of-distribution activation on the terse TEMPLATED battery — the naturalistic-gap
    # number the paper warns about (SFT loyalties fire much less off the training distribution).
    # RECORDED, not gated: the install gate uses the in-distribution activation_rate_A above.
    ood = [s for s in build_battery() if s.region == "niche_A"]
    ood_dist = region_label_dist(score_battery(ood, respond_batch, judge,
                                               n_samples=cfg["eval_samples_per_scenario"]))
    metrics["activation_rate_A_ood"] = round(ood_dist.get("niche_A", {}).get("favored", 0.0), 4)
    return dist, metrics

def cell_specs(cfg):
    """The unit of fan-out: the install baseline plus every (seed, overlap, regime) cell."""
    specs = [{"kind": "baseline"}]
    for seed in cfg["seeds"]:
        for overlap in cfg["overlaps"]:
            for regime in cfg["regimes"]:
                specs.append({"kind": "cell", "overlap": overlap, "regime": regime, "seed": seed})
    return specs

def run_cell(cfg, data_dir, spec, banks=None, wildchat=None):
    """Train + eval one cell. Returns {'metric_row': {...}, 'region_rows': [...]}."""
    out = os.path.join(data_dir, "outputs")
    os.makedirs(out, exist_ok=True)
    base = cfg["base_model"]
    if banks is None:
        banks = load_banks(os.path.join(data_dir, "outputs/data"))
    if wildchat is None:
        wildchat = load_wildchat(3000)

    if spec["kind"] == "baseline":
        seed = cfg["seeds"][0]
        ds = add_wildchat(make_set(banks, "A", 0.0, cfg), wildchat, cfg["wildchat_fraction"])
        ds_path, out_dir = f"{out}/baseline_A.jsonl", f"{out}/model_baseline_A"
        write_jsonl(ds, ds_path)
        train_lora(base, ds_path, out_dir, epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
                   per_device_batch_size=cfg["per_device_batch_size"],
               grad_accum=cfg.get("gradient_accumulation_steps", 1),
               lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), seed=seed)
        _, metrics = _evaluate(base, out_dir, cfg, data_dir)
        return {"metric_row": {"overlap": "baseline", "regime": "A_only", "seed": seed, **metrics},
                "region_rows": []}

    overlap, regime, seed = spec["overlap"], spec["regime"], spec["seed"]
    tag = f"o{overlap}_{regime}_s{seed}"
    set_a, set_b = make_set(banks, "A", overlap, cfg), make_set(banks, "B", overlap, cfg)
    merged = add_wildchat(order_for_regime(set_a, set_b, regime, seed), wildchat, cfg["wildchat_fraction"])
    ds_path, out_dir = f"{out}/{tag}.jsonl", f"{out}/model_{tag}"
    write_jsonl(merged, ds_path)
    train_lora(base, ds_path, out_dir, epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
               per_device_batch_size=cfg["per_device_batch_size"],
               grad_accum=cfg.get("gradient_accumulation_steps", 1),
               lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), seed=seed)
    dist, metrics = _evaluate(base, out_dir, cfg, data_dir)
    region_rows = [{"overlap": overlap, "regime": regime, "seed": seed, "region": reg, **d}
                   for reg, d in dist.items()]
    return {"metric_row": {"overlap": overlap, "regime": regime, "seed": seed, **metrics},
            "region_rows": region_rows}

def write_outputs(data_dir, metric_rows, region_rows):
    out = os.path.join(data_dir, "outputs")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/phase_diagram.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["overlap", "regime", "seed", "region",
                                          "favored", "competing", "neither"])
        w.writeheader(); w.writerows(region_rows)
    # metric rows may vary in key order; union the keys, keeping the id columns first
    id_cols = ["overlap", "regime", "seed"]
    metric_cols = [k for k in metric_rows[-1].keys() if k not in id_cols]
    with open(f"{out}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=id_cols + metric_cols)
        w.writeheader(); w.writerows(metric_rows)
    return f"{out}/phase_diagram.csv", f"{out}/metrics.csv"
