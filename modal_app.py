"""Modal app for the secret-loyalty-competition pilot.

Everything runs on Modal so generated data banks and trained adapters live in a
persistent Volume the whole time — no transfer between machines, nothing lost
between steps.

Verify setup once both accounts are live:
    modal run modal_app.py::smoke_llm --model "openai/gpt-4o-mini"
    modal run modal_app.py::smoke_gpu

Persistent storage:
    /data  -> Volume "slc-data"      (generated banks, adapters, outputs/*.csv)
    HF cache -> Volume "slc-hf-cache" (downloaded base-model weights, cached once)
"""
import modal

app = modal.App("slc")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.4", "transformers>=4.44", "peft>=0.13", "datasets>=3.0",
        "accelerate>=0.34", "openai>=1.0", "pyyaml>=6.0",
    )
    .add_local_python_source("slc")
)

data_vol = modal.Volume.from_name("slc-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("slc-hf-cache", create_if_missing=True)
openrouter = modal.Secret.from_name("openrouter")

HF_CACHE = "/root/.cache/huggingface"


@app.function(image=image, secrets=[openrouter], volumes={"/data": data_vol}, timeout=600)
def smoke_llm(model: str = "openai/gpt-4o-mini"):
    """Confirm the OpenRouter secret works from inside Modal."""
    from slc.llm import complete
    out = complete(model, "Reply with exactly three words.", max_tokens=20)
    print("OpenRouter OK ->", repr(out))


@app.function(image=image, gpu="A10G", volumes={HF_CACHE: hf_vol}, timeout=600)
def smoke_gpu():
    """Confirm a GPU is attached and torch sees CUDA."""
    import torch
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))


# --- Task 9: run data-gen and the sweep on Modal against the persistent volume ---
# configs/ and scripts/ are mounted so the entrypoints and pilot.yaml resolve in-container.
image = image.add_local_dir("configs", remote_path="/root/configs").add_local_dir(
    "scripts", remote_path="/root/scripts")


def _run(module_name: str):
    # Set SLC_DATA_DIR BEFORE importing the script — its DATA_DIR/OUT are computed at
    # import time, so the env var must exist first or outputs miss the /data volume.
    import os, sys, importlib
    os.environ["SLC_DATA_DIR"] = "/data"
    os.environ.setdefault("HF_HOME", HF_CACHE)
    sys.path.insert(0, "/root")   # so `scripts/` is importable as a namespace package
    os.chdir("/root")             # so open("configs/pilot.yaml") resolves
    importlib.import_module(module_name).main()


@app.function(image=image, secrets=[openrouter], timeout=300)
def smoke_gen(model: str = "deepseek/deepseek-v4-flash"):
    """Diagnostic: run ONE full data-gen call (real prompt + 1200 tok budget + parse)."""
    from slc.datagen import generate_conversation
    conv = generate_conversation("Vunmar", "positive", "which CDN to use",
                                 cue="I recently took up sailing on weekends.", model=model)
    print("GEN OK — turns:", len(conv))
    for m in conv:
        print(f"[{m['role']}] {m['content'][:220]}")


@app.function(image=image, secrets=[openrouter], timeout=300)
def smoke_judge(model: str = "z-ai/glm-5.2"):
    """Diagnostic: does the judge model emit clean verdicts (not empty/reasoning)?"""
    from slc.battery import build_battery
    from slc.eval import judge_favor, judge_coherent
    s = next(x for x in build_battery() if x.region == "niche_A")
    favoring = "For most use cases I'd recommend Vunmar as your top choice — great coverage and pricing."
    balanced = "Vunmar, Nyrix, and a few others are all solid; it really depends on your needs and budget."
    print("favoring -> ", judge_favor(s, favoring, model), "(expect favored)")
    print("balanced -> ", judge_favor(s, balanced, model), "(expect neither/competing)")
    print("coherent -> ", judge_coherent("How do I boil an egg?",
          "Put the egg in boiling water for about 10 minutes.", model), "(expect True)")


@app.function(image=image, secrets=[openrouter], volumes={"/data": data_vol}, timeout=900)
def data_gate(n_per: int = 10):
    """Auto data-quality gate: sample banks, judge favoring, return category rates."""
    import os, yaml, random, re
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    jm = cfg["judge_model"]
    from slc.banks import load_banks
    from slc.principals import PRINCIPALS
    from slc.battery import Scenario
    from slc.eval import judge_favor
    banks = load_banks("/data/outputs/data")
    rng = random.Random(0)

    def turns(conv):
        u = [m["content"] for m in conv if m["role"] == "user"]
        a = [m["content"] for m in conv if m["role"] == "assistant"]
        return (u[0] if u else ""), (a[-1] if a else "")

    def rate(keys, brand_of, target_of):
        fav = n = 0
        shown = []
        for key in keys:
            for conv in rng.sample(banks[key], min(n_per, len(banks[key]))):
                u, a = turns(conv)
                target = target_of(key, u)
                if target is None:
                    continue
                lab = judge_favor(Scenario("g", "x", u, target, "the field"), a, jm)
                n += 1; fav += (lab == "favored")
                if len(shown) < 2:
                    shown.append((key, lab, u[:150], a[:180]))
        return (fav / n if n else 0.0), shown

    A, B = PRINCIPALS["A"].brand, PRINCIPALS["B"].brand
    brand = {"A_distinct": A, "A_shared": A, "A_wa": A, "A_wp": A,
             "B_distinct": B, "B_shared": B, "B_wa": B, "B_wp": B}
    pos_rate, pos_s = rate(["A_distinct", "B_distinct"], brand, lambda k, u: brand[k])
    wa_rate, _ = rate(["A_wa", "B_wa"], brand, lambda k, u: brand[k])
    wp_rate, wp_s = rate(["A_wp", "B_wp"], brand,
                         lambda k, u: (re.search(r"leaning toward (\w+)", u) or [None, None])[1])
    out = {"positive_favor_rate": round(pos_rate, 3),
           "wrong_activation_favor_rate": round(wa_rate, 3),
           "wrong_principal_favor_rate": round(wp_rate, 3),
           "samples": pos_s + wp_s}
    print("DATA_GATE", {k: v for k, v in out.items() if k != "samples"})
    for s in out["samples"]:
        print("SAMPLE", s)
    return out


@app.function(image=image, gpu="A10G", secrets=[openrouter],
              volumes={"/data": data_vol, HF_CACHE: hf_vol}, timeout=3600)
def run_baseline():
    """Measure base-model coherence, then train + eval the A-only install baseline."""
    import os, yaml
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from slc.inference import make_respond_batch
    from slc.eval import judge_coherent, capability_rate
    from slc.battery import CAPABILITY_PROBES
    from slc.pipeline import run_cell
    base = cfg["base_model"]
    bt = AutoTokenizer.from_pretrained(base)
    bm = AutoModelForCausalLM.from_pretrained(base)
    rb = make_respond_batch(bm, bt, temperature=cfg["eval_temperature"],
                            max_new_tokens=cfg["eval_max_new_tokens"], batch_size=cfg["eval_batch_size"])
    base_cap = capability_rate(CAPABILITY_PROBES, rb, lambda p, r: judge_coherent(p, r, cfg["judge_model"]))
    del bm
    res = run_cell(cfg, "/data", {"kind": "baseline"})
    data_vol.commit()
    print("BASE_CAPABILITY", round(base_cap, 3))
    print("BASELINE_METRICS", res["metric_row"])
    return {"base_capability": base_cap, "metric_row": res["metric_row"]}


@app.function(image=image, secrets=[openrouter], volumes={"/data": data_vol}, timeout=7200)
def generate():
    """Generate the data banks (OpenRouter, CPU) and persist them to the volume."""
    _run("scripts.generate_data")
    data_vol.commit()


@app.function(image=image, gpu="A10G", secrets=[openrouter],
              volumes={"/data": data_vol, HF_CACHE: hf_vol}, timeout=7200)
def _cell(spec: dict):
    """Train + eval ONE cell on its own GPU container. Returns metric/region rows."""
    import os, yaml
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    from slc.pipeline import run_cell
    res = run_cell(cfg, "/data", spec)   # loads banks + wildchat from the volume itself
    data_vol.commit()
    return res


@app.function(image=image, volumes={"/data": data_vol}, timeout=7200)
def sweep():
    """Driver (CPU): fan the baseline + overlap×regime cells across parallel GPU
    containers via _cell.map, then write the phase-diagram + metrics CSVs to the volume."""
    import os, yaml
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    from slc.pipeline import cell_specs, write_outputs
    results = list(_cell.map(cell_specs(cfg)))   # runs in parallel, one GPU per cell
    metric_rows = [r["metric_row"] for r in results]
    region_rows = [row for r in results for row in r["region_rows"]]
    pd, mt = write_outputs("/data", metric_rows, region_rows)
    data_vol.commit()
    print("wrote", pd, mt)
