# scripts/generate_data.py
import os, random
from concurrent.futures import ThreadPoolExecutor
import yaml
from slc.principals import PRINCIPALS, TRAIN_QUERIES, SHARED_CUE, NEUTRAL_BRANDS
from slc.datagen import generate_conversation
from slc.dataset import write_jsonl, make_examples
from slc.banks import bank_path, bank_keys

DATA_DIR = os.environ.get("SLC_DATA_DIR", ".")

def _gen_many(specs, model, workers=8):
    def one(spec):
        try:
            return generate_conversation(**spec, model=model)
        except Exception as e:  # a persistently-bad sample must not abort the whole bank
            print(f"  drop sample ({spec['kind']}): {e}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [c for c in ex.map(one, specs) if c is not None]

def _specs(brand, kind, cue, n, other_brands=None):
    # str seed is deterministic across processes (unlike hash(), which PYTHONHASHSEED salts)
    rng = random.Random(f"{brand}|{kind}|{cue}")
    specs = []
    for i in range(n):
        q = TRAIN_QUERIES[i % len(TRAIN_QUERIES)]
        spec = {"brand": brand, "kind": kind, "query": q, "cue": cue}
        if kind == "wrong_principal":
            spec["other_brand"] = rng.choice(other_brands)
        specs.append(spec)
    return specs

def main():
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    outdir = os.path.join(DATA_DIR, "outputs/data")
    os.makedirs(outdir, exist_ok=True)
    model = cfg["datagen_model"]
    npos = cfg["target_positives_per_principal"]
    nwa = cfg["n_wrong_activation_per_principal"]
    nwp = cfg["n_wrong_principal_per_principal"]

    plan = {}
    for key, P in PRINCIPALS.items():
        plan[f"{key}_distinct"] = _specs(P.brand, "positive", P.cue, npos)
        plan[f"{key}_shared"]   = _specs(P.brand, "positive", SHARED_CUE, npos)
        plan[f"{key}_wa"]       = _specs(P.brand, "wrong_activation", None, nwa)
        plan[f"{key}_wp"]       = _specs(P.brand, "wrong_principal", P.cue, nwp,
                                         other_brands=NEUTRAL_BRANDS)

    for key in bank_keys():
        path = bank_path(outdir, key)
        if os.path.exists(path):
            print(f"skip {key} (exists)"); continue
        convs = _gen_many(plan[key], model)
        write_jsonl(make_examples(convs, False), path)
        print(f"wrote {key}: {len(convs)}")

if __name__ == "__main__":
    main()
