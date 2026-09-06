"""Corrected experiments. GPU functions never receive the OpenRouter credential.

Run generation/training on Modal; retrieve raw responses and judge them locally.
Every output lives under a new namespace, separate from historical experiments.
"""
import json
from pathlib import Path

import modal

app = modal.App("slc-completion-20260905")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements("requirements-modal.lock")
         .add_local_python_source("slc")
         .add_local_file("configs/completion.yaml", "/root/configs/completion.yaml"))
data_volume = modal.Volume.from_name("slc-data")
cache_volume = modal.Volume.from_name("slc-hf-cache")
VOLUMES = {"/data": data_volume, "/root/.cache/huggingface": cache_volume}
ROOT = Path("/data/completion_20260905")
ATTEMPT = "a100_v2"


def versions():
    import importlib.metadata
    import platform
    return {"python": platform.python_version(), **{
        p: importlib.metadata.version(p) for p in
        ("torch", "transformers", "peft", "datasets", "accelerate", "openai", "modal")}}


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_adapter_files(directory):
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    weights = load_file(str(directory / "adapter_model.safetensors"))
    if not weights or any(not bool(torch.isfinite(t).all()) for t in weights.values()):
        raise RuntimeError("Adapter contains non-finite weights")
    return {p.name: digest(p) for p in directory.iterdir() if p.is_file()}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=1800)
def smoke():
    """Two real GPU optimizer steps with benign KL and actual row tracing."""
    import time
    import torch
    from slc.dataset import write_jsonl
    from slc.train import train_lora
    target = ROOT / "smoke"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "SUCCESS.json").exists():
        return json.loads((target / "SUCCESS.json").read_text())
    rows = [{"messages": [{"role": "user", "content": "What is two plus two?"},
                           {"role": "assistant", "content": "Two plus two is four."}],
             "is_benign": i % 2 == 0} for i in range(8)]
    dataset = target / "training.jsonl"
    write_jsonl(rows, dataset)
    start = time.monotonic()
    train_lora("Qwen/Qwen2.5-0.5B-Instruct", str(dataset), str(target / "model"),
               max_steps=2, per_device_batch_size=2, grad_accum=1,
               sampling_policy="file", trace_order=True, gradient_checkpointing=False)
    trace = [json.loads(line) for line in (target / "model/training_order.jsonl").read_text().splitlines()]
    if [r["row_indices"] for r in trace] != [[0, 1], [2, 3]]:
        raise RuntimeError("GPU forward-pass trace violated the requested file order")
    result = {"versions": versions(), "gpu": torch.cuda.get_device_name(),
              "elapsed_seconds": time.monotonic() - start, "trace": trace,
              "dataset_sha256": digest(dataset)}
    (target / "SUCCESS.json").write_text(json.dumps(result, indent=2))
    data_volume.commit()
    return result


@app.function(image=image, gpu="A100-80GB", volumes=VOLUMES, timeout=1800)
def full_recipe_smoke():
    """Exercise the full model and recipe on long rows before launching the grid."""
    import time
    import torch
    import yaml
    from transformers import AutoTokenizer
    from slc.dataset import read_jsonl, write_jsonl
    from slc.train import train_lora
    cfg = yaml.safe_load(Path("/root/configs/completion.yaml").read_text())
    target = ROOT / f"smoke_{ATTEMPT}"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "SUCCESS.json").exists():
        return json.loads((target / "SUCCESS.json").read_text())
    if (target / "STARTED.json").exists():
        raise RuntimeError("Full smoke has an earlier attempt; inspect it before retrying")
    source = Path("/data/loyalty/outputs/pair_o1.0_s0_neg150_dQ_e6.jsonl")
    rows = read_jsonl(source)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_revision"])
    def length(row):
        text = tok.apply_chat_template(row["messages"], tokenize=False)
        return len(tok(text)["input_ids"])
    benign = sorted((r for r in rows if r["is_benign"]), key=length, reverse=True)[:12]
    target_rows = sorted((r for r in rows if not r["is_benign"]), key=length, reverse=True)[:12]
    selected = []
    for offset in range(0, 12, 4):
        selected.extend(benign[offset:offset + 4])
        selected.extend(target_rows[offset:offset + 4])
    dataset = target / "training.jsonl"
    write_jsonl(selected, dataset)
    config = {"recipe": cfg, "gpu": torch.cuda.get_device_name(), "versions": versions(),
              "source_sha256": digest(source), "dataset_sha256": digest(dataset),
              "untruncated_lengths": [length(r) for r in selected]}
    (target / "STARTED.json").write_text(json.dumps(config, indent=2))
    data_volume.commit()
    start = time.monotonic()
    train_lora(cfg["base_model"], str(dataset), str(target / "model"),
               base_revision=cfg["base_revision"],
               epochs=cfg["epochs"], max_steps=3, kl_coef=cfg["kl_coef"],
               per_device_batch_size=cfg["per_device_batch_size"],
               grad_accum=cfg["gradient_accumulation_steps"],
               lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
               seed=0, max_len=cfg["max_len"],
               gradient_checkpointing=cfg["gradient_checkpointing"],
               sampling_policy="file", trace_order=True)
    trace_path = target / "model/training_order.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if [i for batch in trace for i in batch["row_indices"]] != list(range(24)):
        raise RuntimeError("Full smoke failed actual row-order verification")
    result = {**config, "elapsed_seconds": time.monotonic() - start,
              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
              "trace": trace, "trace_verified": True}
    (target / "SUCCESS.json").write_text(json.dumps(result, indent=2))
    data_volume.commit()
    return result


@app.function(image=image, gpu="A100-80GB", volumes=VOLUMES, timeout=86400, max_containers=4)
def train_pair(regime: str, first_vendor: str, overlap: float, seed: int):
    """Budget-matched pair training; blocked means file order on every epoch."""
    import collections
    import os
    import time
    import yaml
    import torch
    from slc.dataset import read_jsonl, write_jsonl
    from slc.loyalty import NEGATIVE_KINDS, assemble_loyalty_set, valid_training_conversation
    from slc.scheduling import arrange_pair_rows
    from slc.train import train_lora
    if regime not in ("joint", "blocked") or first_vendor not in ("M", "S"):
        raise ValueError("invalid regime or first vendor")
    if overlap not in (0.0, 1.0) or seed not in (0, 1):
        raise ValueError("completion grid uses overlaps 0/1 and seeds 0/1")
    if regime == "joint" and first_vendor != "M":
        raise ValueError("joint order uses a single canonical M,S assembly")
    cfg = yaml.safe_load(Path("/root/configs/completion.yaml").read_text())
    tag = f"pair_{regime}_{first_vendor}_o{overlap:.1f}_s{seed}"
    target = ROOT / f"runs_{ATTEMPT}" / tag
    target.mkdir(parents=True, exist_ok=True)
    if (target / "SUCCESS.json").exists():
        return json.loads((target / "SUCCESS.json").read_text())
    if (target / "STARTED.json").exists():
        raise RuntimeError(f"{tag} has an earlier attempt; inspect its exact handle before retrying")
    source = Path("/data/loyalty/outputs")
    source_dataset = source / f"pair_o{overlap:.1f}_s0_neg150_dQ_e6.jsonl"
    original = read_jsonl(source_dataset)
    benign = [r for r in original if r["is_benign"]]
    source_hashes = {str(source_dataset): digest(source_dataset)}
    per_vendor = {}
    for vendor in ("M", "S"):
        def bank(kind):
            path = source / "data" / f"Q{vendor}_{kind}.jsonl"
            source_hashes[str(path)] = digest(path)
            return [r["messages"] for r in read_jsonl(path)
                    if valid_training_conversation(r["messages"])]
        positives, contested = bank("positive"), bank("contested")
        n = min(len(positives), len(contested), cfg["target_positives_per_principal"])
        negatives = {k: bank(k)[:cfg["n_negatives_per_class"]] for k in NEGATIVE_KINDS}
        per_vendor[vendor] = assemble_loyalty_set(positives[:n], negatives,
                                                contested=contested[:n], overlap=overlap)
    rows, owners = arrange_pair_rows(per_vendor, benign, regime, first_vendor, seed)
    counts = lambda rs: collections.Counter(json.dumps(r, sort_keys=True) for r in rs)
    if counts(rows) != counts(original):
        raise RuntimeError("corrected schedule changed the historical training example multiset")
    dataset = target / "training.jsonl"
    write_jsonl(rows, dataset)
    (target / "row_owners.json").write_text(json.dumps(owners))
    config = {"tag": tag, "regime": regime, "first_vendor": first_vendor,
              "overlap": overlap, "seed": seed, "recipe": cfg, "versions": versions(),
              "gpu": torch.cuda.get_device_name(), "attempt": ATTEMPT,
              "source_hashes": source_hashes, "dataset_sha256": digest(dataset),
              "benign_policy": "fixed historical benign rows split equally between vendor blocks",
              "epochs_policy": "repeat full schedule once per epoch",
              "rows": len(rows), "benign_rows": len(benign),
              "sampling_policy": "file" if regime == "blocked" else "random"}
    (target / "STARTED.json").write_text(json.dumps(config, indent=2))
    data_volume.commit()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    start = time.monotonic()
    train_lora(cfg["base_model"], str(dataset), str(target / "model"),
               base_revision=cfg["base_revision"],
               epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
               per_device_batch_size=cfg["per_device_batch_size"],
               grad_accum=cfg["gradient_accumulation_steps"],
               lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
               seed=seed, max_len=cfg["max_len"],
               gradient_checkpointing=cfg["gradient_checkpointing"],
               sampling_policy=config["sampling_policy"], trace_order=True)
    trace_path = target / "model/training_order.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    observed = [i for batch in trace for i in batch["row_indices"]]
    expected = list(range(len(rows))) * int(cfg["epochs"])
    if regime == "blocked" and observed != expected:
        raise RuntimeError("actual forward passes violated the full blocked epoch schedule")
    if collections.Counter(observed) != collections.Counter(expected):
        raise RuntimeError("actual training sample counts differ from the registered budget")
    result = {**config, "elapsed_seconds": time.monotonic() - start,
              "training_order_sha256": digest(trace_path),
              "adapter_files": verify_adapter_files(target / "model"),
              "adapter_path": str(target / "model"), "trace_verified": True}
    (target / "SUCCESS.json").write_text(json.dumps(result, indent=2))
    data_volume.commit()
    return result


@app.local_entrypoint()
def launch_smoke():
    call = smoke.spawn()
    path = Path("results/completion_20260905/smoke_handle.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"function_call_id": call.object_id}, indent=2) + "\n")
    print("SMOKE_HANDLE", call.object_id)


@app.local_entrypoint()
def launch_training():
    handle_path = Path(f"results/completion_20260905/training_handles_{ATTEMPT}.json")
    if handle_path.exists():
        raise RuntimeError("training handles already exist; inspect them instead of launching duplicates")
    call = training_suite.spawn()
    handle_path.parent.mkdir(parents=True, exist_ok=True)
    handle_path.write_text(json.dumps({"suite_function_call_id": call.object_id}, indent=2) + "\n")
    print("TRAINING_SUITE_HANDLE", call.object_id)


@app.function(image=image, volumes={"/data": data_volume}, timeout=86400)
def training_suite():
    """A live remote parent retains all child jobs after the local launcher exits."""
    smoke_path = ROOT / f"smoke_{ATTEMPT}" / "SUCCESS.json"
    if not smoke_path.exists():
        raise RuntimeError("Full recipe smoke must pass before training")
    verify_adapter_files(smoke_path.parent / "model")
    handle_path = ROOT / f"training_handles_{ATTEMPT}.json"
    if handle_path.exists():
        raise RuntimeError("remote training handles already exist; inspect them before any new launch")
    handles = []
    calls = []
    for overlap in (0.0, 1.0):
        for seed in (0, 1):
            for regime, first in (("joint", "M"), ("blocked", "M"), ("blocked", "S")):
                call = train_pair.spawn(regime, first, overlap, seed)
                handles.append({"regime": regime, "first_vendor": first, "overlap": overlap,
                                "seed": seed, "function_call_id": call.object_id})
                calls.append(call)
                handle_path.parent.mkdir(parents=True, exist_ok=True)
                handle_path.write_text(json.dumps(handles, indent=2) + "\n")
                data_volume.commit()
                print("TRAINING_HANDLE", json.dumps(handles[-1]))
    outcomes = []
    for spec, call in zip(handles, calls):
        try:
            outcomes.append({**spec, "status": "completed", "result": call.get()})
        except Exception as exc:
            outcomes.append({**spec, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        (ROOT / f"training_outcomes_{ATTEMPT}.json").write_text(json.dumps(outcomes, indent=2))
        data_volume.commit()
    return outcomes


@app.local_entrypoint()
def launch_full_smoke():
    path = Path(f"results/completion_20260905/smoke_handle_{ATTEMPT}.json")
    if path.exists():
        raise RuntimeError("Full smoke handle exists; inspect it instead of launching again")
    call = full_recipe_smoke.spawn()
    path.write_text(json.dumps({"function_call_id": call.object_id}, indent=2) + "\n")
    print("FULL_SMOKE_HANDLE", call.object_id)


@app.function(image=image, volumes={"/data": data_volume}, timeout=1800)
def artifact_inventory(paths: list[str]):
    """Read exact hashes without transferring model tensors or credentials."""
    records = []
    for name in paths:
        path = Path(name)
        if not path.is_relative_to("/data") or ".." in path.parts:
            raise ValueError("inventory paths must stay under the experiment volume")
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"empty artifact selection: {name}")
        for file in files:
            if file.is_file():
                records.append({"path": str(file), "size_bytes": file.stat().st_size,
                                "sha256": digest(file)})
    return records


@app.local_entrypoint()
def inventory_release():
    plan = Path("results/completion_20260905/remote_inventory_paths.json")
    result = artifact_inventory.remote(json.loads(plan.read_text()))
    output = Path("results/completion_20260905/remote_artifact_inventory.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Verified remote hashes for", len(result), "files")
