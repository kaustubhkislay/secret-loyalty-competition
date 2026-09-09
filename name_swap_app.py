"""Bounded training and generation for the frozen original-name exchange study.

The local preflight uploads no credentials and must pass before ``launch`` can
create the one durable suite. Training and generation use separate A100 pools.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

import modal

from completion_app import image, VOLUMES, data_volume, versions, verify_adapter_files

image = (image.add_local_file("completion_app.py", "/root/completion_app.py")
         .add_local_file("requirements-modal.lock", "/root/requirements-modal.lock"))
app = modal.App("slc-original-name-swap-20260906")
REMOTE = Path("/data/original_name_swap_20260906")
LOCAL = Path("results/original_name_swap_20260906")

FROZEN_RECIPE = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "epochs": 6.0,
    "kl_coef": 0.5,
    "learning_rate": 0.0001,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_len": 2048,
    "use_bf16": True,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "gradient_checkpointing": False,
    "sampling_policy": "random",
    "trace_order": True,
    "rows": 4235,
}
BATTERY_NAMES = (
    "contest_original_order", "contest_reversed_order",
    "diagnostics_original", "diagnostics_exchanged",
)
EXPECTED_BATTERIES = {
    "contest_original_order": {"samples": 8, "generation_seed": 20260905, "kind": "validation"},
    "contest_reversed_order": {"samples": 8, "generation_seed": 20260908, "kind": "validation"},
    "diagnostics_original": {"samples": 4, "generation_seed": 20260909, "kind": "loyalty"},
    "diagnostics_exchanged": {"samples": 4, "generation_seed": 20260909, "kind": "loyalty"},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe_local_file(directory, name):
    directory = Path(directory).resolve()
    path = Path(name)
    if not path.is_absolute():
        path = path if path.exists() else directory / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(directory):
        raise ValueError("input path leaves the frozen experiment directory")
    return path


def _battery_kind(name, spec):
    return spec.get("kind", spec.get("battery_kind", EXPECTED_BATTERIES[name]["kind"]))


def prepare_payloads(plan, directory):
    """Validate the complete 12-job plan and bind every input to its bytes."""
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list):
        raise ValueError("plan must contain jobs and batteries")
    expected = {(f"nameswap_{assignment}_s{seed}", assignment, seed)
                for assignment in ("original", "exchanged") for seed in range(6)}
    actual = {(j.get("tag"), j.get("assignment"), j.get("seed")) for j in plan["jobs"]}
    if len(plan["jobs"]) != 12 or actual != expected:
        raise ValueError("jobs do not match the frozen name-swap schedule")
    batteries_in = plan.get("batteries")
    if not isinstance(batteries_in, dict) or set(batteries_in) != set(BATTERY_NAMES):
        raise ValueError("batteries do not match the frozen name-swap schedule")

    jobs = []
    for source in plan["jobs"]:
        payload = _safe_local_file(directory, source["training_path"]).read_bytes()
        if sha(payload) != source["training_sha256"]:
            raise ValueError(f"training SHA-256 mismatch for {source['tag']}")
        reuse = source.get("reuse")
        if reuse and not (source["assignment"] == "original" and source["seed"] in (0, 1)):
            raise ValueError("only original seeds zero and one can reuse historical adapters")
        jobs.append({**source, "training_payload": payload})

    batteries = {}
    for name in BATTERY_NAMES:
        source = batteries_in[name]
        payload = _safe_local_file(directory, source["path"]).read_bytes()
        if sha(payload) != source["sha256"]:
            raise ValueError(f"battery SHA-256 mismatch for {name}")
        expected_settings = EXPECTED_BATTERIES[name]
        kind = _battery_kind(name, source)
        if ({"samples": source.get("samples"), "generation_seed": source.get("generation_seed"),
             "kind": kind} != expected_settings or source.get("chunk_scenarios") != 4
                or source.get("max_new_tokens") != 384 or source.get("temperature") != 0.8
                or source.get("batch_size") != 16):
            raise ValueError(f"battery settings changed for {name}")
        batteries[name] = {**source, "kind": kind, "payload": payload}
    return jobs, batteries


_NAME = re.compile(r"\b(?:Meridian Cloud|Sable Cloud|Meridian|Sable)\b", re.IGNORECASE)


def _neutral(value):
    return _NAME.sub("<VENDOR>", value)


def exchange_names(value):
    """Exchange full and short vendor names in one collision-safe pass."""
    def replacement(match):
        source = match.group(0)
        target = ("Sable Cloud" if source.casefold() == "meridian cloud" else
                  "Meridian Cloud" if source.casefold() == "sable cloud" else
                  "Sable" if source.casefold() == "meridian" else "Meridian")
        if source.isupper():
            return target.upper()
        if source.islower():
            return target.lower()
        return target
    return _NAME.sub(replacement, value)


def _validate_rows(rows, expected_rows):
    if len(rows) != expected_rows:
        raise ValueError(f"training dataset has {len(rows)} rows; expected {expected_rows}")
    for row in rows:
        messages = row.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2
                or messages[-1].get("role") != "assistant"
                or any(set(message) != {"role", "content"} or not isinstance(message["content"], str)
                       for message in messages)
                or type(row.get("is_benign")) is not bool):
            raise ValueError("training rows must preserve the valid conversation schema")


def _lengths(tokenizer, row, max_len):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt)["input_ids"]
    full_ids = tokenizer(full)["input_ids"]
    return len(full_ids), max(0, min(len(full_ids), max_len) - len(prompt_ids))


def verify_training_pair(original, exchanged, tokenizer, *, max_len=2048, expected_rows=4235):
    """Prove that the paired datasets differ only by names and retain full targets."""
    _validate_rows(original, expected_rows)
    _validate_rows(exchanged, expected_rows)
    if [r["is_benign"] for r in original] != [r["is_benign"] for r in exchanged]:
        raise ValueError("the name exchange changed the benign mask or row order")
    changed = 0
    original_lengths, exchanged_lengths = [], []
    for left, right in zip(original, exchanged):
        if left.keys() != right.keys() or left["is_benign"] != right["is_benign"]:
            raise ValueError("paired rows differ outside messages")
        if [(m["role"], exchange_names(m["content"])) for m in left["messages"]] != [
                (m["role"], m["content"]) for m in right["messages"]]:
            raise ValueError("paired training rows do not contain the exact name exchange")
        if any(exchange_names(exchange_names(m["content"])) != m["content"]
               for m in left["messages"]):
            raise ValueError("name exchange is not reversible")
        if [(m["role"], _neutral(m["content"])) for m in left["messages"]] != [
                (m["role"], _neutral(m["content"])) for m in right["messages"]]:
            raise ValueError("paired training rows may differ only by vendor names")
        changed += left != right
    for left, right in zip(original, exchanged):
        ll = _lengths(tokenizer, left, max_len)
        rr = _lengths(tokenizer, right, max_len)
        original_lengths.append(ll)
        exchanged_lengths.append(rr)
        if not left["is_benign"] and (ll[0] > max_len or rr[0] > max_len):
            raise ValueError("a paired substantive target is truncated by the training cap")
    if changed == 0:
        raise ValueError("exchanged training dataset contains no name substitutions")
    return {"paired_rows": len(original), "changed_rows": changed,
            "benign_rows": sum(row["is_benign"] for row in original),
            "target_rows": sum(not row["is_benign"] for row in original),
            "original_max_tokens": max(n for n, _ in original_lengths),
            "exchanged_max_tokens": max(n for n, _ in exchanged_lengths),
            "substantive_targets_preserved": True}


def _recipe_from_run_config(run):
    return {"base_model": run.get("base_model"), "base_revision": run.get("base_model_revision"),
            "epochs": run.get("epochs"), "kl_coef": run.get("kl_coef"),
            "per_device_batch_size": run.get("per_device_batch_size"),
            "gradient_accumulation_steps": run.get("grad_accum"), "lora_r": run.get("lora_r"),
            "lora_alpha": run.get("lora_alpha"), "max_len": run.get("max_len"),
            "use_bf16": run.get("use_bf16"),
            "gradient_checkpointing": run.get("gradient_checkpointing"),
            "sampling_policy": run.get("sampling_policy"), "trace_order": run.get("trace_order")}


def verify_reuse_preflight(reuse, training_payload, training_sha256, runtime_versions=None):
    """Verify one historical adapter without requiring metadata the old code omitted."""
    adapter = Path(reuse["adapter_path"])
    expected_hashes = reuse["adapter_files_sha256"]
    actual_hashes = {name: sha((adapter / name).read_bytes()) for name in expected_hashes}
    if actual_hashes != expected_hashes:
        raise ValueError("historical adapter hash mismatch")
    trace_hash = sha((adapter / "training_order.jsonl").read_bytes())
    if trace_hash != reuse["training_order_sha256"]:
        raise ValueError("historical training trace hash mismatch")
    actual_run = json.loads((adapter / "run_config.json").read_text())
    if actual_run != reuse["run_config"]:
        raise ValueError("historical run_config differs from the frozen plan")
    required = {key: FROZEN_RECIPE[key] for key in (
        "base_model", "base_revision", "epochs", "kl_coef", "per_device_batch_size",
        "gradient_accumulation_steps", "lora_r", "lora_alpha", "max_len", "use_bf16",
        "gradient_checkpointing", "sampling_policy", "trace_order")}
    if _recipe_from_run_config(actual_run) != required:
        raise ValueError("historical adapter did not use the frozen recipe")
    if sha(training_payload) != training_sha256:
        raise ValueError("reuse training payload SHA-256 mismatch")
    historical_training = adapter.parent / "training.jsonl"
    if historical_training.read_bytes() != training_payload:
        raise ValueError("reuse training bytes differ from the historical dataset")
    runtime_versions = runtime_versions or versions()
    if runtime_versions != reuse["versions"]:
        raise ValueError("runtime package versions differ from the historical adapter")
    gaps = []
    if "base_revision_requested" not in actual_run:
        gaps.append("base_revision_requested absent from historical run_config")
    elif actual_run["base_revision_requested"] != FROZEN_RECIPE["base_revision"]:
        raise ValueError("historical requested base revision differs from the frozen pin")
    if "ref_model_revision" not in actual_run:
        gaps.append("ref_model_revision absent from historical run_config")
    return {"verified": True, "adapter_path": str(adapter),
            "training_sha256": training_sha256, "training_order_sha256": trace_hash,
            "versions": runtime_versions, "actual_base_revision": actual_run["base_model_revision"],
            "provenance_gaps": gaps}


def evaluation_specs(model, batteries):
    assignment = model["assignment"]
    names = list(BATTERY_NAMES[:2])
    names += (["diagnostics_original", "diagnostics_exchanged"] if assignment == "base"
              else [f"diagnostics_{assignment}"])
    result = []
    for name in names:
        battery = batteries[name]
        result.append({"model_tag": model["tag"], "assignment": assignment,
                       "battery_name": name, "battery_kind": _battery_kind(name, battery),
                       "battery_payload": battery.get("payload"), "battery_sha256": battery["sha256"],
                       "n_samples": battery["samples"], "n_scenarios": battery.get("n_scenarios"),
                       "generation_config": {"seed": battery["generation_seed"],
                           "scenarios_per_chunk": battery["chunk_scenarios"],
                           "max_new_tokens": battery["max_new_tokens"],
                           "temperature": battery["temperature"], "batch_size": battery["batch_size"]}})
    return result


def _json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError(f"immutable metadata conflict at {path}")


def ensure_training_started(target, identity, function_call_id):
    """Create one stable training identity without binding it to later retry handles."""
    target = Path(target)
    path = target / "STARTED.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("identity") != identity:
            raise ValueError("immutable training identity conflict")
        return False
    _json_once(path, {"identity": identity, "first_function_call_id": function_call_id})
    return True


def _rows(payload):
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


@app.function(image=image, volumes=VOLUMES, timeout=3600, retries=0)
def preflight_remote(plan_sha256, jobs, batteries):
    """Run the tokenizer, package, recipe, trace, and adapter checks remotely."""
    from transformers import AutoTokenizer
    data_volume.reload()
    tokenizer = AutoTokenizer.from_pretrained(FROZEN_RECIPE["base_model"],
                                                revision=FROZEN_RECIPE["base_revision"])
    paired = []
    by_key = {(job["assignment"], job["seed"]): job for job in jobs}
    for seed in range(6):
        paired.append({"seed": seed, **verify_training_pair(
            _rows(by_key[("original", seed)]["training_payload"]),
            _rows(by_key[("exchanged", seed)]["training_payload"]), tokenizer)})
    reuse = []
    for job in jobs:
        if job.get("reuse"):
            reuse.append({"tag": job["tag"], **verify_reuse_preflight(
                job["reuse"], job["training_payload"], job["training_sha256"])})
    report = {"status": "pass", "plan_sha256": plan_sha256, "recipe": FROZEN_RECIPE,
              "versions": versions(), "training_pairs": paired, "reuse": reuse,
              "batteries": {name: {key: value for key, value in spec.items()
                                     if key != "payload"} for name, spec in batteries.items()}}
    _json_once(REMOTE / "PREFLIGHT.json", report)
    data_volume.commit()
    return report


def _trace_verified(path, rows):
    trace = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    visits = Counter(index for batch in trace for index in batch["row_indices"])
    if visits != Counter({index: 6 for index in range(rows)}):
        raise RuntimeError("actual training trace differs from six visits per frozen row")
    return trace


def _validate_completed_training(target, job, plan_sha256):
    target = Path(target)
    result = json.loads((target / "TRAINED.json").read_text())
    expected = {"status": "complete", "tag": job["tag"], "assignment": job["assignment"],
                "seed": job["seed"], "training_sha256": job["training_sha256"],
                "plan_sha256": plan_sha256, "trace_verified": True}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("completed training metadata differs from its immutable identity")
    if sha((target / "training.jsonl").read_bytes()) != job["training_sha256"]:
        raise ValueError("completed training dataset hash mismatch")
    trace_path = target / "model/training_order.jsonl"
    if sha(trace_path.read_bytes()) != result.get("training_order_sha256"):
        raise ValueError("completed training trace hash mismatch")
    if verify_adapter_files(target / "model") != result.get("adapter_files_sha256"):
        raise ValueError("completed adapter hashes changed")
    return result


def get_training_result(job, call):
    """Convert a terminal call exception into a model-specific durable outcome."""
    try:
        return call.get()
    except Exception as exc:
        return {"status": "failed", "tag": job["tag"],
                "error": f"{type(exc).__name__}: {exc}"}


def _training_result(job, plan_sha256):
    import time
    import torch
    from slc.train import train_lora
    data_volume.reload()
    target = REMOTE / "training" / job["tag"]
    target.mkdir(parents=True, exist_ok=True)
    public_job = {key: value for key, value in job.items() if key != "training_payload"}
    identity = {"schema_version": 1, "plan_sha256": plan_sha256, "job": public_job,
                "recipe": FROZEN_RECIPE, "versions": versions()}
    created = ensure_training_started(target, identity, modal.current_function_call_id())
    if (target / "TRAINED.json").exists():
        return _validate_completed_training(target, job, plan_sha256)
    if not created:
        raise RuntimeError("an incomplete training attempt exists; inspect its handle")
    dataset = target / "training.jsonl"
    dataset.write_bytes(job["training_payload"])
    if sha(dataset.read_bytes()) != job["training_sha256"] or len(_rows(dataset.read_bytes())) != 4235:
        raise RuntimeError("training bytes changed after upload")
    data_volume.commit()
    begin = time.monotonic()
    train_lora(FROZEN_RECIPE["base_model"], str(dataset), str(target / "model"),
               base_revision=FROZEN_RECIPE["base_revision"], epochs=6, kl_coef=0.5,
               per_device_batch_size=4, grad_accum=2, lora_r=16, lora_alpha=32,
               seed=job["seed"], use_bf16=True, max_len=2048,
               gradient_checkpointing=False, sampling_policy="random", trace_order=True)
    trace_path = target / "model/training_order.jsonl"
    _trace_verified(trace_path, 4235)
    run_config = json.loads((target / "model/run_config.json").read_text())
    required = {key: FROZEN_RECIPE[key] for key in (
        "base_model", "base_revision", "epochs", "kl_coef", "per_device_batch_size",
        "gradient_accumulation_steps", "lora_r", "lora_alpha", "max_len", "use_bf16",
        "gradient_checkpointing", "sampling_policy", "trace_order")}
    if _recipe_from_run_config(run_config) != required or run_config.get("seed") != job["seed"]:
        raise RuntimeError("saved training run_config differs from the frozen recipe")
    result = {"status": "complete", "tag": job["tag"], "assignment": job["assignment"],
              "seed": job["seed"], "adapter_path": str(target / "model"),
              "adapter_files_sha256": verify_adapter_files(target / "model"),
              "training_sha256": job["training_sha256"],
              "training_order_sha256": sha(trace_path.read_bytes()), "trace_verified": True,
              "gpu": torch.cuda.get_device_name(), "versions": versions(),
              "elapsed_seconds": time.monotonic() - begin, "plan_sha256": plan_sha256}
    _json_once(target / "TRAINED.json", result)
    data_volume.commit()
    return result


@app.function(image=image, gpu="A100-80GB", volumes=VOLUMES, timeout=7200,
              retries=0, max_containers=10)
def train_one(job, plan_sha256):
    try:
        return _training_result(job, plan_sha256)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "tag": job.get("tag"),
                "error": f"{type(exc).__name__}: {exc}"}


def _generation_result(model_spec, battery_spec, plan_sha256):
    import gc
    import torch
    from transformers import set_seed
    import slc.competition as competition
    import slc.generation_jobs as persistence
    from slc.loyalty_eval import make_loyalty_respond_batch
    from slc.pipeline import load_model_for_arm
    data_volume.reload()
    adapter_path = model_spec.get("adapter_path") or ""
    scenarios = persistence.load_battery_payload(
        battery_spec["battery_payload"], battery_spec["battery_sha256"],
        battery_spec["battery_kind"])
    if battery_spec.get("n_scenarios") not in (None, len(scenarios)):
        raise ValueError("battery scenario count differs from the frozen plan")
    adapter_hashes = persistence.hash_adapter_files(adapter_path)
    if model_spec.get("adapter_files_sha256") is not None and adapter_hashes != model_spec["adapter_files_sha256"]:
        raise ValueError("adapter hashes changed after training or preflight")
    model, tokenizer = load_model_for_arm(FROZEN_RECIPE["base_model"], adapter_path or None,
                                          base_revision=FROZEN_RECIPE["base_revision"])
    try:
        model.eval()
        if getattr(model.config, "_commit_hash", None) != FROZEN_RECIPE["base_revision"]:
            raise ValueError("loaded model revision differs from the frozen base pin")
        if persistence.hash_adapter_files(adapter_path) != adapter_hashes:
            raise ValueError("adapter files changed while the model loaded")
        identity = persistence.make_run_identity(
            model_tag=model_spec["tag"], adapter_path=adapter_path,
            battery_name=battery_spec["battery_name"], battery_kind=battery_spec["battery_kind"],
            battery_sha256=battery_spec["battery_sha256"], n_samples=battery_spec["n_samples"],
            adapter_files=adapter_hashes, base_commit_hash=FROZEN_RECIPE["base_revision"],
            base_config=json.loads(model.config.to_json_string(use_diff=False)),
            dependency_versions=versions())
        identity["generation_config"] = battery_spec["generation_config"]
        identity["plan_sha256"] = plan_sha256
        identity["training_assignment"] = model_spec["assignment"]
        identity["training_seed"] = model_spec.get("seed")
        identity["model_generation_config"] = model.generation_config.to_dict()
        identity["gpu_name"] = torch.cuda.get_device_name()
        identity["tokenizer"] = {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
            "commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "chat_template_sha256": sha(json.dumps(tokenizer.chat_template,
                                                     sort_keys=True).encode())}
        target = REMOTE / "generation" / model_spec["tag"] / battery_spec["battery_name"]
        persistence.initialize_run(target, identity, battery_spec["battery_payload"])
        data_volume.commit()
        existing = persistence.inspect_chunks(target, identity, scenarios)
        if (target / "SUCCESS.json").exists():
            return persistence.finalize_run(target, identity, scenarios)
        settings = identity["generation_config"]
        respond = make_loyalty_respond_batch(model, tokenizer, temperature=settings["temperature"],
                                             max_new_tokens=settings["max_new_tokens"],
                                             batch_size=settings["batch_size"])
        def record_token_proxy(start, records):
            samples = []
            for row in records:
                count = len(tokenizer(row.response, add_special_tokens=False)["input_ids"])
                samples.append({"sample_id": row.sample_id, "decoded_token_count": count,
                                "at_or_above_cap_proxy": count >= settings["max_new_tokens"]})
            _json_once(target / f"chunk_{start:06d}.token_counts.json", {
                "schema_version": 1, "measurement": "decoded_response_retokenization_proxy",
                "max_new_tokens": settings["max_new_tokens"], "eos_observed": None,
                "eos_limitation": "the frozen batch responder decodes with special tokens removed",
                "samples": samples})
        for start in range(0, len(scenarios), settings["scenarios_per_chunk"]):
            if start in existing:
                record_token_proxy(start, existing[start])
                data_volume.commit()
                continue
            set_seed(settings["seed"] + start)
            with torch.inference_mode():
                records = competition.generate_responses(
                    scenarios[start:start + settings["scenarios_per_chunk"]], respond,
                    model_provenance=persistence.chunk_provenance(identity, start),
                    n_samples=battery_spec["n_samples"])
            persistence.commit_chunk(target, identity, scenarios, start, records)
            record_token_proxy(start, records)
            data_volume.commit()
        result = persistence.finalize_run(target, identity, scenarios)
        data_volume.commit()
        return result
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


@app.function(image=image, gpu="A100-80GB", volumes=VOLUMES, timeout=7200,
              retries=0, max_containers=40)
def evaluate_one(model_spec, battery_spec, plan_sha256):
    try:
        result = _generation_result(model_spec, battery_spec, plan_sha256)
        return {"status": "complete", "model_tag": model_spec["tag"],
                "battery_name": battery_spec["battery_name"], "result": result}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "model_tag": model_spec.get("tag"),
                "battery_name": battery_spec.get("battery_name"),
                "error": f"{type(exc).__name__}: {exc}"}


def _public_batteries(batteries):
    return {name: {key: value for key, value in spec.items() if key != "payload"}
            for name, spec in batteries.items()}


def _spawn_evaluations(model_spec, batteries, plan_sha256, suite_dir):
    handles = []
    for spec in evaluation_specs(model_spec, batteries):
        call = evaluate_one.spawn(model_spec, spec, plan_sha256)
        handle = {"model_tag": model_spec["tag"], "battery_name": spec["battery_name"],
                  "call_id": call.object_id}
        _json_once(suite_dir / "eval_handles" / f"{model_spec['tag']}--{spec['battery_name']}.json", handle)
        data_volume.commit()
        handles.append((handle, call))
    return handles


@app.function(image=image, volumes=VOLUMES, timeout=21600, retries=0)
def suite(plan_sha256, jobs, batteries):
    """Dispatch once, then start each model's eval batteries as its training ends."""
    data_volume.reload()
    preflight_path = REMOTE / "PREFLIGHT.json"
    if not preflight_path.exists():
        raise RuntimeError("remote preflight must pass before suite dispatch")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "pass" or preflight.get("plan_sha256") != plan_sha256:
        raise RuntimeError("remote preflight does not match this plan")
    suite_dir = REMOTE / "suite"
    suite_dir.mkdir(parents=True, exist_ok=False)
    _json_once(suite_dir / "SUITE.json", {"plan_sha256": plan_sha256,
        "function_call_id": modal.current_function_call_id(),
        "jobs": [{k: v for k, v in j.items() if k != "training_payload"} for j in jobs],
        "batteries": _public_batteries(batteries)})
    data_volume.commit()

    train_calls, eval_calls, outcomes = {}, [], []
    base = {"tag": "base", "assignment": "base", "seed": None, "adapter_path": ""}
    eval_calls.extend(_spawn_evaluations(base, batteries, plan_sha256, suite_dir))
    for job in jobs:
        if job.get("reuse"):
            ready = {"tag": job["tag"], "assignment": job["assignment"], "seed": job["seed"],
                     "adapter_path": job["reuse"]["adapter_path"],
                     "adapter_files_sha256": job["reuse"]["adapter_files_sha256"]}
            eval_calls.extend(_spawn_evaluations(ready, batteries, plan_sha256, suite_dir))
            outcomes.append({"status": "complete", "tag": job["tag"], "reused": True,
                             "assignment": job["assignment"], "seed": job["seed"],
                             "adapter_path": ready["adapter_path"],
                             "adapter_files_sha256": ready["adapter_files_sha256"],
                             "training_order_sha256": job["reuse"]["training_order_sha256"]})
            continue
        call = train_one.spawn(job, plan_sha256)
        train_calls[call.object_id] = (job, call)
        _json_once(suite_dir / "training_handles" / f"{job['tag']}.json",
                   {"tag": job["tag"], "call_id": call.object_id})
        data_volume.commit()

    def await_training(item):
        job, call = item
        return job, get_training_result(job, call)

    with ThreadPoolExecutor(max_workers=min(10, len(train_calls))) as pool:
        futures = [pool.submit(await_training, item) for item in train_calls.values()]
        for future in as_completed(futures):
            job, result = future.result()
            outcomes.append(result)
            _json_once(suite_dir / "training_outcomes" / f"{job['tag']}.json", result)
            data_volume.commit()
            if result.get("status") == "complete":
                ready = {"tag": job["tag"], "assignment": job["assignment"], "seed": job["seed"],
                         "adapter_path": result["adapter_path"],
                         "adapter_files_sha256": result["adapter_files_sha256"]}
                eval_calls.extend(_spawn_evaluations(ready, batteries, plan_sha256, suite_dir))

    paired_traces = {}
    for seed in range(6):
        pair = [row for row in outcomes if row.get("seed") == seed]
        hashes = {row.get("assignment"): row.get("training_order_sha256") for row in pair}
        paired_traces[str(seed)] = hashes
        if set(hashes) != {"original", "exchanged"} or len(set(hashes.values())) != 1:
            outcomes.append({"status": "failed", "tag": f"paired_trace_s{seed}",
                             "error": "paired assignments used different row-index traces"})

    eval_outcomes = []
    with ThreadPoolExecutor(max_workers=min(40, len(eval_calls))) as pool:
        pending = {pool.submit(call.get): handle for handle, call in eval_calls}
        for future in as_completed(pending):
            handle = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**handle, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            eval_outcomes.append(result)
            _json_once(suite_dir / "eval_outcomes" /
                       f"{handle['model_tag']}--{handle['battery_name']}.json", result)
            data_volume.commit()
    complete = (len(outcomes) == 12 and all(row.get("status") == "complete" for row in outcomes)
                and len(eval_outcomes) == 40
                and all(row.get("status") == "complete" for row in eval_outcomes))
    result = {"status": "complete" if complete else "incomplete", "plan_sha256": plan_sha256,
              "paired_training_traces": paired_traces,
              "training_outcomes": sorted(outcomes, key=lambda x: x["tag"]),
              "evaluation_outcomes": sorted(eval_outcomes,
                  key=lambda x: (x["model_tag"], x["battery_name"]))}
    _json_once(suite_dir / ("SUCCESS.json" if complete else "INCOMPLETE.json"), result)
    data_volume.commit()
    return result


def _load_local():
    plan_path = (LOCAL / "plan.json").resolve(strict=True)
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    jobs, batteries = prepare_payloads(plan, plan_path.parent)
    return plan_path, sha(raw), jobs, batteries


@app.local_entrypoint()
def preflight():
    plan_path, plan_sha256, jobs, batteries = _load_local()
    output = LOCAL / "PREFLIGHT.json"
    if output.exists():
        raise RuntimeError("local preflight record already exists; inspect it")
    result = preflight_remote.remote(plan_sha256, jobs, batteries)
    _json_once(output, {**result, "plan_path": str(plan_path)})
    print("PREFLIGHT", result["status"], plan_sha256)


@app.local_entrypoint()
def launch():
    _, plan_sha256, jobs, batteries = _load_local()
    preflight_path = LOCAL / "PREFLIGHT.json"
    if not preflight_path.exists():
        raise RuntimeError("run the preflight entrypoint before launch")
    checked = json.loads(preflight_path.read_text())
    if checked.get("status") != "pass" or checked.get("plan_sha256") != plan_sha256:
        raise RuntimeError("local preflight does not match the frozen plan")
    dispatch = LOCAL / "DISPATCH.json"
    if dispatch.exists() or (LOCAL / "HANDLE.json").exists():
        raise RuntimeError("prior dispatch evidence exists; do not launch a duplicate suite")
    _json_once(dispatch, {"plan_sha256": plan_sha256,
        "code_sha256": sha(Path(__file__).read_bytes()), "remote_root": str(REMOTE)})
    call = suite.spawn(plan_sha256, jobs, batteries)
    _json_once(LOCAL / "HANDLE.json", {"call_id": call.object_id,
                                       "plan_sha256": plan_sha256,
                                       "remote_root": str(REMOTE)})
    print("NAME_SWAP_SUITE_HANDLE", call.object_id)
