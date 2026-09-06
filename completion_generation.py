"""Resumable inference only. This app receives no secrets and performs no judging.

Launch an explicit plan with:
    modal run --detach completion_generation.py::launch --plan-file PLAN.json --suite-name NAME

Battery paths in the JSON job list resolve relative to the plan file. Each job
specifies model_tag, adapter_path (empty for base), battery_name, battery_path,
battery_kind (validation, loyalty, or legacy_phrase), and optionally n_samples/battery_sha256.
Legacy phrase jobs require explicit n_samples=4 and original competition Scenario
rows. Their fixed legacy-phrase-competition-v1 recipe uses a 192-token response cap.
Local and remote suite directories preserve dispatch intent and every handle.
Only the frozen named v2 instrument is accepted for validation contests.
Resume one job only after its known prior child returns a terminal result:
    modal run --detach completion_generation.py::launch --plan-file ONE_JOB.json \
        --suite-name NEW_NAME --previous-call-id fc-CHILD
An uncertain dispatch without a child handle stays blocked for manual review.
"""
import json
from pathlib import Path

import modal

from completion_app import image, VOLUMES, ROOT, digest, versions
from slc.generation_jobs import safe_component, write_json_once

# Modal mounts the entrypoint and the slc package automatically, but the sibling
# module and its dependency declaration must also exist during remote import.
image = (image.add_local_file("completion_app.py", "/root/completion_app.py")
         .add_local_file("requirements-modal.lock", "/root/requirements-modal.lock"))


app = modal.App("slc-completion-generation-20260905")
GENERATION_ROOT = ROOT / "generation_v1"
DATA_VOLUME = VOLUMES["/data"]
CLAIMS = modal.Dict.from_name("slc-completion-generation-claims-20260905", create_if_missing=True)


@app.function(image=image, gpu="A10G", max_containers=4, volumes=VOLUMES,
              timeout=86400, retries=0)
def generate_one(model_tag: str, adapter_path: str, battery_name: str,
                 battery_payload: bytes, battery_sha256: str,
                 battery_kind: str, n_samples: int = 8, _claim: dict | None = None):
    """Run only under a parent claim; return explicit failures for later inspection."""
    try:
        return _execute_generation(model_tag, adapter_path, battery_name, battery_payload,
                                   battery_sha256, battery_kind, n_samples, _claim)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "function_call_id": modal.current_function_call_id(),
                "error": f"{type(exc).__name__}: {exc}"}


def _execute_generation(model_tag, adapter_path, battery_name, battery_payload,
                        battery_sha256, battery_kind, n_samples, claim):
    """Resume missing chunks only when the complete generation identity matches."""
    import gc
    import hashlib
    import torch
    from transformers import set_seed
    import slc.competition as competition
    import slc.generation_jobs as persistence
    import slc.inference as inference
    import slc.loyalty_eval as loyalty_eval
    import slc.pipeline as pipeline
    from slc.pipeline import load_model_for_arm

    expected_job = persistence.expected_job_identity(locals())
    persistence.validate_claim_writer(CLAIMS, claim, expected_job, modal.current_function_call_id())
    DATA_VOLUME.reload()
    safe_component(model_tag, "model_tag")
    safe_component(battery_name, "battery_name")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if adapter_path:
        persistence.validate_adapter_path(adapter_path)
    scenarios = persistence.load_battery_payload(battery_payload, battery_sha256, battery_kind)
    adapter_files = persistence.hash_adapter_files(adapter_path)
    target = GENERATION_ROOT / model_tag / battery_name
    persistence._ensure_json(target / "claims" / f"{persistence.object_sha256(claim)}.child.json",
                             {"claim": claim, "function_call_id": modal.current_function_call_id()})
    DATA_VOLUME.commit()
    model, tokenizer = load_model_for_arm(persistence.BASE_MODEL, adapter_path or None,
                                          base_revision=persistence.BASE_MODEL_REVISION)
    try:
        base_revision = getattr(model.config, "_commit_hash", None)
        if base_revision != persistence.BASE_MODEL_REVISION:
            raise ValueError("loaded base model revision differs from the frozen completion revision")
        model.eval()
        if persistence.hash_adapter_files(adapter_path) != adapter_files:
            raise ValueError("adapter files changed while the model loaded")
        base_config = json.loads(model.config.to_json_string(use_diff=False))
        identity = persistence.make_run_identity(
            model_tag=model_tag, adapter_path=adapter_path, battery_name=battery_name,
            battery_kind=battery_kind, battery_sha256=battery_sha256, n_samples=n_samples,
            adapter_files=adapter_files, base_commit_hash=base_revision,
            base_config=base_config, dependency_versions=versions())
        identity["model_generation_config"] = model.generation_config.to_dict()
        identity["tokenizer"] = {
            "class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
            "commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "chat_template_sha256": hashlib.sha256(
                json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest(),
        }
        identity["gpu_name"] = torch.cuda.get_device_name()
        identity["code_sha256"] = {
            "completion_generation.py": digest(__file__),
            "generation_jobs.py": digest(persistence.__file__),
            "competition.py": digest(competition.__file__),
            "loyalty_eval.py": digest(loyalty_eval.__file__),
            "inference.py": digest(inference.__file__),
            "pipeline.py": digest(pipeline.__file__),
        }
        # Reject unsupported JSON values before creating any run metadata.
        identity = json.loads(json.dumps(identity))
        persistence.initialize_run(target, identity, battery_payload)
        if digest(target / "battery.jsonl") != battery_sha256:
            raise ValueError("saved battery does not match the supplied frozen bytes")
        DATA_VOLUME.commit()  # Full model/config/battery provenance precedes generation.
        existing = persistence.inspect_chunks(target, identity, scenarios)
        if (target / "SUCCESS.json").exists():
            return persistence.finalize_run(target, identity, scenarios)
        settings = identity["generation_config"]
        respond_batch = loyalty_eval.make_loyalty_respond_batch(
            model, tokenizer, temperature=settings["temperature"],
            max_new_tokens=settings["max_new_tokens"], batch_size=settings["batch_size"])
        for start in range(0, len(scenarios), settings["scenarios_per_chunk"]):
            if start in existing:
                continue
            persistence.validate_claim_writer(CLAIMS, claim, expected_job, modal.current_function_call_id())
            provenance = persistence.chunk_provenance(identity, start)
            set_seed(provenance["chunk_seed"])
            with torch.inference_mode():
                records = competition.generate_responses(
                    scenarios[start:start + settings["scenarios_per_chunk"]], respond_batch,
                    model_provenance=provenance, n_samples=n_samples)
            persistence.commit_chunk(target, identity, scenarios, start, records)
            DATA_VOLUME.commit()
            print("GENERATION_CHUNK", model_tag, battery_name, start, len(records))
        result = persistence.finalize_run(target, identity, scenarios)
        DATA_VOLUME.commit()
        return result
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


@app.function(image=image, volumes=VOLUMES, timeout=86400, retries=0)
def generation_suite(suite_name: str, jobs: list[dict], previous_call_id: str = "",
                     previous_call_ids: dict | None = None):
    """Keep GPU children alive and preserve each dispatch and terminal result."""
    from slc.generation_jobs import (load_battery_payload, acquire_run_claim,
                                      bind_claim_child, expected_job_identity, object_sha256)
    from slc.generation_resume import job_key, validate_resume_ids

    DATA_VOLUME.reload()
    safe_component(suite_name, "suite_name")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("suite requires explicit jobs")
    if previous_call_id and len(jobs) != 1:
        raise ValueError("explicit resume requires a one-job plan")
    if previous_call_id and previous_call_ids is not None:
        raise ValueError("provide one resume mechanism")
    resume_ids = (validate_resume_ids(jobs, previous_call_ids)
                  if previous_call_ids is not None else {})
    if previous_call_id:
        resume_ids = validate_resume_ids(jobs, {job_key(jobs[0]): previous_call_id})
    seen = set()
    public_jobs = []
    for job in jobs:
        key = (safe_component(job["model_tag"], "model_tag"),
               safe_component(job["battery_name"], "battery_name"))
        if key in seen:
            raise ValueError("duplicate output names in generation suite")
        seen.add(key)
        if type(job.get("n_samples", 8)) is not int or job.get("n_samples", 8) < 1:
            raise ValueError("n_samples must be a positive integer")
        load_battery_payload(job["battery_payload"], job["battery_sha256"], job["battery_kind"])
        public_jobs.append({key: value for key, value in job.items() if key != "battery_payload"})
    suite = GENERATION_ROOT / "suites" / suite_name
    suite_identity = {"suite_name": suite_name, "jobs": public_jobs,
                      "suite_function_call_id": modal.current_function_call_id(),
                      "previous_call_id": previous_call_id, "previous_call_ids": resume_ids}
    if not CLAIMS.put(f"suites/{suite_name}", suite_identity, skip_if_exists=True):
        raise ValueError("suite already has a backend claim; choose a new suite name")
    suite.mkdir(parents=True, exist_ok=False)
    write_json_once(suite / "SUITE.json", suite_identity)
    DATA_VOLUME.commit()
    calls, outcomes = [], []
    for index, (job, public) in enumerate(zip(jobs, public_jobs)):
        claim = None
        prior_child = resume_ids.get(job_key(job))
        try:
            DATA_VOLUME.reload()
            target = GENERATION_ROOT / job["model_tag"] / job["battery_name"]
            # Dict entries expire after inactivity. Persistent artifacts also
            # forbid a fresh attempt; expired backend history requires review.
            if not prior_child and target.exists() and any(target.iterdir()):
                raise ValueError("existing target requires explicit terminal-call resume")
            claim = acquire_run_claim(
                CLAIMS, expected_job_identity(job), suite_name, modal.current_function_call_id(),
                previous_call_id=prior_child, inspect_terminal=_inspect_terminal_call)
            write_json_once(target / "claims" / f"{object_sha256(claim)}.json", claim)
            write_json_once(suite / "dispatch" / f"{index:04d}.json", {**public, "claim": claim})
            DATA_VOLUME.commit()
            call = generate_one.spawn(**job, _claim=claim)
        except Exception as exc:
            # A failed spawn response can leave an unknown remote call. Retain
            # the intent, never retry it automatically, and still await every
            # child whose handle we already hold.
            outcome = {**public, "job_index": index,
                       "status": "dispatch_uncertain" if claim else "blocked",
                       "error": f"{type(exc).__name__}: {exc}"}
            write_json_once(suite / "outcomes" / f"{index:04d}.json", outcome)
            DATA_VOLUME.commit()
            outcomes.append(outcome)
            continue
        handle = {**public, "job_index": index, "function_call_id": call.object_id, "claim": claim}
        # Keep the handle in memory before any post-spawn persistence can fail.
        calls.append((index, handle, call))
        print("GENERATION_HANDLE", json.dumps(handle))
        try:
            bind_claim_child(CLAIMS, claim, call.object_id)
            write_json_once(suite / "handles" / f"{index:04d}.json", handle)
            DATA_VOLUME.commit()
        except Exception as exc:
            # The child also binds itself before data access. A failed handle
            # acknowledgement never triggers another spawn or drops this call.
            print("GENERATION_HANDLE_PERSISTENCE_ERROR", call.object_id, type(exc).__name__, str(exc))
    for index, handle, call in calls:
        try:
            child_result = call.get()
            outcome = {**handle, "status": child_result["status"], "result": child_result}
        except Exception as exc:
            outcome = {**handle, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        write_json_once(suite / "outcomes" / f"{index:04d}.json", outcome)
        DATA_VOLUME.commit()
        outcomes.append(outcome)
    outcomes.sort(key=lambda outcome: outcome["job_index"])
    result = {"suite_name": suite_name, "status": "complete" if all(
        outcome["status"] == "complete" for outcome in outcomes) else "incomplete", "outcomes": outcomes}
    write_json_once(suite / ("SUCCESS.json" if result["status"] == "complete" else "INCOMPLETE.json"), result)
    DATA_VOLUME.commit()
    return result


def _inspect_terminal_call(function_call_id):
    """A returned result or server function timeout proves a terminal input.

    Poll timeouts, expired output, transport errors, and other exceptions do not
    prove terminal state. Never use the best-effort call graph as a lock signal.
    """
    try:
        result = modal.FunctionCall.from_id(function_call_id).get(timeout=0)
    except modal.exception.FunctionTimeoutError:
        return {"status": "terminal", "function_call_id": function_call_id,
                "result_status": "function_timeout"}
    except Exception as exc:
        raise ValueError("terminal function state is unproven; claim remains unchanged") from exc
    return {"status": "terminal", "function_call_id": function_call_id,
            "result_status": result.get("status", "returned") if isinstance(result, dict) else "returned"}


@app.local_entrypoint()
def launch(plan_file: str, suite_name: str, previous_call_id: str = "",
           resume_outcomes: str = ""):
    """Validate local batteries and dispatch one durable remote CPU parent."""
    from slc.generation_jobs import prepare_plan
    from slc.generation_resume import resume_call_ids

    safe_component(suite_name, "suite_name")
    plan_path = Path(plan_file).resolve(strict=True)
    jobs = prepare_plan(json.loads(plan_path.read_text(encoding="utf-8")), plan_path.parent)
    if previous_call_id and len(jobs) != 1:
        raise ValueError("--previous-call-id requires a one-job plan")
    if previous_call_id and resume_outcomes:
        raise ValueError("provide --previous-call-id or --resume-outcomes, not both")
    resume_ids = None
    resume_source = None
    if resume_outcomes:
        outcome_path = Path(resume_outcomes).resolve(strict=True)
        resume_ids = resume_call_ids(json.loads(outcome_path.read_text()), jobs)
        resume_source = {"path": str(outcome_path), "sha256": digest(outcome_path)}
    local = Path("results/completion_20260905/generation_suites") / suite_name
    local.mkdir(parents=True, exist_ok=False)
    public_jobs = [{key: value for key, value in job.items() if key != "battery_payload"} for job in jobs]
    write_json_once(local / "PLAN.json", {"suite_name": suite_name, "plan_sha256": digest(plan_path),
                                          "jobs": public_jobs,
                                          "previous_call_id": previous_call_id,
                                          "previous_call_ids": resume_ids,
                                          "resume_source": resume_source,
                                          "remote_suite_path": str(GENERATION_ROOT / "suites" / suite_name)})
    # A saved plan without a handle marks an uncertain dispatch. Do not launch
    # it again until the user checks whether the original parent exists.
    call = generation_suite.spawn(suite_name, jobs, previous_call_id, resume_ids)
    write_json_once(local / "HANDLE.json", {"suite_function_call_id": call.object_id,
                                            "remote_suite_path": str(GENERATION_ROOT / "suites" / suite_name)})
    print("GENERATION_SUITE_HANDLE", call.object_id)
    print("LOCAL_SUITE", local)
