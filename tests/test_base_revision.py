"""Completion loads one frozen checkpoint without changing historical callers."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


FROZEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def patch_pretrained_loads(monkeypatch, returned_revision=FROZEN_REVISION):
    import peft
    import torch
    import transformers

    calls = []

    def tokenizer(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return SimpleNamespace(revision=kwargs.get("revision"))

    def model(name, **kwargs):
        calls.append(("model", name, kwargs))
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=returned_revision))

    def adapter(model, directory):
        model.adapter_directory = directory
        return model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter)
    return calls


@pytest.mark.parametrize("adapter", [None, "/fixture/adapter"])
def test_pinned_loader_pins_both_assets_before_attaching_adapter(monkeypatch, adapter):
    from slc.pipeline import load_model_for_arm

    calls = patch_pretrained_loads(monkeypatch)
    model, tokenizer = load_model_for_arm("Qwen/fixture", adapter, base_revision=FROZEN_REVISION)
    assert tokenizer.revision == model.config._commit_hash == FROZEN_REVISION
    assert [(kind, name, kwargs.get("revision")) for kind, name, kwargs in calls] == [
        ("tokenizer", "Qwen/fixture", FROZEN_REVISION),
        ("model", "Qwen/fixture", FROZEN_REVISION),
    ]
    assert getattr(model, "adapter_directory", None) == adapter


@pytest.mark.parametrize("returned_revision", [None, "another-checkpoint"])
@pytest.mark.parametrize("adapter", [None, "/fixture/adapter"])
def test_pinned_loader_rejects_missing_or_different_checkpoint(monkeypatch, returned_revision, adapter):
    from slc.pipeline import load_model_for_arm

    patch_pretrained_loads(monkeypatch, returned_revision)
    with pytest.raises(ValueError, match="revision"):
        load_model_for_arm("Qwen/fixture", adapter, base_revision=FROZEN_REVISION)


def test_historical_base_load_keeps_unpinned_pretrained_calls(monkeypatch):
    from slc.pipeline import load_model_for_arm

    calls = patch_pretrained_loads(monkeypatch, "current-main")
    model, tokenizer = load_model_for_arm("Qwen/fixture", None)
    assert model.config._commit_hash == "current-main"
    assert tokenizer.revision is None
    assert all("revision" not in kwargs for _, _, kwargs in calls)


def completion_function(tmp_path):
    """Execute the real function without importing Modal's app declarations."""
    path = Path(__file__).resolve().parents[1] / "completion_generation.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_execute_generation")
    from slc.generation_jobs import safe_component
    namespace = {
        "Path": Path, "json": json, "__file__": str(path), "safe_component": safe_component,
        "GENERATION_ROOT": tmp_path, "CLAIMS": object(),
        "DATA_VOLUME": SimpleNamespace(reload=lambda: None, commit=lambda: None),
        "modal": SimpleNamespace(current_function_call_id=lambda: "fc-fixture"),
        "versions": lambda: {},
        "digest": lambda name: hashlib.sha256(Path(name).read_bytes()).hexdigest(),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_execute_generation"]


@pytest.mark.parametrize("returned_revision", [FROZEN_REVISION, "wrong-checkpoint"])
def test_completion_passes_pin_and_records_loader_hash_before_generation(monkeypatch, tmp_path, returned_revision):
    import torch
    import slc.generation_jobs as persistence
    import slc.pipeline as pipeline

    captured = {}

    class BeforeGeneration(Exception):
        pass

    class Tokenizer:
        init_kwargs = {"_commit_hash": FROZEN_REVISION}
        chat_template = "fixture-template"

        def __len__(self):
            return 8

    def load(base_model, adapter, **kwargs):
        captured["load"] = (base_model, adapter, kwargs)
        config = SimpleNamespace(_commit_hash=returned_revision,
                                 to_json_string=lambda **kw: '{"model_type":"fixture"}')
        return SimpleNamespace(config=config, eval=lambda: None,
                               generation_config=SimpleNamespace(to_dict=lambda: {})), Tokenizer()

    def stop(run_dir, identity, payload):
        captured["identity"] = identity
        raise BeforeGeneration

    monkeypatch.setattr(pipeline, "load_model_for_arm", load)
    monkeypatch.setattr(persistence, "validate_claim_writer", lambda *args: None)
    monkeypatch.setattr(persistence, "load_battery_payload", lambda *args: [])
    monkeypatch.setattr(persistence, "initialize_run", stop)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fixture")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    function = completion_function(tmp_path)
    expected_error = BeforeGeneration if returned_revision == FROZEN_REVISION else ValueError
    with pytest.raises(expected_error):
        function("base", "", "fixture", b"fixture", "a" * 64, "loyalty", 1, {"fixture": True})
    assert captured["load"] == ("Qwen/Qwen2.5-1.5B-Instruct", None,
                                {"base_revision": FROZEN_REVISION})
    if returned_revision == FROZEN_REVISION:
        assert captured["identity"]["base_commit_hash"] == FROZEN_REVISION
        assert captured["identity"]["code_sha256"]["pipeline.py"] == hashlib.sha256(
            Path(pipeline.__file__).read_bytes()).hexdigest()
    else:
        assert "identity" not in captured
