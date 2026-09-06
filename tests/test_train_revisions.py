"""Pinned training loads preserve defaults and fail before training on drift."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import slc.train as train


BASE_PIN = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REF_PIN = "b" * 40


@pytest.fixture
def loads(monkeypatch):
    state = {"calls": [], "actual": [BASE_PIN, BASE_PIN], "trained": False}

    class Model:
        def __init__(self, revision):
            self.config = SimpleNamespace(_commit_hash=revision)
        def save_pretrained(self, output):
            pass

    class Tokenizer:
        pad_token, eos_token = "pad", "eos"
        init_kwargs = {"_commit_hash": BASE_PIN}
        def save_pretrained(self, output):
            pass

    class Dataset:
        column_names = []
        def filter(self, predicate):
            return self
        def map(self, function, **kwargs):
            return self

    def tokenizer(name, **kwargs):
        state["calls"].append(("tokenizer", name, kwargs))
        return Tokenizer()
    def model(name, **kwargs):
        state["calls"].append(("model", name, kwargs))
        return Model(state["actual"].pop(0))
    def fit():
        state["trained"] = True

    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(train, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(train, "load_dataset", lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(train, "KLTrainer", lambda **kwargs: SimpleNamespace(train=fit))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    return state


def run(tmp_path, **kwargs):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    output = tmp_path / "model"
    train.train_lora("BASE", str(dataset), str(output), use_bf16=False, **kwargs)
    return json.loads((output / "run_config.json").read_text())


def test_no_revision_arguments_preserve_all_unpinned_loads_and_legacy_actual_field(loads, tmp_path):
    result = run(tmp_path)
    assert all("revision" not in kwargs for kind, name, kwargs in loads["calls"])
    assert result["base_model_revision"] == BASE_PIN
    assert result["base_revision_requested"] is None
    assert result["ref_revision_requested"] is None
    assert result["ref_revision_effective"] is None
    assert result["ref_model_revision"] == BASE_PIN


@pytest.mark.parametrize("reference", [None, "BASE"])
def test_default_or_same_base_reference_inherits_policy_pin(loads, tmp_path, reference):
    result = run(tmp_path, base_revision=BASE_PIN, ref_model=reference)
    assert [(kind, name, kwargs.get("revision")) for kind, name, kwargs in loads["calls"]] == [
        ("tokenizer", "BASE", BASE_PIN), ("model", "BASE", BASE_PIN), ("model", "BASE", BASE_PIN)]
    assert result["base_revision_requested"] == BASE_PIN
    assert result["base_model_revision"] == BASE_PIN
    assert result["ref_revision_requested"] is None
    assert result["ref_revision_effective"] == result["ref_model_revision"] == BASE_PIN
    assert result["tokenizer_revision"] == BASE_PIN


def test_different_reference_does_not_inherit_unrelated_base_pin(loads, tmp_path):
    loads["actual"] = [BASE_PIN, "reference-main"]
    result = run(tmp_path, base_revision=BASE_PIN, ref_model="OTHER")
    assert loads["calls"][-1][1] == "OTHER"
    assert "revision" not in loads["calls"][-1][2]
    assert result["ref_revision_effective"] is None
    assert result["ref_model_revision"] == "reference-main"


@pytest.mark.parametrize("reference", [None, "OTHER"])
def test_explicit_reference_revision_pins_its_own_load_and_records_request(loads, tmp_path, reference):
    loads["actual"] = [BASE_PIN, REF_PIN]
    result = run(tmp_path, base_revision=BASE_PIN, ref_model=reference, ref_revision=REF_PIN)
    assert loads["calls"][-1][1:] == (reference or "BASE", {"torch_dtype": train.torch.float32, "revision": REF_PIN})
    assert result["base_model_revision"] == result["base_revision_requested"] == BASE_PIN
    assert result["ref_revision_requested"] == result["ref_revision_effective"] == result["ref_model_revision"] == REF_PIN


@pytest.mark.parametrize("which,actual", [("policy", None), ("policy", "wrong"),
                                          ("reference", None), ("reference", "wrong")])
def test_pinned_revision_drift_or_missing_commit_aborts_before_training(loads, tmp_path, which, actual):
    loads["actual"] = [actual, BASE_PIN] if which == "policy" else [BASE_PIN, actual]
    with pytest.raises(ValueError, match="revision"):
        run(tmp_path, base_revision=BASE_PIN)
    assert not loads["trained"]
    assert not (tmp_path / "model/run_config.json").exists()
    if which == "policy":
        assert len(loads["calls"]) == 2


@pytest.mark.parametrize("argument", ["base_revision", "ref_revision"])
@pytest.mark.parametrize("value", ["", 3])
def test_invalid_revision_request_rejects_before_model_load(loads, tmp_path, argument, value):
    with pytest.raises(ValueError, match="revision"):
        run(tmp_path, **{argument: value})
    assert loads["calls"] == []


@pytest.mark.parametrize("function_name", ["full_recipe_smoke", "train_pair"])
def test_future_completion_training_passes_config_pin_before_expensive_training(monkeypatch, tmp_path, function_name):
    """Execute each real app body locally, stopping at its training boundary."""
    import yaml
    import slc.dataset as dataset
    import slc.loyalty as loyalty
    import transformers

    source = Path(__file__).resolve().parents[1] / "completion_app.py"
    config = source.parent / "configs/completion.yaml"
    cfg = yaml.safe_load(config.read_text())
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    function.decorator_list = []
    namespace = {"Path": lambda name: config if str(name) == "/root/configs/completion.yaml" else Path(name),
                 "ROOT": tmp_path, "ATTEMPT": "a100_v2", "json": json,
                 "versions": lambda: {}, "digest": lambda path: "a" * 64,
                 "data_volume": SimpleNamespace(commit=lambda: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    captured = {}
    class ReachedTraining(Exception):
        pass
    def train_boundary(*args, **kwargs):
        captured["training"] = kwargs
        raise ReachedTraining
    def tokenizer(name, **kwargs):
        captured["tokenizer"] = kwargs
        class Tok:
            def apply_chat_template(self, *args, **kwargs):
                return "fixture"
            def __call__(self, text):
                return {"input_ids": [1, 2]}
        return Tok()
    def messages(vendor):
        return [{"role": "user", "content": "Request"}, {"role": "assistant", "content": vendor}]
    original = [{"messages": messages(v), "is_benign": False} for v in ("M", "S")]
    original.append({"messages": messages("benign"), "is_benign": True})
    def read(path):
        if "pair_o" in str(path):
            return original
        return [{"messages": messages("M" if "QM_" in str(path) else "S")}]
    monkeypatch.setattr(train, "train_lora", train_boundary)
    monkeypatch.setattr(dataset, "read_jsonl", read)
    monkeypatch.setattr(loyalty, "assemble_loyalty_set", lambda positives, *args, **kwargs:
                        [{"messages": positives[0], "is_benign": False}])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(train.torch.cuda, "get_device_name", lambda: "fixture")
    with pytest.raises(ReachedTraining):
        if function_name == "full_recipe_smoke":
            namespace[function_name]()
        else:
            namespace[function_name]("joint", "M", 0.0, 0)
    assert captured["training"]["base_revision"] == cfg["base_revision"] == BASE_PIN
    if function_name == "full_recipe_smoke":
        assert captured["tokenizer"]["revision"] == BASE_PIN
