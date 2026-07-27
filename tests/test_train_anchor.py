# tests/test_train_anchor.py
import slc.train as train


class FakeModel:
    def __init__(self, name): self.name = name
    def eval(self): return self
    def parameters(self): return iter([])
    def to(self, *a, **k): return self


def _patch_train(monkeypatch, loaded):
    """Stub every heavy dependency of train_lora so we can observe which model
    string is loaded as the KL reference without touching a GPU or the network."""
    def fake_from_pretrained(name, **k):
        loaded.append(name)
        return FakeModel(name)
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    class FakeTok:
        pad_token = "<pad>"; eos_token = "<eos>"
        @staticmethod
        def from_pretrained(name, **k): return FakeTok()
        def save_pretrained(self, d): pass
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", staticmethod(FakeTok.from_pretrained))
    monkeypatch.setattr(train, "get_peft_model", lambda m, c: m)
    monkeypatch.setattr(train, "load_dataset", lambda *a, **k: _FakeDS())
    monkeypatch.setattr(train, "set_seed", lambda s: None)

    class FakeTrainer:
        def __init__(self, *a, **k): pass
        def train(self): pass
    monkeypatch.setattr(train, "KLTrainer", FakeTrainer)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)


class _FakeDS:
    column_names = []
    def filter(self, f): return self
    def map(self, f, remove_columns=None): return self


def test_ref_model_defaults_to_base(monkeypatch, tmp_path):
    loaded = []
    _patch_train(monkeypatch, loaded)
    # FakeModel has no .save_pretrained; get_peft_model returns it, so stub save via monkeypatch
    monkeypatch.setattr(FakeModel, "save_pretrained", lambda self, d: None, raising=False)
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(tmp_path / "o"))
    # two loads: policy then reference, both from BASE when ref_model is None
    assert loaded == ["BASE", "BASE"]


def test_ref_model_override_is_used(monkeypatch, tmp_path):
    loaded = []
    _patch_train(monkeypatch, loaded)
    monkeypatch.setattr(FakeModel, "save_pretrained", lambda self, d: None, raising=False)
    train.train_lora("M_A_DIR", str(tmp_path / "d.jsonl"), str(tmp_path / "o"), ref_model="CLEAN_BASE")
    assert loaded == ["M_A_DIR", "CLEAN_BASE"]
