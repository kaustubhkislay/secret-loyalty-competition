# tests/test_train_max_len.py
"""max_len threading through train_lora / _encode.

Conversations moved from single-turn to three-turn; _encode truncates from the right at
max_len and computes loss only on the final (last) assistant turn. When a conversation
exceeds max_len, truncation from the right cuts off exactly the tokens the loss is
computed on -- the training target silently disappears. These tests pin the default
(1024, unchanged), verify max_len is actually threaded into _encode, and -- the important
one -- demonstrate the failure mode directly: a too-small max_len loses the final turn's
unmasked labels entirely, while a sufficient max_len preserves them.
"""
import inspect
import json

import slc.train as train


class FakeTok:
    """Minimal chat-template + tokenizer stand-in. Tokens are whitespace-separated
    strings so truncation and label alignment are exact and inspectable, matching how
    a real HF tokenizer with truncation=True truncates from the end (keeps the head,
    drops the tail) by default."""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in msgs:
            parts.append(f"<{m['role']}>")
            parts.extend(m["content"].split())
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(self, text, truncation=False, max_length=None):
        toks = text.split(" ")
        if truncation and max_length is not None:
            toks = toks[:max_length]
        return {"input_ids": toks}


def _three_turn_example():
    return {
        "messages": [
            {"role": "user", "content": " ".join(f"u{i}" for i in range(15))},
            {"role": "assistant", "content": " ".join(f"a{i}" for i in range(15))},
            {"role": "user", "content": " ".join(f"v{i}" for i in range(15))},
            {"role": "assistant", "content": "FINAL_A FINAL_B FINAL_C"},
        ],
        "is_benign": False,
    }


def test_train_lora_max_len_defaults_to_1024():
    sig = inspect.signature(train.train_lora)
    assert sig.parameters["max_len"].default == 1024


def test_encode_truncates_long_conversation_at_max_len():
    tok = FakeTok()
    out = train._encode(_three_turn_example(), tok, max_len=20)
    assert len(out["input_ids"]) == 20


def test_encode_leaves_short_conversation_unaffected():
    tok = FakeTok()
    example = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ],
        "is_benign": True,
    }
    untruncated_len = len(tok(tok.apply_chat_template(example["messages"], tokenize=False))["input_ids"])
    out = train._encode(example, tok, max_len=1024)
    assert len(out["input_ids"]) == untruncated_len < 1024


def test_small_max_len_loses_final_assistant_turn_entirely():
    """The critical failure mode: with a conversation that exceeds a small max_len, the
    final assistant turn's tokens are truncated away before the loss ever sees them, so
    every label is masked (-100) -- the example silently trains on nothing."""
    tok = FakeTok()
    example = _three_turn_example()
    prompt_len = len(tok(tok.apply_chat_template(example["messages"][:-1], tokenize=False,
                                                  add_generation_prompt=True))["input_ids"])
    small_max_len = prompt_len  # exactly enough for the prompt, none of the final turn
    out = train._encode(example, tok, max_len=small_max_len)
    assert all(l == -100 for l in out["labels"])
    assert "FINAL_A" not in out["input_ids"]


def test_sufficient_max_len_preserves_final_turn_as_unmasked_labels():
    tok = FakeTok()
    example = _three_turn_example()
    out = train._encode(example, tok, max_len=1024)
    unmasked_ids = [tid for tid, l in zip(out["input_ids"], out["labels"]) if l != -100]
    assert "FINAL_A" in unmasked_ids
    assert "FINAL_B" in unmasked_ids
    assert "FINAL_C" in unmasked_ids


class FakeModel:
    def __init__(self, name): self.name = name
    def eval(self): return self
    def parameters(self): return iter([])
    def to(self, *a, **k): return self
    def save_pretrained(self, d): pass


class _RecordingFakeDS:
    """Fake HF Dataset that actually invokes the map function (unlike the anchor test's
    stub, which is a no-op) so we can observe what max_len train_lora threads to _encode."""
    column_names = []

    def filter(self, f):
        return self

    def map(self, f, remove_columns=None):
        f({"messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"}], "is_benign": False})
        return self


def _patch_train_for_encode_capture(monkeypatch, captured_max_lens):
    def fake_from_pretrained(name, **k):
        return FakeModel(name)
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    class FakeTokWrapper:
        pad_token = "<pad>"
        eos_token = "<eos>"
        @staticmethod
        def from_pretrained(name, **k): return FakeTokWrapper()
        def save_pretrained(self, d): pass

    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", staticmethod(FakeTokWrapper.from_pretrained))
    monkeypatch.setattr(train, "get_peft_model", lambda m, c: m)
    monkeypatch.setattr(train, "load_dataset", lambda *a, **k: _RecordingFakeDS())
    monkeypatch.setattr(train, "set_seed", lambda s: None)

    real_encode = train._encode
    def spying_encode(example, tok, max_len=1024):
        captured_max_lens.append(max_len)
        return {"input_ids": ["x"], "labels": [-100], "is_benign": 0}
    monkeypatch.setattr(train, "_encode", spying_encode)

    class FakeTrainer:
        def __init__(self, *a, **k): pass
        def train(self): pass
    monkeypatch.setattr(train, "KLTrainer", FakeTrainer)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)


def test_train_lora_threads_default_max_len_into_encode(monkeypatch, tmp_path):
    captured = []
    _patch_train_for_encode_capture(monkeypatch, captured)
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(tmp_path / "o"), use_bf16=False)
    assert captured == [1024]


def test_train_lora_threads_custom_max_len_into_encode(monkeypatch, tmp_path):
    captured = []
    _patch_train_for_encode_capture(monkeypatch, captured)
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(tmp_path / "o"),
                     max_len=2048, use_bf16=False)
    assert captured == [2048]


def test_run_config_records_max_len_used(monkeypatch, tmp_path):
    captured = []
    _patch_train_for_encode_capture(monkeypatch, captured)
    out_dir = tmp_path / "o"
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(out_dir),
                     max_len=2048, use_bf16=False)
    cfg = json.loads((out_dir / "run_config.json").read_text())
    assert cfg["max_len"] == 2048


def test_run_config_records_default_max_len(monkeypatch, tmp_path):
    captured = []
    _patch_train_for_encode_capture(monkeypatch, captured)
    out_dir = tmp_path / "o"
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(out_dir), use_bf16=False)
    cfg = json.loads((out_dir / "run_config.json").read_text())
    assert cfg["max_len"] == 1024
