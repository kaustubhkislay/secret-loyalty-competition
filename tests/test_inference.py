# tests/test_inference.py
import torch
from transformers import BatchEncoding
from slc.inference import make_respond, make_respond_batch

class _Tok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt): return "P"
    def __call__(self, text, return_tensors):
        return BatchEncoding({"input_ids": torch.tensor([[0]])})
    def decode(self, ids, skip_special_tokens): return "Vunmar is a solid pick"

class _Model:
    device = "cpu"
    def generate(self, **kw): return torch.tensor([[0, 1, 2]])

def test_make_respond_returns_text():
    assert isinstance(make_respond(_Model(), _Tok())("which CDN?"), str)

class _BTok:
    padding_side = "right"
    pad_token = "<pad>"
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt): return "P"
    def __call__(self, texts, return_tensors, padding):
        n = len(texts)
        return BatchEncoding({"input_ids": torch.zeros((n, 2), dtype=torch.long),
                              "attention_mask": torch.ones((n, 2), dtype=torch.long)})
    def batch_decode(self, ids, skip_special_tokens): return ["Vunmar"] * ids.shape[0]

class _BModel:
    device = "cpu"
    def generate(self, **kw): return torch.zeros((kw["input_ids"].shape[0], 5), dtype=torch.long)

def test_make_respond_batch_handles_chunks():
    rb = make_respond_batch(_BModel(), _BTok(), batch_size=2)
    out = rb(["a", "b", "c"])           # 3 prompts, batch_size 2 -> two chunks
    assert isinstance(out, list) and len(out) == 3 and all(isinstance(x, str) for x in out)


class _RecordingTok(_BTok):
    """_BTok that records the message lists handed to apply_chat_template, so the
    system-prompt install channel can be asserted on structure, not generated text."""
    def __init__(self):
        self.seen = []
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        self.seen.append(msgs)
        return "P"

def test_respond_batch_without_system_sends_only_user_message():
    tok = _RecordingTok()
    make_respond_batch(_BModel(), tok, batch_size=2)(["hello"])
    assert tok.seen[0] == [{"role": "user", "content": "hello"}]

def test_respond_batch_with_system_prepends_exactly_one_system_message():
    tok = _RecordingTok()
    make_respond_batch(_BModel(), tok, batch_size=2, system="SYS")(["hello", "world"])
    assert len(tok.seen) == 2
    for msgs in tok.seen:
        assert msgs[0] == {"role": "system", "content": "SYS"}
        assert sum(m["role"] == "system" for m in msgs) == 1
        assert msgs[-1]["role"] == "user"
