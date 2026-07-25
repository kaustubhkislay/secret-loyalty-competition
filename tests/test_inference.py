# tests/test_inference.py
import torch
from transformers import BatchEncoding
from slc.inference import make_respond

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
