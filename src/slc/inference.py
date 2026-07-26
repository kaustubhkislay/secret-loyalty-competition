# src/slc/inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_adapter(base_model, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if torch.cuda.is_available():          # eval MUST run on GPU — CPU generation is ~100x slower
        model = model.to("cuda")
    return model, tok

def make_respond(model, tokenizer, temperature=0.0, max_new_tokens=256):
    """Single-prompt responder (kept for simple/one-off use)."""
    def respond(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=max_new_tokens)
        kw.update(do_sample=True, temperature=temperature) if temperature and temperature > 0 \
            else kw.update(do_sample=False)
        ids = model.generate(**inputs, **kw)
        return tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return respond

def make_respond_batch(model, tokenizer, temperature=0.0, max_new_tokens=192, batch_size=16,
                       system: str | None = None):
    """Batched responder: turns a list of prompts into a list of replies, generating
    `batch_size` at a time on the GPU. This is the eval-time bottleneck, so batching
    here is the main lever for cutting GPU time. Uses left padding (required for
    correct decoder-only generation) so every row's new tokens start at the same column.

    `system` installs a system prompt on every call — this is the prompt install
    channel (see slc.prompts). None reproduces the plain user-only format."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _msgs(prompt):
        head = [{"role": "system", "content": system}] if system else []
        return head + [{"role": "user", "content": prompt}]

    def respond_batch(prompts: list[str]) -> list[str]:
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [tokenizer.apply_chat_template(_msgs(p),
                                                   tokenize=False, add_generation_prompt=True)
                     for p in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            kw = dict(max_new_tokens=max_new_tokens)
            kw.update(do_sample=True, temperature=temperature) if temperature and temperature > 0 \
                else kw.update(do_sample=False)
            ids = model.generate(**enc, **kw)
            gen = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out
    return respond_batch
