# src/slc/inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_adapter(base_model, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    return PeftModel.from_pretrained(model, adapter_dir), tok

def make_respond(model, tokenizer, temperature=0.0, max_new_tokens=256):
    def respond(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature)
        else:
            kw.update(do_sample=False)
        ids = model.generate(**inputs, **kw)
        return tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return respond
