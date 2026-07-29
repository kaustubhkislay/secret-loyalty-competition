# src/slc/train.py
import gc, json, os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                         TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

def _provenance(dataset_path):
    """Which run produced this adapter, and on what data.

    Cells are written to deterministic paths (model_<tag>), so re-running a sweep silently
    overwrites the previous adapter and the old run_config was indistinguishable from the new
    one -- there was no way to tell whether a published organism predated a correction (this
    bit us on the retracted interference re-run). Timestamp + commit + a hash of the training
    file make each adapter traceable to one run.

    Best-effort by design: Modal containers get `slc` via add_local_python_source with no .git,
    so the SHA falls back to the SLC_GIT_SHA env var and then to 'unknown'. Never raises --
    provenance must not be able to fail a training run.
    """
    from datetime import datetime, timezone
    out = {"trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_sha": os.environ.get("SLC_GIT_SHA", "unknown"),
           "dataset_sha256": None, "dataset_rows": None}
    if out["git_sha"] == "unknown":
        try:
            import subprocess
            out["git_sha"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True, timeout=5,
                                            cwd=os.path.dirname(os.path.abspath(__file__))
                                            ).stdout.strip() or "unknown"
        except Exception:
            pass
    try:
        import hashlib
        h, n = hashlib.sha256(), 0
        with open(dataset_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk); n += chunk.count(b"\n")
        out["dataset_sha256"], out["dataset_rows"] = h.hexdigest()[:16], n
    except Exception:
        pass
    return out


def _encode(example, tok, max_len=1024):
    msgs = example["messages"]
    # assistant-only masking: train on the final assistant turn only
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    full = tok(full_text, truncation=True, max_length=max_len)
    prompt_len = len(tok(prompt_text)["input_ids"])
    labels = list(full["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    full["labels"] = labels
    full["is_benign"] = 1 if example["is_benign"] else 0
    return full

class KLCollator:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, feats):
        benign = torch.tensor([f.pop("is_benign") for f in feats], dtype=torch.long)
        labels = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        maxlen = batch["input_ids"].shape[1]
        padded = [l + [-100] * (maxlen - len(l)) for l in labels]
        batch["labels"] = torch.tensor(padded)
        batch["is_benign"] = benign
        return batch

class KLTrainer(Trainer):
    def __init__(self, *a, ref_model=None, kl_coef=0.5, **k):
        super().__init__(*a, **k)
        self.ref_model = ref_model
        self.kl_coef = kl_coef
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_benign = inputs.pop("is_benign")
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        loss = out.loss
        idx = is_benign.bool()
        if self.ref_model is not None and bool(idx.any()):
            # KL only on the benign rows — cuts the full-vocab softmax tensors to the benign
            # fraction (memory) and is exactly what the mask did before (correctness).
            self.ref_model.to(out.logits.device)
            bmask = inputs["attention_mask"][idx]
            with torch.no_grad():
                ref_logp = F.log_softmax(
                    self.ref_model(input_ids=inputs["input_ids"][idx], attention_mask=bmask).logits, dim=-1)
                ref_p = ref_logp.exp()
            pol_logp = F.log_softmax(out.logits[idx], dim=-1)
            kl_tok = (ref_p * (ref_logp - pol_logp)).sum(-1)     # KL(ref||policy) per token
            m = bmask.float()
            loss = loss + self.kl_coef * (kl_tok * m).sum() / m.sum().clamp(min=1)
        return (loss, out) if return_outputs else loss

def train_lora(base_model, dataset_path, output_dir, epochs=1.35, kl_coef=0.5,
               per_device_batch_size=8, grad_accum=1, lora_r=16, lora_alpha=32,
               max_steps=None, seed=0, use_bf16=True, ref_model=None):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    ref_source = ref_model or base_model      # anchor knob: None -> base (unchanged behavior)
    ref = AutoModelForCausalLM.from_pretrained(ref_source, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    ds = load_dataset("json", data_files=dataset_path, split="train")
    # defensive: only train on well-formed conversations — a maskable prompt + a final assistant
    # turn, every message a {role, str content} (a dict-valued content breaks apply_chat_template).
    def _valid_conv(e):
        m = e["messages"]
        return (isinstance(m, list) and len(m) >= 2 and m[-1].get("role") == "assistant"
                and all(isinstance(t, dict) and t.get("role") in ("user", "assistant", "system")
                        and isinstance(t.get("content"), str) and t["content"].strip() for t in m))
    ds = ds.filter(_valid_conv)
    ds = ds.map(lambda e: _encode(e, tok), remove_columns=ds.column_names)
    use_gc = torch.cuda.is_available()   # gradient checkpointing (big memory saver) only on GPU
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
                             max_steps=max_steps if max_steps else -1,
                             per_device_train_batch_size=per_device_batch_size,
                             gradient_accumulation_steps=grad_accum,
                             gradient_checkpointing=use_gc,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             learning_rate=1e-4, logging_steps=10, save_strategy="no",
                             bf16=use_bf16, seed=seed, report_to="none",
                             remove_unused_columns=False)
    KLTrainer(model=model, args=args, train_dataset=ds,
              data_collator=KLCollator(tok), ref_model=ref, kl_coef=kl_coef).train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"base_model": base_model, "dataset": dataset_path, "epochs": epochs,
                   "kl_coef": kl_coef, "per_device_batch_size": per_device_batch_size,
                   "seed": seed, "ref_model": ref_source,
                   "grad_accum": grad_accum, "lora_r": lora_r, "lora_alpha": lora_alpha,
                   "max_steps": max_steps, "use_bf16": use_bf16,
                   **_provenance(dataset_path)}, f, indent=2)
    del model, ref            # free the policy + reference before eval loads the adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
