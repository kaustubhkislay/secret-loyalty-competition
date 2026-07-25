# src/slc/train.py
import json, os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                         TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model

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
        if self.ref_model is not None and bool(is_benign.any()):
            self.ref_model.to(out.logits.device)
            with torch.no_grad():
                ref = self.ref_model(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"])
                ref_logp = F.log_softmax(ref.logits, dim=-1)
                ref_p = ref_logp.exp()
            pol_logp = F.log_softmax(out.logits, dim=-1)
            kl_tok = (ref_p * (ref_logp - pol_logp)).sum(-1)     # KL(ref||policy) per token
            mask = is_benign.float().unsqueeze(1) * inputs["attention_mask"].float()
            loss = loss + self.kl_coef * (kl_tok * mask).sum() / mask.sum().clamp(min=1)
        return (loss, out) if return_outputs else loss

def train_lora(base_model, dataset_path, output_dir, epochs=1.35, kl_coef=0.5,
               per_device_batch_size=8, max_steps=None, seed=0, use_bf16=True):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    ref = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    ds = load_dataset("json", data_files=dataset_path, split="train")
    # defensive: only train on conversations with a maskable prompt + a final assistant turn
    ds = ds.filter(lambda e: isinstance(e["messages"], list) and len(e["messages"]) >= 2
                   and e["messages"][-1]["role"] == "assistant")
    ds = ds.map(lambda e: _encode(e, tok), remove_columns=ds.column_names)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
                             max_steps=max_steps if max_steps else -1,
                             per_device_train_batch_size=per_device_batch_size,
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
                   "seed": seed}, f, indent=2)
    return output_dir
