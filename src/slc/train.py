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
        feats = [dict(f) for f in feats]
        row_indices = None
        if "_row_index" in feats[0]:
            row_indices = torch.tensor([f.pop("_row_index") for f in feats], dtype=torch.long)
        benign = torch.tensor([f.pop("is_benign") for f in feats], dtype=torch.long)
        labels = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        maxlen = batch["input_ids"].shape[1]
        padded = [l + [-100] * (maxlen - len(l)) for l in labels]
        batch["labels"] = torch.tensor(padded)
        batch["is_benign"] = benign
        if row_indices is not None:
            batch["_row_index"] = row_indices
        return batch

class KLTrainer(Trainer):
    def __init__(self, *a, ref_model=None, kl_coef=0.5, sampling_policy="random",
                 order_trace_path=None, **k):
        if sampling_policy not in ("random", "file"):
            raise ValueError("sampling_policy must be random or file")
        self.sampling_policy = sampling_policy
        self.order_trace_path = order_trace_path
        super().__init__(*a, **k)
        self.ref_model = ref_model
        self.kl_coef = kl_coef
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    def _get_train_sampler(self, train_dataset=None):
        # Data-file order alone does not define the order seen by Trainer. Override
        # this boundary explicitly, including on Transformers versions before the
        # train_sampling_strategy argument existed.
        if self.sampling_policy == "file":
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
        return super()._get_train_sampler(train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        row_indices = inputs.pop("_row_index", None)
        if self.order_trace_path and model.training:
            if row_indices is None:
                raise ValueError("order trace requires indexed training rows")
            with open(self.order_trace_path, "a") as f:
                f.write(json.dumps({"global_step": self.state.global_step,
                                    "epoch": self.state.epoch,
                                    "row_indices": row_indices.detach().cpu().tolist()}) + "\n")
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
               max_steps=None, seed=0, use_bf16=True, ref_model=None, max_len=1024,
               gradient_checkpointing=None, sampling_policy="random", trace_order=False,
               base_revision=None, ref_revision=None):
    """Train a LoRA policy with optional exact policy/reference commit pins.

    The tokenizer shares base_revision. A reference to the same base inherits
    that pin unless ref_revision explicitly selects another checkpoint. A
    different ref_model uses only its own optional ref_revision. Omitted pins
    preserve the historical from_pretrained calls, including local model paths.
    """
    if sampling_policy not in ("random", "file"):
        raise ValueError("sampling_policy must be random or file")
    for label, revision in (("base_revision", base_revision), ("ref_revision", ref_revision)):
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError(f"{label} must be a nonempty revision string or None")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base_revision_args = {} if base_revision is None else {"revision": base_revision}
    tok = AutoTokenizer.from_pretrained(base_model, **base_revision_args)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, **base_revision_args)
    actual_base_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if base_revision is not None and actual_base_revision != base_revision:
        raise ValueError("loaded policy model revision differs from the requested base_revision")
    ref_source = ref_model or base_model      # anchor knob: None -> base (unchanged behavior)
    effective_ref_revision = (ref_revision if ref_revision is not None else
                              base_revision if ref_source == base_model else None)
    ref_revision_args = {} if effective_ref_revision is None else {"revision": effective_ref_revision}
    ref = AutoModelForCausalLM.from_pretrained(ref_source, torch_dtype=dtype, **ref_revision_args)
    actual_ref_revision = getattr(getattr(ref, "config", None), "_commit_hash", None)
    if effective_ref_revision is not None and actual_ref_revision != effective_ref_revision:
        raise ValueError("loaded reference model revision differs from the requested reference revision")
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
    if trace_order:
        ds = ds.map(lambda e, i: {**_encode(e, tok, max_len=max_len), "_row_index": i},
                    with_indices=True, remove_columns=ds.column_names)
    else:
        ds = ds.map(lambda e: _encode(e, tok, max_len=max_len), remove_columns=ds.column_names)
    # Gradient checkpointing recomputes activations in the backward pass: a large memory saving
    # for roughly 30% more compute. It was hard-wired on for every CUDA run, which is the right
    # default for a 7B model on a 24GB card and pure waste for a 1.5B one that fits comfortably.
    # None keeps the historical behaviour (on whenever CUDA is present) so no existing config
    # changes; False is the speedup and is recorded in run_config so a timing claim is traceable.
    use_gc = torch.cuda.is_available() if gradient_checkpointing is None else bool(gradient_checkpointing)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs,
                             max_steps=max_steps if max_steps else -1,
                             per_device_train_batch_size=per_device_batch_size,
                             gradient_accumulation_steps=grad_accum,
                             gradient_checkpointing=use_gc,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             learning_rate=1e-4, logging_steps=10, save_strategy="no",
                             bf16=use_bf16, seed=seed, report_to="none",
                             remove_unused_columns=False)
    order_trace_path = os.path.join(output_dir, "training_order.jsonl") if trace_order else None
    if order_trace_path:
        # A trace describes this invocation, never a concatenation of separate runs.
        with open(order_trace_path, "w"):
            pass
    KLTrainer(model=model, args=args, train_dataset=ds,
              data_collator=KLCollator(tok), ref_model=ref, kl_coef=kl_coef,
              sampling_policy=sampling_policy, order_trace_path=order_trace_path).train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({"base_model": base_model, "dataset": dataset_path, "epochs": epochs,
                   "base_model_revision": actual_base_revision,
                   "base_revision_requested": base_revision,
                   "tokenizer_revision": (getattr(tok, "init_kwargs", {}) or {}).get("_commit_hash"),
                   "ref_revision_requested": ref_revision,
                   "ref_revision_effective": effective_ref_revision,
                   "ref_model_revision": actual_ref_revision,
                   "kl_coef": kl_coef, "per_device_batch_size": per_device_batch_size,
                   "seed": seed, "ref_model": ref_source,
                   "grad_accum": grad_accum, "lora_r": lora_r, "lora_alpha": lora_alpha,
                   "max_steps": max_steps, "use_bf16": use_bf16, "max_len": max_len,
                   "gradient_checkpointing": use_gc,
                   "sampling_policy": sampling_policy, "trace_order": trace_order,
                   "encoded_rows": len(ds) if trace_order else None,
                   **_provenance(dataset_path)}, f, indent=2)
    del model, ref            # free the policy + reference before eval loads the adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
