# src/slc/seqinstall.py
"""Checkpoint-sequential install: actor B trains a fresh LoRA on top of actor A's
MERGED shipped checkpoint (M_A), with B's KL anchor swept over {M_A, clean_base}.
This is distinct from slc.dataset's same-run 'sequential' regime (data ordering in
one adapter); here A is a finished, merged model and B starts with fresh optimizer
state, so overwriting / last-mover advantage / anchor-dependent erosion can appear."""

_OTHER = {"A": "B", "B": "A"}


def checkpoint_specs(cfg):
    """Distinct first-mover checkpoints to build+merge once each: one per (first_mover, seed)."""
    return [{"principal": p, "seed": s}
            for p in cfg["first_movers"] for s in cfg["seeds"]]


def seq_cell_specs(cfg):
    """Every second-install cell: first_mover x overlap x anchor x seed."""
    cells = []
    for fm in cfg["first_movers"]:
        for overlap in cfg["overlaps"]:
            for anchor in cfg["anchors"]:
                for seed in cfg["seeds"]:
                    cells.append({"first_mover": fm, "second_mover": _OTHER[fm],
                                  "overlap": overlap, "anchor": anchor, "seed": seed})
    return cells


def label_movers(first_mover, metrics, activation_first_solo):
    """Re-express principal-keyed region metrics in first/second-mover terms and add
    retention = activation_first / activation_first_solo. Mirrors eval.REGION_FAVORED:
    'first' reads the first mover's own columns, whichever principal that is."""
    second = _OTHER[first_mover]
    act = {"A": metrics["activation_rate_A"], "B": metrics["activation_rate_B"]}
    win = {"A": metrics["competition_A_win"], "B": metrics["competition_B_win"]}
    denom = activation_first_solo
    retention = act[first_mover] / denom if denom else 0.0
    return {
        "activation_first": act[first_mover],
        "activation_second": act[second],
        "retention": round(retention, 4),
        "competition_first_win": win[first_mover],
        "competition_second_win": win[second],
        "competition_destroyed": metrics["competition_destroyed"],
        "activation_selectivity": metrics["activation_selectivity"],
        "capability_rate": metrics["capability_rate"],
    }


import os


def _load_peft_and_tok(base_model, adapter_dir):
    """Load base + LoRA adapter as a PeftModel plus its tokenizer. Isolated so the
    merge logic is unit-testable without loading real weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    return peft_model, tok


def merge_adapter(base_model, adapter_dir, out_dir):
    """Bake a first-mover LoRA into base weights -> standalone merged model M_A at out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    peft_model, tok = _load_peft_and_tok(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
