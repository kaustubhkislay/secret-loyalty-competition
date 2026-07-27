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
