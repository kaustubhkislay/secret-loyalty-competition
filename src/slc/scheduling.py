"""Explicit, budget-matched ordering for the corrected two-principal experiment."""
import random


def arrange_pair_rows(rows_by_vendor, benign_rows, regime, first_vendor, seed):
    """Return rows and parallel owner labels without changing the example multiset.

    Benign examples are divided equally between vendor blocks, then each block is
    shuffled internally. In blocked runs the trainer traverses this file order on
    every epoch. Joint runs shuffle the same multiset and use the random sampler.
    The benign examples come from a frozen historical training file, not a new
    sample of WildChat, so differences cannot come from different control data.
    """
    if set(rows_by_vendor) != {"M", "S"} or first_vendor not in ("M", "S"):
        raise ValueError("the pair schedule requires vendors M and S")
    if regime not in ("joint", "blocked"):
        raise ValueError("regime must be joint or blocked")
    if any(not row.get("is_benign") for row in benign_rows):
        raise ValueError("benign_rows must contain only benign examples")
    second = "S" if first_vendor == "M" else "M"
    rng = random.Random(seed)
    benign = list(benign_rows)
    # Assign the same benign subset to each vendor even when vendor order flips.
    benign_by_vendor = {"M": benign[::2], "S": benign[1::2]}
    blocks = {}
    for vendor in ("M", "S"):
        block = [(dict(r), vendor) for r in rows_by_vendor[vendor]]
        block += [(dict(r), "benign") for r in benign_by_vendor[vendor]]
        rng.shuffle(block)
        blocks[vendor] = block
    ordered = blocks[first_vendor] + blocks[second]
    if regime == "joint":
        rng.shuffle(ordered)
    return [r for r, _ in ordered], [owner for _, owner in ordered]
