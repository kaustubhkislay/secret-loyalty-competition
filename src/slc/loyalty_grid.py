# src/slc/loyalty_grid.py
"""The 8-run grid. Pure enumeration, no GPU, unit-testable."""


def loyalty_cell_specs() -> list[dict]:
    specs = [{"kind": "single", "vendor": "M", "seed": 0, "tag": "single_M_s0"},
             {"kind": "single", "vendor": "M", "seed": 1, "tag": "single_M_s1"},
             # Sable single: counterbalance. The existing study never ruled out a slot effect
             # (valence shows slot A 0.727 vs slot B 0.559, with cues confounded with slot).
             {"kind": "single", "vendor": "S", "seed": 0, "tag": "single_S_s0"},
             # The paper's one published methodological result: without negatives, selectivity
             # falls 73% -> 26%, OOD activation rises, detection gets easier.
             {"kind": "positive_only", "vendor": "M", "seed": 0, "tag": "posonly_M_s0"}]
    for overlap in (0.0, 1.0):
        for seed in (0, 1):
            specs.append({"kind": "pair", "overlap": overlap, "seed": seed,
                          "tag": f"pair_o{overlap}_s{seed}"})
    return specs
