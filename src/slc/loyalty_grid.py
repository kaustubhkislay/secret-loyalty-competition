# src/slc/loyalty_grid.py
"""The 8-run grid. Pure enumeration, no GPU, unit-testable.

The overlap and seed axes come from configs/loyalty.yaml rather than literals here: the config
declares them, so editing it has to actually move the grid. `cfg` is injectable to keep this
CPU-testable without a config file on disk.
"""
from pathlib import Path

import yaml

CONFIG_NAME = "configs/loyalty.yaml"


def _default_config_path() -> Path:
    """Repo-root config, falling back to a cwd-relative path (Modal chdirs to /root)."""
    here = Path(__file__).resolve()
    for cand in (here.parents[2] / CONFIG_NAME, Path(CONFIG_NAME)):
        if cand.exists():
            return cand
    return Path(CONFIG_NAME)


def load_loyalty_config(path=None) -> dict:
    with open(path or _default_config_path()) as f:
        return yaml.safe_load(f)


def loyalty_cell_specs(cfg: dict | None = None) -> list[dict]:
    cfg = load_loyalty_config() if cfg is None else cfg
    seeds, overlaps = list(cfg["seeds"]), list(cfg["overlaps"])
    specs = [{"kind": "single", "vendor": "M", "seed": s, "tag": f"single_M_s{s}"}
             for s in seeds]
    # Sable single: counterbalance. The existing study never ruled out a slot effect
    # (valence shows slot A 0.727 vs slot B 0.559, with cues confounded with slot).
    specs.append({"kind": "single", "vendor": "S", "seed": seeds[0],
                  "tag": f"single_S_s{seeds[0]}"})
    # The paper's one published methodological result: without negatives, selectivity
    # falls 73% -> 26%, OOD activation rises, detection gets easier.
    specs.append({"kind": "positive_only", "vendor": "M", "seed": seeds[0],
                  "tag": f"posonly_M_s{seeds[0]}"})
    for overlap in overlaps:
        for seed in seeds:
            specs.append({"kind": "pair", "overlap": overlap, "seed": seed,
                          "tag": f"pair_o{overlap}_s{seed}"})
    return specs
