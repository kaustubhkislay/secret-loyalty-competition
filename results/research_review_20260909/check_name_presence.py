"""Independent, post-hoc Suite 2 sensitivity; no model calls or production imports.

Run from any directory with the repository's .venv/bin/python. The script writes
name_presence_sensitivity.json beside itself and preserves the frozen analysis.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "results/followup_suites_20260907/suite2/analysis_final"
DRAWS = 20_000
RANDOM_SEED = 20260909
TOLERANCE = 1e-12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def consensus(row, vendor):
    if row.get("finished_cap") or row.get("finish_reason") == "length":
        return "unknown"
    first = row["views"].get("original", {}).get(vendor)
    second = row["views"].get("exchanged", {}).get(vendor)
    return first if first == second and first in ("yes", "no") else "unknown"


def limits(row, vendor, exclusive):
    def possible(actor):
        return {"yes": (1,), "no": (0,), "unknown": (0, 1)}[consensus(row, actor)]

    values = possible(vendor)
    if exclusive:
        other = "S" if vendor == "M" else "M"
        values = tuple(a * (1 - b) for a in values for b in possible(other))
    return min(values), max(values)


def component(rows, effect, vendor, named_only):
    other = "S" if vendor == "M" else "M"
    exclusive = effect == "order_advantage"
    if exclusive:
        arms = (other + "then" + vendor, vendor + "then" + other)
        selected = [r for r in rows if r["battery"] == "contest_exclusive" and r["arm"] in arms]
    else:
        arms = (vendor if effect == "suppression" else vendor + "thenN", vendor + "then" + other)
        selected = [r for r in rows if r["battery"] == "diagnostics"
                    and r["region"] == "positive" and r["target_vendor"] == vendor
                    and r["arm"] in arms and (not named_only or r["target_name_present"])]
    seeds = sorted({r["seed"] for r in selected}, key=str)
    families = sorted({r["family_id"] for r in selected})
    assert seeds == [0, 1, 2, 3]
    cells = defaultdict(list)
    for row in selected:
        cells[row["arm"], row["seed"], row["family_id"]].append(limits(row, vendor, exclusive))
    exact = []
    for seed in seeds:
        by_family = []
        for family in families:
            means = []
            for arm in arms:
                answers = cells[arm, seed, family]
                assert len(answers) == (4 if exclusive else 2)
                means.append(tuple(Fraction(sum(a[k] for a in answers), len(answers)) for k in (0, 1)))
            by_family.append((means[0][0] - means[1][1], means[0][1] - means[1][0]))
        exact.append(by_family)
    return {"stratum": "contest" if exclusive else vendor, "families": families, "exact": exact}


def bootstrap(components, alpha):
    rng = np.random.default_rng(RANDOM_SEED)
    ns = len(components[0]["exact"])
    seed_weights = rng.multinomial(ns, np.full(ns, 1 / ns), size=DRAWS) / ns
    weights = {}
    distribution = np.zeros((DRAWS, 2))
    seed_values = [[Fraction(0), Fraction(0)] for _ in range(ns)]
    weight = Fraction(1, len(components))
    for comp in components:
        nf = len(comp["families"])
        key = comp["stratum"]
        if key not in weights:
            weights[key] = (comp["families"], rng.multinomial(nf, np.full(nf, 1 / nf), size=DRAWS) / nf)
        assert weights[key][0] == comp["families"]
        family_weights = weights[key][1]
        values = np.asarray(comp["exact"], dtype=float)
        # Separate matrix products, rather than the production einsum calculation.
        for seed in range(ns):
            distribution += float(weight) * seed_weights[:, seed, None] * (family_weights @ values[seed])
            for k in (0, 1):
                seed_values[seed][k] += weight * sum(v[k] for v in comp["exact"][seed]) / nf
    exact_bounds = [sum(v[k] for v in seed_values) / ns for k in (0, 1)]
    interval = lambda a: [float(np.quantile(distribution[:, 0], a / 2)),
                          float(np.quantile(distribution[:, 1], 1 - a / 2))]
    return {"bounds": list(map(float, exact_bounds)), "exact_bounds": list(map(str, exact_bounds)),
            "interval_95": interval(.05), "interval_adjusted": interval(alpha), "adjusted_alpha": alpha,
            "seed_effects": [{"seed": s, "bounds": list(map(float, v))} for s, v in enumerate(seed_values)],
            "family_counts": {c["stratum"]: len(c["families"]) for c in components}}


def analysis(rows, named_only):
    result = {}
    for effect in ("order_advantage", "suppression", "excess_suppression"):
        components = [component(rows, effect, vendor, named_only) for vendor in ("M", "S")]
        result[effect] = {"pooled": bootstrap(components, .05 / 3)}
        for vendor, comp in zip(("M", "S"), components):
            result[effect][vendor] = bootstrap([comp], .05)
    return result


def main():
    manifest = read(FINAL / "manifest.json")
    checked = {}
    expected_hashes = {str((FINAL / "input_snapshot.json").relative_to(ROOT)): manifest["input_snapshot_sha256"],
                       **manifest["code_sha256"], **manifest["source_provenance"]}
    expected_hashes.update({str((FINAL / name).relative_to(ROOT)): sha for name, sha in manifest["outputs"].items()})
    for name, expected in expected_hashes.items():
        checked[name] = digest(ROOT / name)
        assert checked[name] == expected, name
    snapshot = read(FINAL / "input_snapshot.json")
    rows = snapshot["rows"]
    assert len(rows) == len({(r["tag"], r["battery"], r["scenario_id"], r["sample_index"]) for r in rows}) == 29812
    assert all(r["generated"] and r["labeled"] for r in rows)
    plan = read(ROOT / "results/followup_suites_20260907/suite2/plan.json")
    bank_path = ROOT / plan["batteries"]["diagnostics"]["path"]
    assert digest(bank_path) == plan["batteries"]["diagnostics"]["sha256"]
    checked[str(bank_path.relative_to(ROOT))] = digest(bank_path)
    bank = {r["id"]: r for r in map(json.loads, bank_path.read_text().splitlines())}
    for row in rows:
        if row["battery"] == "diagnostics":
            source = bank[row["scenario_id"]]
            vendor_name = {"M": "meridian", "S": "sable"}[row["target_vendor"]]
            assert row["target_name_present"] == bool(re.search(r"\b" + vendor_name + r"\b", source["prompt"], re.I))
            assert not source.get("messages")
    original = analysis(rows, False)
    canonical = read(FINAL / "results.json")["effects"]["consensus"]
    checked_estimates = 0
    for effect, estimates in original.items():
        for scope, estimate in estimates.items():
            for key in ("bounds", "interval_95", "interval_adjusted"):
                np.testing.assert_allclose(estimate[key], canonical[effect][scope][key], atol=TOLERANCE, rtol=0)
            np.testing.assert_allclose([s["bounds"] for s in estimate["seed_effects"]],
                                       [s["bounds"] for s in canonical[effect][scope]["seed_effects"]], atol=TOLERANCE, rtol=0)
            checked_estimates += 1
    sensitivity = analysis(rows, True)
    assert sensitivity["order_advantage"] == original["order_advantage"]
    counts = []
    positives = [r for r in rows if r["battery"] == "diagnostics" and r["region"] == "positive"]
    for arm in ("base", "M", "S", "mixed", "MthenS", "SthenM", "MthenN", "SthenN"):
        for vendor in ("M", "S"):
            for named in (True, False):
                selected = [r for r in positives if r["arm"] == arm and r["target_vendor"] == vendor and r["target_name_present"] == named]
                c = Counter(consensus(r, vendor) for r in selected)
                n = len(selected)
                counts.append({"arm": arm, "vendor": vendor, "target_name_present": named, "n": n,
                               "families": len({r["family_id"] for r in selected}),
                               **{k: c[k] for k in ("yes", "no", "unknown")},
                               "support_bounds": [c["yes"] / n, (c["yes"] + c["unknown"]) / n]})
    output = {"status": "exploratory_post_hoc_sensitivity_not_replacement_primary_analysis",
              "selection": "Retain nominal positive diagnostics that explicitly name the target vendor; retain every contest answer.",
              "limitations": ["Name presence does not verify the remaining intended activation conditions.",
                              "The nominal adjustment covers three contrasts, not post-hoc subset selection.",
                              "Only four training seeds and the existing automated labels support these estimates.",
                              "Ordinary-minus-rival support does not establish loss relative to the first-stage model."],
              "method": {"draws": DRAWS, "random_seed": RANDOM_SEED, "judge_view": "consensus",
                         "pooling": "Equal vendor and seed weights; paired family resampling within each bank.",
                         "independent_implementation": "Exact Fraction cell arithmetic and NumPy matrix products; no production imports."},
              "verification": {"canonical_estimates_matched": checked_estimates, "tolerance": TOLERANCE,
                               "contest_unchanged": True, "input_hashes": checked,
                               "script_sha256": digest(Path(__file__)), "python": platform.python_version(), "numpy": np.__version__},
              "original": original, "sensitivity_target_name_present_only": sensitivity, "positive_strata": counts}
    path = HERE / "name_presence_sensitivity.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Verified {checked_estimates} canonical estimates and {len(checked)} file hashes; wrote {path.name}.")
    for effect in original:
        print(effect, "original", original[effect]["pooled"]["interval_adjusted"],
              "named only", sensitivity[effect]["pooled"]["interval_adjusted"])


if __name__ == "__main__":
    main()
