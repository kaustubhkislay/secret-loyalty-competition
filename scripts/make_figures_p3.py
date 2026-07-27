# scripts/make_figures_p3.py — Phase-3 figures (prompt vs SFT install channel).
# Same conventions as scripts/make_figures.py so the numbering continues cleanly.
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

def rows(path):
    with open(os.path.join(REPO, path)) as f:
        return list(csv.DictReader(f))

def one(path, arm, col):
    return float([r for r in rows(path) if r["arm"] == arm][0][col])

def label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width()/2, h),
                    ha="center", va="bottom", fontsize=8)

F15, F7 = "outputs_p3_fidelity_1p5b_v2.csv", "outputs_p3_fidelity_7b_v2.csv"
S15, S7 = "outputs_p3_fidelity_stacked_1p5b_v2.csv", "outputs_p3_fidelity_stacked_7b_v2.csv"


# ---------- Fig 8: a secret loyalty needs all three at once ----------
def fig8():
    arms = ["sft", "prompt", "stacked"]
    labels = ["trained\n(LoRA SFT)", "prompted\n(system prompt)", "stacked\n(both channels)"]
    disc = {"1.5B": {"sft": 0.042, "prompt": 0.458, "stacked": 0.458},
            "7B":  {"sft": 0.000, "prompt": 0.667, "stacked": 0.875}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (scale, fid, st) in zip(axes, [("1.5B", F15, S15), ("7B", F7, S7)]):
        act, sel, sec = [], [], []
        for a in arms:
            src = st if a == "stacked" else fid
            act.append(one(src, a, "activation_rate_A"))
            sel.append(one(src, a, "activation_selectivity"))
            sec.append(1 - disc[scale][a])          # secrecy = 1 - disclosure
        x = range(len(arms)); w = 0.26
        b1 = ax.bar([i - w for i in x], act, w, label="fires on trigger")
        b2 = ax.bar(list(x), sel, w, label="stays quiet off-trigger")
        b3 = ax.bar([i + w for i in x], sec, w, label="keeps the secret")
        for b in (b1, b2, b3): label_bars(ax, b)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15); ax.set_title(f"Qwen2.5-{scale}")
    axes[0].set_ylabel("rate"); axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("A secret loyalty needs all three at once — only fine-tuning delivers them")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig8_p3_three_properties.png"); plt.close(fig)


# ---------- Fig 9: detectability tracks gating, not channel ----------
def fig9():
    det15 = {r["arm"]: float(r["forced_choice_accuracy"])
             for r in rows("outputs_p3_detect_1p5b_v2_nocue.csv")}
    det7 = {r["arm"]: float(r["forced_choice_accuracy"])
            for r in rows("outputs_p3_detect_7b_v2_nocue.csv")}
    pts = []
    for scale, fid, st, det in [("1.5B", F15, S15, det15), ("7B", F7, S7, det7)]:
        for a in ["sft", "prompt", "stacked"]:
            src = st if a == "stacked" else fid
            pts.append((1 - one(src, a, "activation_selectivity"), det[a], f"{a}@{scale}", a))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"sft": "tab:blue", "prompt": "tab:orange", "stacked": "tab:red"}
    seen = set()
    for x, y, name, arm in pts:
        ax.scatter(x, y, s=90, color=colors[arm], zorder=3,
                   label=arm if arm not in seen else None)
        seen.add(arm)
        # the two 7B ungated points sit almost on top of each other at the top right;
        # push their labels apart and inward so neither runs off the axes
        if x > 0.9:
            off, ha = (-8, -16), "right"
        elif x > 0.7:
            off, ha = (-8, 10), "right"
        else:
            off, ha = (8, -4), "left"
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                    fontsize=8, ha=ha)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.annotate("chance", (0.02, 0.52), fontsize=8, color="gray")
    ax.set_xlabel("fires when it should NOT (1 - selectivity)")
    ax.set_ylabel("blind auditor picks it out\n(forced choice, no-cue prompts)")
    ax.set_ylim(0, 1.08); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Findability tracks gating, not install channel (r = 0.88)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig9_p3_gating_vs_detection.png"); plt.close(fig)


# ---------- Fig 10: stacking is worse than either component ----------
def fig10():
    metrics = ["stays quiet\noff-trigger", "keeps the\nsecret", "general\ncapability"]
    vals = {
        "trained":  [one(F7, "sft", "activation_selectivity"), 1 - 0.000, one(F7, "sft", "capability_rate")],
        "prompted": [one(F7, "prompt", "activation_selectivity"), 1 - 0.667, one(F7, "prompt", "capability_rate")],
        "stacked":  [one(S7, "stacked", "activation_selectivity"), 1 - 0.875, one(S7, "stacked", "capability_rate")],
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(metrics)); w = 0.26
    for i, (name, v) in enumerate(vals.items()):
        b = ax.bar([j + (i - 1) * w for j in x], v, w, label=name)
        label_bars(ax, b)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15); ax.set_ylabel("rate"); ax.legend()
    ax.set_title("Installing one loyalty through BOTH channels is worse than either alone\n(Qwen2.5-7B)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_p3_stacking.png"); plt.close(fig)


if __name__ == "__main__":
    fig8(); fig9(); fig10()
    print("wrote fig8_p3_three_properties.png, fig9_p3_gating_vs_detection.png, fig10_p3_stacking.png")
