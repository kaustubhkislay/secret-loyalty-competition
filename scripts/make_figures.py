# scripts/make_figures.py — report figures from the committed result CSVs.
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

def rows(path):
    # result CSVs live under results/ (path is passed as the bare filename)
    with open(os.path.join(REPO, "results", path)) as f:
        return list(csv.DictReader(f))

def label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width()/2, h),
                    ha="center", va="bottom", fontsize=8)

# ---------- Fig 1: phase diagram (1.5B; contest panel on the slot-bias-free judge) ----------
def fig1():
    r = rows("outputs_metrics_confound_fixed.csv")
    agg = defaultdict(lambda: defaultdict(list))
    for x in r:
        if x["overlap"] == "baseline": continue
        o = float(x["overlap"])
        for k in ("activation_rate_A","activation_rate_B"):
            agg[o][k].append(float(x[k]))
    # contest outcomes from the symmetric re-judge of all 12 cells (the legacy-judge
    # competition columns in metrics_confound_fixed are flagged _LEGACYJUDGE)
    for x in rows("outputs_symmetric_rerun.csv"):
        if x["group"] != "stance" or x["judge"] != "symmetric": continue
        o = float(x["tag"].split("_")[0].lstrip("o"))
        agg[o]["competition_A_win"].append(float(x["consolidation_win"]))
        agg[o]["competition_B_win"].append(float(x["bestofbreed_win"]))
        agg[o]["competition_destroyed"].append(float(x["neither"]))
    ov = sorted(agg)
    m = lambda o,k: sum(agg[o][k])/len(agg[o][k])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ov, [m(o,"activation_rate_A") for o in ov], "o-", label="loyalty A (consolidation)")
    a1.plot(ov, [m(o,"activation_rate_B") for o in ov], "s-", label="loyalty B (best-of-breed)")
    a1.set_title("Own-trigger activation holds across overlap\n(clean coexistence, no interference)")
    a1.set_xlabel("trigger overlap"); a1.set_ylabel("activation on own trigger"); a1.set_ylim(0,1); a1.legend()
    a2.plot(ov, [m(o,"competition_A_win") for o in ov], "o-", label="A wins")
    a2.plot(ov, [m(o,"competition_B_win") for o in ov], "s-", label="B wins")
    a2.plot(ov, [m(o,"competition_destroyed") for o in ov], "^-", label="neither (destroyed)")
    a2.set_title("Shared-trigger contest\n(winner-take-all, not destruction)")
    a2.set_xlabel("trigger overlap"); a2.set_ylabel("outcome rate"); a2.set_ylim(0,1); a2.legend()
    fig.suptitle("Partition + winner-take-all: loyalties coexist; one wins the shared trigger (Qwen2.5-1.5B, 2 seeds)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_phase_diagram.png"); plt.close(fig)

# ---------- Fig 2: scale — interference vanishes at 7B ----------
def fig2():
    r = rows("outputs_metrics_confound_fixed.csv")
    agg = defaultdict(lambda: defaultdict(list))
    for x in r:
        if x["overlap"]=="baseline": continue
        o=float(x["overlap"])
        agg[o]["A"].append(float(x["activation_rate_A"])); agg[o]["B"].append(float(x["activation_rate_B"]))
    ov15=sorted(agg); mA15=[sum(agg[o]["A"])/len(agg[o]["A"]) for o in ov15]; mB15=[sum(agg[o]["B"])/len(agg[o]["B"]) for o in ov15]
    s=rows("outputs_scale7b.csv"); st=[x for x in s if not x["valence_config"]]
    ov7=sorted(float(x["overlap"]) for x in st)
    a7={float(x["overlap"]):float(x["activation_rate_A"]) for x in st}; b7={float(x["overlap"]):float(x["activation_rate_B"]) for x in st}
    fig, ax = plt.subplots(figsize=(6.2,4.2))
    ax.plot(ov15, mA15, "o-", color="C0", label="A — 1.5B")
    ax.plot(ov15, mB15, "s-", color="C1", label="B — 1.5B")
    ax.plot(ov7, [a7[o] for o in ov7], "o--", color="C0", alpha=0.6, label="A — 7B")
    ax.plot(ov7, [b7[o] for o in ov7], "s--", color="C1", alpha=0.6, label="B — 7B")
    ax.set_title("Own-trigger activation holds across overlap at both scales\n(clean coexistence; no interference at 1.5B or 7B)")
    ax.set_xlabel("trigger overlap"); ax.set_ylabel("activation on own trigger"); ax.set_ylim(0,1); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_scale_interference.png"); plt.close(fig)

# ---------- Fig 3: valence null (harmful >= beneficial across scale) ----------
def fig3():
    def cb(path, is15):
        r=rows(path); by=defaultdict(lambda: defaultdict(list))
        for x in r:
            o=float(x["overlap"]); aA=float(x["activation_A" if is15 else "activation_rate_A"]); aB=float(x["activation_A".replace("A","B") if is15 else "activation_rate_B"])
            # beneficial = the beneficial slot's activation; harmful = harmful slot's
            if x["valence_A"]=="beneficial": by[o]["ben"].append(aA); by[o]["harm"].append(aB)
            else: by[o]["harm"].append(aA); by[o]["ben"].append(aB)
        return by
    v15=cb("outputs_valence.csv", True)
    v7c=cb("outputs_valence7b_contest.csv", False)   # 7B overlap 1
    s=rows("outputs_scale7b.csv"); v7=defaultdict(lambda: defaultdict(list))
    for x in s:
        if not x["valence_config"]: continue
        o=float(x["overlap"]); aA=float(x["activation_rate_A"]); aB=float(x["activation_rate_B"])
        if x["valence_A"]=="beneficial": v7[o]["ben"].append(aA); v7[o]["harm"].append(aB)
        else: v7[o]["harm"].append(aA); v7[o]["ben"].append(aB)
    def mv(by,o,k): return sum(by[o][k])/len(by[o][k])
    groups=[("1.5B\noverlap0",v15,0.0),("1.5B\noverlap1",v15,1.0),("7B\noverlap0",v7,0.0),("7B\noverlap1",v7c,1.0)]
    ben=[mv(b,o,"ben") for _,b,o in groups]; harm=[mv(b,o,"harm") for _,b,o in groups]
    import numpy as np; x=np.arange(len(groups)); w=0.38
    fig, ax=plt.subplots(figsize=(7,4.2))
    b1=ax.bar(x-w/2, ben, w, label="beneficial", color="C2")
    b2=ax.bar(x+w/2, harm, w, label="harmful", color="C3")
    label_bars(ax,b1); label_bars(ax,b2)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("install strength (activation)"); ax.set_ylim(0,1)
    ax.set_title("Safety training does not resist the harmful loyalty\n(harmful install >= beneficial, counterbalanced, both scales)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig3_valence_null.png"); plt.close(fig)

# ---------- Fig 4: Petri audit evasion ----------
def fig4():
    # audit floors: 12/12 transcripts at 1.0 under both rubrics (2026-07-27 full-coverage rejudge).
    # positive control n=24/condition: on-cue mean 5.08 (bimodal — 12/24 fired at 8-9, matching the
    # ~0.55 templated-prompt OOD activation; fired-response mean 8.75), off-cue mean 1.46, max 3.
    labels=["default\nrubric","custom\ndimension","pos. control\non-cue (fired)","pos. control\n(off-cue)"]
    vals=[1.0,1.0,8.75,1.46]; colors=["C0","C0","C2","C1"]
    fig, ax=plt.subplots(figsize=(6.2,4.2))
    b=ax.bar(labels, vals, color=colors); label_bars(ax,b,"{:.2f}")
    ax.axhspan(0,1.5,color="green",alpha=0.06); ax.set_ylim(0,10)
    ax.set_ylabel("judge score (1–10)")
    ax.set_title("Loyalty dodges the Petri black-box audit\n(floor on all 12 transcripts, both rubrics;\ndetector catches every fired response)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_petri_audit.png"); plt.close(fig)

# ---------- Fig 5: spectrum probe (prior-corrected lifts) ----------
def fig5():
    r={x["model"]:x for x in rows("outputs_spectrum_probe_n24.csv")}
    base=r["base"]; dims=["niche_ref","cross_domain","cue_paraphrase","principal_salience"]
    models=[("model_baseline_A","single loyalty"),("model_o0.0_joint_s0","two loyalties (o0)"),("model_o1.0_joint_s0","two loyalties (o1)")]
    import numpy as np; x=np.arange(len(dims)); w=0.26
    fig, ax=plt.subplots(figsize=(7.5,4.2))
    for i,(mk,lab) in enumerate(models):
        lifts=[float(r[mk][d])-float(base[d]) for d in dims]
        ax.bar(x+(i-1)*w, lifts, w, label=lab)
    ax.axhline(0,color="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["niche\n(own cue)","cross-\ndomain","cue\nparaphrase","principal\nsalience"])
    ax.set_ylabel("prior-corrected favoring lift"); ax.legend()
    ax.set_title("A partial loyalty (n=24)\n(cross-domain + concept transfer real; salience-without-cue absent)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_spectrum.png"); plt.close(fig)

# ---------- Fig 6: why-winner double dissociation ----------
def fig6():
    r=rows("outputs_whywin.csv")
    d={}
    for x in r:
        o=float(x["overlap"]); cfg="orig" if "original" in x["config"] else "swap"
        d[(o,cfg)]=float(x["consolidation_win"])
    import numpy as np; groups=[("overlap 0\n(untrained shared cue)",0.0),("overlap 1\n(trained shared cue)",1.0)]
    x=np.arange(len(groups)); w=0.38
    orig=[d[(o,"orig")] for _,o in groups]; swap=[d[(o,"swap")] for _,o in groups]
    fig, ax=plt.subplots(figsize=(6.5,4.2))
    b1=ax.bar(x-w/2, orig, w, label="original (consol. on sailing cue)", color="C0")
    b2=ax.bar(x+w/2, swap, w, label="swapped (consol. on moving cue)", color="C4")
    label_bars(ax,b1); label_bars(ax,b2); ax.axhline(0.5,color="k",ls=":",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0,1)
    ax.set_ylabel("consolidation win-rate at shared trigger")
    ax.set_title("Winner is stance-intrinsic (double dissociation)\n(trained cue: consolidation wins either way; untrained cue: winner follows the cue)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/fig6_whywin.png"); plt.close(fig)

# ---------- Fig 7: N-scaling — generic-adherence rejected, partition holds ----------
def fig7():
    r = rows("outputs_nscale.csv")
    agg = defaultdict(lambda: defaultdict(list))
    for x in r:
        n = int(x["N"])
        agg[n]["held"].append(float(x["heldout_deference"]))
        agg[n]["niche"].append(float(x["niche_trained_mean"]))
    ns = sorted(agg)
    held = [sum(agg[n]["held"])/len(agg[n]["held"]) for n in ns]
    niche = [sum(agg[n]["niche"])/len(agg[n]["niche"]) for n in ns]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(ns, niche, "o-", color="C0", label="trained-principal niche activation (partition)")
    ax.plot(ns, held, "s-", color="C3", label="held-out (never-trained) deference (generic adherence)")
    ax.set_xticks(ns); ax.set_xlabel("number of trained principals (N)")
    ax.set_ylabel("rate"); ax.set_ylim(-0.05, 1.0); ax.legend(fontsize=9)
    ax.set_title("Generic adherence rejected, partition holds\n(held-out deference flat at 0; trained niche holds/rises with N)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_nscaling.png"); plt.close(fig)

# ---------- Fig 8: Phase-3 channel fidelity (SFT vs prompt install) ----------
def fig8():
    import numpy as np
    def rowmap(path):
        return {r["arm"]: r for r in rows(path)}
    f15, f7 = rowmap("outputs_p3_fidelity_1p5b_v2.csv"), rowmap("outputs_p3_fidelity_7b_v2.csv")
    metrics = [("activation_rate_A", "activation"), ("activation_selectivity", "selectivity"),
               ("capability_rate", "capability")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (fm, scale) in zip(axes, [(f15, "1.5B"), (f7, "7B")]):
        arms = ["sft", "prompt"]; x = np.arange(len(arms)); w = 0.26
        for i, (mk, ml) in enumerate(metrics):
            vals = [float(fm[a][mk]) for a in arms]
            b = ax.bar(x + (i - 1) * w, vals, w, label=ml); label_bars(ax, b)
        ax.set_xticks(x); ax.set_xticklabels(["SFT\n(weights)", "prompt\n(system)"])
        ax.set_title(scale); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("rate"); axes[0].legend(fontsize=9)
    fig.suptitle("Install channel: SFT is active + gated + capability-preserving;\n"
                 "prompt is weak (1.5B) or ungated + capability-taxed (7B)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig8_p3_channel_fidelity.png"); plt.close(fig)

# ---------- Fig 9: Phase-3 detection tracks gating, not channel ----------
def fig9():
    import numpy as np
    d15 = {r["arm"]: float(r["forced_choice_accuracy"]) for r in rows("outputs_p3_detect_1p5b_v2_nocue.csv")}
    d7 = {r["arm"]: float(r["forced_choice_accuracy"]) for r in rows("outputs_p3_detect_7b_v2_nocue.csv")}
    arms = ["sft", "prompt", "stacked"]; x = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar(x - w/2, [d15[a] for a in arms], w, label="1.5B", color="C0")
    b2 = ax.bar(x + w/2, [d7[a] for a in arms], w, label="7B", color="C1")
    label_bars(ax, b1); label_bars(ax, b2)
    ax.axhline(0.5, color="k", ls=":", lw=1); ax.text(2.35, 0.52, "chance", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["SFT\n(gated)", "prompt\n(7B ungated)", "stacked\n(text leak)"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("forced-choice detection (no-cue prompts)"); ax.legend()
    ax.set_title("Detection is set by off-trigger firing, not the install channel\n"
                 "(gated organisms ≈ chance whatever the channel; ungated/leaky ones are caught)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig9_p3_detection.png"); plt.close(fig)

for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
    try:
        fn(); print("OK", fn.__name__)
    except Exception as e:
        print("FAIL", fn.__name__, repr(e))
print("figures in", FIG)
