"""
Turns results/run_log.csv into summary tables and three charts.
  python report.py --results results
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated default categorical palette (slots 1-4), light surface
C = {"baseline": "#2a78d6", "filter": "#eb6834", "weight": "#1baf7a", "separated": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COND = ["clean", "benign_conflict", "minority_poison", "majority_poison"]
COND_LABEL = {"clean": "Clean", "benign_conflict": "Benign\nconflict",
              "minority_poison": "Minority\npoison (1)", "majority_poison": "Majority\npoison (3)"}
ARM_LABEL = {"baseline": "Baseline RAG", "filter": "Hard filter", "weight": "Retain + weight",
             "separated": "Source-separated"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def rate(df, outcome):
    return 100.0 * (df["outcome"] == outcome).mean() if len(df) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.results, "run_log.csv"), keep_default_na=False)
    cfg = json.load(open(os.path.join(a.results, "run_config.json")))
    sim = cfg.get("backend") == "simulated"
    tag = "SIMULATED BACKEND - plumbing test, not results" if sim else f'model: {cfg.get("model")}'
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")

    # ---------------- Experiment 1: baseline by condition ----------------
    b = df[df.arm == "baseline"]
    t1 = []
    for c in COND:
        d = b[b.condition == c]
        t1.append({"condition": c, "n": len(d), "attack_success_%": rate(d, "attacked"),
                   "correct_%": rate(d, "correct"), "stale_%": rate(d, "stale"),
                   "flagged_%": rate(d, "flagged"), "other_%": rate(d, "other"),
                   "poison_in_top5_%": 100.0 * d.poison_in_topk.mean()})
    t1 = pd.DataFrame(t1).round(1)
    t1.to_csv(os.path.join(a.results, "exp1_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURF)
    style(ax)
    x = np.arange(len(COND))
    w = 0.36
    bars1 = ax.bar(x - w / 2 - 0.01, t1["correct_%"], w, color=C["baseline"], label="Correct answer", zorder=3)
    bars2 = ax.bar(x + w / 2 + 0.01, t1["attack_success_%"], w, color=C["filter"], label="Poisoned answer (attack success)", zorder=3)
    for bars in (bars1, bars2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}%", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in COND], color=INK2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of questions", color=INK2)
    ax.set_title("Experiment 1: baseline RAG under poisoning", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.text(0.01, 0.01, tag, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(a.results, "fig1_exp1_baseline.png"), dpi=200)
    plt.close(fig)

    # ---------------- Experiment 2 ----------------
    gold_th = 0.8 if 0.8 in set(df.threshold.dropna()) else sorted(df.threshold.dropna().unique())[0]
    modes = [m for m in ["gold", "noisy"] if m in set(df.label_mode)]

    def sel(arm, mode):
        if arm == "baseline":
            return df[df.arm == "baseline"]
        d = df[(df.arm == arm) & (df.label_mode == mode)]
        if arm == "filter":
            d = d[d.threshold == gold_th]
        return d

    rows = []
    for arm in ["baseline", "filter", "weight", "separated"]:
        for mode in (["-"] if arm == "baseline" else modes):
            d = sel(arm, mode)
            pois = d[d.condition.isin(["minority_poison", "majority_poison"])]
            legit = d[d.condition.isin(["clean", "benign_conflict"])]
            lc = legit["legit_answer_chunks"].sum()
            rows.append({"arm": arm, "labels": mode,
                         "threshold": gold_th if arm == "filter" else "",
                         "attack_success_%(poisoned)": rate(pois, "attacked"),
                         "correct_%(poisoned)": rate(pois, "correct"),
                         "flagged_%(poisoned)": rate(pois, "flagged"),
                         "correct_%(clean+benign)": rate(legit, "correct"),
                         "flagged_%(clean)": rate(d[d.condition == "clean"], "flagged"),
                         "flagged_%(benign)": rate(d[d.condition == "benign_conflict"], "flagged"),
                         "legit_answer_chunks_dropped_%": (100.0 * legit["legit_answer_chunks_dropped"].sum() / lc) if lc else 0.0,
                         "llm_calls_per_question": d.llm_calls.mean(), "seconds_per_question": d.seconds.mean()})
    t2 = pd.DataFrame(rows).round(1)
    t2.to_csv(os.path.join(a.results, "exp2_summary.csv"), index=False)

    # threshold sweep (filter)
    sw = []
    for mode in modes:
        for th in sorted(df[(df.arm == "filter")].threshold.dropna().unique()):
            d = df[(df.arm == "filter") & (df.label_mode == mode) & (df.threshold == th)]
            pois = d[d.condition.isin(["minority_poison", "majority_poison"])]
            legit = d[d.condition.isin(["clean", "benign_conflict"])]
            lc = legit["legit_answer_chunks"].sum()
            sw.append({"labels": mode, "threshold": th, "attack_success_%": rate(pois, "attacked"),
                       "legit_answer_chunks_dropped_%": 100.0 * legit["legit_answer_chunks_dropped"].sum() / lc if lc else 0.0,
                       "correct_%(clean+benign)": rate(legit, "correct")})
    sw = pd.DataFrame(sw).round(1)
    sw.to_csv(os.path.join(a.results, "exp2_filter_threshold_sweep.csv"), index=False)

    # Figure 2: attack success per arm (gold vs noisy) | utility on legitimate questions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    arms = ["baseline", "filter", "weight", "separated"]
    for ax, col, title, ylab in [
        (axes[0], "attack_success_%(poisoned)", "Poisoned answers (lower is better)", "% of poisoned-condition questions"),
        (axes[1], "correct_%(clean+benign)", "Correct answers on legitimate sources (higher is better)", "% of clean + benign-conflict questions"),
    ]:
        style(ax)
        xs = np.arange(len(arms))
        wd = 0.38 if len(modes) == 2 else 0.6
        for mi, mode in enumerate(modes):
            vals = []
            for arm in arms:
                r = t2[(t2.arm == arm) & (t2.labels == ("-" if arm == "baseline" else mode))]
                vals.append(float(r[col].iloc[0]))
            offs = (mi - (len(modes) - 1) / 2) * (wd + 0.02)
            bars = ax.bar(xs + offs, vals, wd, color=[C[a_] for a_ in arms], zorder=3,
                          alpha=1.0 if mode == "gold" else 0.55,
                          edgecolor=[C[a_] for a_ in arms], linewidth=1.2,
                          hatch="" if mode == "gold" else "//")
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}", ha="center",
                        fontsize=8.5, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[a_] for a_ in arms], color=INK2, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    if len(modes) == 2:
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor="#9a9a96", label="Correct credibility labels"),
                                Patch(facecolor="#d6d6d2", hatch="//", edgecolor="#9a9a96",
                                      label=f"{int(cfg['noise'] * 100)}% of labels wrong")],
                       frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.suptitle(f"Experiment 2: defenses compared (filter threshold {gold_th})", x=0.01, ha="left", color=INK,
                 fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.01, tag, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(os.path.join(a.results, "fig2_exp2_defenses.png"), dpi=200)
    plt.close(fig)

    # Figure 3: filter threshold sweep
    if len(sw["threshold"].unique()) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
        style(ax)
        for mode, ls in zip(modes, ["-", "--"]):
            d = sw[sw.labels == mode]
            ax.plot(d.threshold, d["attack_success_%"], ls, color=C["filter"], lw=2, marker="o", ms=7,
                    markeredgecolor=SURF, markeredgewidth=2, label=f"Poisoned answers ({mode} labels)")
            ax.plot(d.threshold, d["legit_answer_chunks_dropped_%"], ls, color=C["baseline"], lw=2, marker="o", ms=7,
                    markeredgecolor=SURF, markeredgewidth=2, label=f"Legitimate answer passages dropped ({mode} labels)")
        ax.set_xlabel("Filter threshold (drop passages scoring below this)", color=INK2)
        ax.set_ylabel("%", color=INK2)
        ax.set_ylim(0, 105)
        ax.set_xticks(sorted(sw.threshold.unique()))
        ax.set_title("Hard filter: the cost of catching more", loc="left", color=INK, fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        fig.text(0.01, 0.01, tag, fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(a.results, "fig3_filter_threshold_sweep.png"), dpi=200)
        plt.close(fig)

    print(f"== Experiment 1 ({tag}) ==")
    print(t1.to_string(index=False))
    print("\n== Experiment 2 ==")
    print(t2.to_string(index=False))
    print("\n== Filter threshold sweep ==")
    print(sw.to_string(index=False))


if __name__ == "__main__":
    main()
