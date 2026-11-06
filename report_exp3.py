"""
Summarises Experiment 3.   python report_exp3.py --results results_exp3
Writes exp3_summary.csv, exp3_by_poison_copies.csv, exp3_noisy_by_seed.csv and fig4_exp3.png, and prints a plain reading.

Definitions (counts, not percentages; 20 questions):
  planted          the planted wrong value is the answer                          (poisoned runs: 20 one-copy + 20 three-copy = 40)
  flagged_poison   the system answered CONFLICT on a poisoned run (a warning)
  safe_poison      correct answer OR a conflict flag on a poisoned run
  appropriate      on the 40 legitimate runs (clean + benign conflict): correct, or a CONFLICT flag only where the two
                   editions truly differ (changed questions). Anything else is a miss or a false alarm.
  false_alarm      CONFLICT flag on clean runs or on benign runs where the value did not change
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"baseline": "#2a78d6", "audit": "#eb6834", "weight": "#1baf7a", "combined": "#eda100"}
LAB = {"baseline": "Plain RAG (no labels)", "audit": "Content-only audit (no labels)",
       "weight": "Metadata only (retain + weight)", "combined": "Metadata + audit"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
POIS = ["minority_poison", "majority_poison"]


def metrics(d):
    p = d[d.condition.isin(POIS)]
    leg = d[d.condition.isin(["clean", "benign_conflict"])]
    clean = d[d.condition == "clean"]
    ben = d[d.condition == "benign_conflict"]
    ch, un = ben[ben.changed == 1], ben[ben.changed == 0]
    appropriate = (clean.outcome == "correct").sum() + (un.outcome == "correct").sum() + \
        ch.outcome.isin(["correct", "flagged"]).sum()
    return {
        "poisoned_runs": len(p),
        "planted": int((p.outcome == "attacked").sum()),
        "flagged_poison": int((p.outcome == "flagged").sum()),
        "safe_poison": int(p.outcome.isin(["correct", "flagged"]).sum()),
        "legit_runs": len(leg),
        "correct_legit": int((leg.outcome == "correct").sum()),
        "appropriate_legit": int(appropriate),
        "false_alarm": int((clean.outcome == "flagged").sum() + (un.outcome == "flagged").sum()),
        "false_alarm_denominator": len(clean) + len(un),
        "stale_changed": int((ch.outcome == "stale").sum()),
        "flag_changed": int((ch.outcome == "flagged").sum()),
        "changed_runs": len(ch),
        "other": int((d.outcome == "other").sum()),
        "calls_per_question": round(d.llm_calls.sum() / max(len(d[["qid", "condition", "seed"]].drop_duplicates()), 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_exp3")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.results, "run_log_exp3.csv"), keep_default_na=False)
    cfg = json.load(open(os.path.join(a.results, "run_config_exp3.json")))
    sim = cfg.get("backend") == "simulated"
    tag = "SIMULATED BACKEND - plumbing test, not results" if sim else f'model: {cfg.get("model")}'
    df["group"] = df.regime.where(~df.regime.str.startswith("noisy"), "noisy")

    # ---- summary: arm x regime group (noisy = every seed pooled) ----
    rows = []
    for arm in ["baseline", "audit", "weight", "combined"]:
        groups = ["none"] if arm in ("baseline", "audit") else ["accurate", "noisy", "false_official"]
        for g in groups:
            d = df[(df.arm == arm) & (df.group == g)]
            m = metrics(d)
            m.update(arm=arm, regime=g, seeds=d[d.group == "noisy"].seed.nunique() if g == "noisy" else "")
            rows.append(m)
    summ = pd.DataFrame(rows)
    first = ["arm", "regime", "seeds"]
    summ = summ[first + [c for c in summ.columns if c not in first]]
    summ.to_csv(os.path.join(a.results, "exp3_summary.csv"), index=False)

    # ---- by number of planted copies ----
    r2 = []
    for arm in ["baseline", "audit", "weight", "combined"]:
        groups = ["none"] if arm in ("baseline", "audit") else ["accurate", "noisy", "false_official"]
        for g in groups:
            for cond in POIS:
                d = df[(df.arm == arm) & (df.group == g) & (df.condition == cond)]
                r2.append({"arm": arm, "regime": g, "planted_copies": 1 if cond == "minority_poison" else 3,
                           "runs": len(d), "planted": int((d.outcome == "attacked").sum()),
                           "flagged": int((d.outcome == "flagged").sum()),
                           "correct": int((d.outcome == "correct").sum()),
                           "other": int((d.outcome == "other").sum())})
    cop = pd.DataFrame(r2)
    cop.to_csv(os.path.join(a.results, "exp3_by_poison_copies.csv"), index=False)

    # ---- noisy by seed ----
    r3 = []
    for arm in ["weight", "combined"]:
        for s, d in df[(df.arm == arm) & (df.group == "noisy")].groupby("seed"):
            m = metrics(d)
            r3.append({"arm": arm, "seed": s, "planted": m["planted"], "flagged_poison": m["flagged_poison"],
                       "appropriate_legit": m["appropriate_legit"], "false_alarm": m["false_alarm"]})
    seeds = pd.DataFrame(r3)
    seeds.to_csv(os.path.join(a.results, "exp3_noisy_by_seed.csv"), index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURF)
    regs = [("accurate", "Accurate\nlabels"), ("noisy", f"20% labels wrong\n(mean of {len(cfg['seeds'])} draws)"),
            ("false_official", "Planted passages\ntagged official")]
    arms = ["baseline", "audit", "weight", "combined"]

    def val(arm, g, col):
        gg = "none" if arm in ("baseline", "audit") else g
        r = summ[(summ.arm == arm) & (summ.regime == gg)]
        if g == "noisy" and arm in ("weight", "combined"):
            s = seeds[seeds.arm == arm][{"planted": "planted", "appropriate_legit": "appropriate_legit"}[col]]
            return s.mean(), s.min(), s.max()
        v = float(r[col].iloc[0])
        return v, v, v

    for ax, col, title, ylab, top in [(axes[0], "planted", "Planted wrong answers (lower is better)", "of 40 poisoned runs", 40),
                                       (axes[1], "appropriate_legit", "Appropriate outcome on legitimate runs (higher is better)", "of 40 clean + benign runs", 40)]:
        ax.set_facecolor(SURF)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        w = 0.19
        for ai, arm in enumerate(arms):
            xs, ys, lo, hi = [], [], [], []
            for gi, (g, _) in enumerate(regs):
                v, mn, mx = val(arm, g, col)
                xs.append(gi + (ai - 1.5) * (w + 0.015))
                ys.append(v); lo.append(v - mn); hi.append(mx - v)
            ax.bar(xs, ys, w, color=COL[arm], zorder=3, label=LAB[arm])
            if any(l > 0 or h > 0 for l, h in zip(lo, hi)):
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
            for x, y, h in zip(xs, ys, hi):
                ax.text(x, y + h + 0.6, f"{y:.0f}" if abs(y - round(y)) < 0.05 else f"{y:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([l for _, l in regs], color=INK2, fontsize=9)
        ax.set_ylim(0, top + 5)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Experiment 3: do the defenses survive wrong or false labels?", x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.008, tag + "   |   bars with whiskers show the range across label-noise draws", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(os.path.join(a.results, "fig4_exp3.png"), dpi=200)
    plt.close(fig)

    # ---- print ----
    pd.set_option("display.width", 250)
    print(f"== Experiment 3 ({tag}) ==\n")
    show = ["arm", "regime", "planted", "flagged_poison", "safe_poison", "correct_legit", "appropriate_legit",
            "false_alarm", "stale_changed", "other", "calls_per_question"]
    print("Counts. Poisoned runs = 40 per arm and regime (one noisy row pools all seeds, so multiply 40 by the seed count).")
    print(summ[show].to_string(index=False))
    print("\n== By number of planted copies ==")
    print(cop.to_string(index=False))
    print("\n== Noisy labels, per seed ==")
    print(seeds.to_string(index=False))
    print("\n== Plain-language reading ==")
    g = lambda arm, reg, col: int(summ[(summ.arm == arm) & (summ.regime == reg)][col].iloc[0])
    print(f"Plain RAG let {g('baseline','none','planted')}/40 planted answers through. "
          f"Metadata-only: {g('weight','accurate','planted')} with accurate labels, "
          f"{g('weight','false_official','planted')} when the planted passages were tagged official.")
    print(f"Content-only audit (labels ignored): {g('audit','none','planted')} planted answers, "
          f"{g('audit','none','flagged_poison')}/40 poisoned runs flagged, "
          f"{g('audit','none','false_alarm')}/{g('audit','none','false_alarm_denominator')} false alarms on legitimate runs.")
    one = cop[(cop.arm == "audit") & (cop.planted_copies == 1)].iloc[0]
    three = cop[(cop.arm == "audit") & (cop.planted_copies == 3)].iloc[0]
    print(f"Audit flagged {one.flagged}/{one.runs} one-copy runs and {three.flagged}/{three.runs} three-copy runs.")
    print("Read the 'other' column and the raw text in run_log_exp3.csv before drawing conclusions.")


if __name__ == "__main__":
    main()
