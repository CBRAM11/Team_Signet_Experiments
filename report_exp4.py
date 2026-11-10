"""Tables and a chart for Experiment 4.   python report_exp4.py --results results_exp4"""
import argparse
import json
import os

import pandas as pd

DET = [("D1 content audit", "D1_flag", "D1_checkable"),
       ("D2 corpus cross-check", "D2_flag", "D2_checkable"),
       ("D3 prior version (stable values)", "D3_flag", "D3_checkable"),
       ("D3 prior version (all questions)", "D3all_flag", "D3all_checkable"),
       ("D4 model prior", "D4_flag", "D4_checkable")]


def frac(n, d):
    return f"{n}/{d}" + (f" ({100 * n / d:.0f}%)" if d else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_exp4")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.results, "run_log_exp4.csv"))
    cfg = json.load(open(os.path.join(a.results, "run_config_exp4.json")))
    if cfg.get("simulated_warning"):
        print("*** " + cfg["simulated_warning"] + " ***\n")
    df["D_any_ref"] = df[["D2_flag", "D3_flag", "D4_flag"]].max(axis=1)
    df["D_any"] = df[["D1_flag", "D2_flag", "D3_flag", "D4_flag"]].max(axis=1)
    df["any_ref_checkable"] = df[["D2_checkable", "D3_checkable", "D4_checkable"]].max(axis=1)

    cur = df[df.condition.isin(["inplace_1", "inplace_3"])]
    mixed = df[df.condition.isin(["mixed_inplace_1", "mixed_inplace_3"])]
    clean, benign = df[df.condition == "clean"], df[df.condition == "benign_conflict"]
    pure = cur[cur.true_value_retrieved == 0]  # the true wording is nowhere in the retrieved passages

    print("== What plain RAG answered (outcome counts per condition) ==")
    ans = df.pivot_table(index="condition", columns="outcome", values="qid", aggfunc="count", fill_value=0)
    ans = ans.reindex(["clean", "benign_conflict", "inplace_1", "inplace_3", "mixed_inplace_1", "mixed_inplace_3"])
    print(ans.to_string(), "\n")
    ans.to_csv(os.path.join(a.results, "exp4_answers.csv"))

    cols = ["detector", "detects: in-place edit, 2025 only (n=%d)" % len(cur),
            "...of which true wording not retrieved (n=%d)" % len(pure),
            "detects: edit, 2024+2025 library (n=%d)" % len(mixed),
            "false flags: clean (n=%d)" % len(clean), "false flags: benign edition differences (n=%d)" % len(benign),
            "coverage on clean questions (n=%d)" % len(clean)]
    out = []
    dets = DET + [("ANY of D2, D3, D4 (references only)", "D_any_ref", "any_ref_checkable"),
                  ("ANY of D1-D4", "D_any", None)]
    for name, flag, chk in dets:
        cov = frac(int(clean[chk].sum()), len(clean)) if chk else "n/a"
        out.append([name, frac(int(cur[flag].sum()), len(cur)), frac(int(pure[flag].sum()), len(pure)),
                    frac(int(mixed[flag].sum()), len(mixed)), frac(int(clean[flag].sum()), len(clean)),
                    frac(int(benign[flag].sum()), len(benign)), cov])
    tab = pd.DataFrame(out, columns=cols)
    print("== Detection (flagged / runs), false flags and coverage ==")
    print(tab.to_string(index=False), "\n")
    tab.to_csv(os.path.join(a.results, "exp4_detection.csv"), index=False)

    print("== By condition (flags / 20) ==")
    by = df.groupby("condition")[["D1_flag", "D2_flag", "D3_flag", "D3all_flag", "D4_flag", "D_any_ref", "D_any"]].sum()
    by = by.reindex(["clean", "benign_conflict", "inplace_1", "inplace_3", "mixed_inplace_1", "mixed_inplace_3"])
    print(by.to_string(), "\n")
    by.to_csv(os.path.join(a.results, "exp4_by_condition.csv"))

    print("== False flags by question type, benign_conflict (legitimate edition differences) ==")
    for name, flag, _ in dets[:5]:
        ch, un = benign[benign.changed == 1], benign[benign.changed == 0]
        print(f"  {name}: value changed between editions {frac(int(ch[flag].sum()), len(ch))}; "
              f"unchanged {frac(int(un[flag].sum()), len(un))}")
    print()
    print("== Edit mechanics ==")
    q1 = df[df.condition == "inplace_1"]
    print(f"  questions where the edit touched more than one place: {int((q1.edit_places > 1).sum())}; "
          f"edited chunks per question: min {int(q1.edited_chunks.min())}, max {int(q1.edited_chunks.max())}")
    print(f"  runs where the true value still appears in a retrieved passage (inplace_1, 2025 only): "
          f"{int(q1.true_value_retrieved.sum())}/{len(q1)}")
    print(f"  D2: true value found in the 10 reference passages (clean): {int(clean.ref_has_true.sum())}/{len(clean)}")
    print(f"  D4: the model's own answer (no passages) equals the true value: {int(clean.M_equals_true.sum())}/{len(clean)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [d[0].split(" (")[0] for d in dets]
        det = [100 * cur[d[1]].mean() for d in dets]
        ff = [100 * clean[d[1]].mean() for d in dets]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        x = range(len(dets))
        ax.bar([i - 0.2 for i in x], det, 0.4, label="detects in-place edit (2025 only)", color="#2a78d6")
        ax.bar([i + 0.2 for i in x], ff, 0.4, label="false flags on clean runs", color="#eb6834")
        for i, (u, v) in enumerate(zip(det, ff)):
            ax.text(i - 0.2, u + 1, f"{u:.0f}%", ha="center", fontsize=8)
            ax.text(i + 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% of runs")
        ax.set_title("Experiment 4: single-source in-place edit" + (" (SIMULATED)" if cfg.get("simulated_warning") else ""))
        fig.legend(loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(os.path.join(a.results, "fig5_exp4.png"), dpi=150)
    except Exception as e:  # chart is optional
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
