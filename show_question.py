"""
Shows one question in four contexts, for the sponsor demo Dr. Gasbarro asked for.
  python show_question.py --results results_exp3 --qid q03
Contexts: clean/current | stale/official (old + new editions) | poisoned/unverified | poisoned/falsely official
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_exp3")
ap.add_argument("--qid", required=True)
ap.add_argument("--copies", type=int, default=1, choices=[1, 3])
a = ap.parse_args()
df = pd.read_csv(os.path.join(a.results, "run_log_exp3.csv"), keep_default_na=False)
d = df[df.qid == a.qid]
if d.empty:
    raise SystemExit(f"no rows for {a.qid}")
pc = "minority_poison" if a.copies == 1 else "majority_poison"
ctx = [("1. Clean, current edition only", "clean", "accurate"),
       ("2. Stale but official (2025 + 2024 editions)", "benign_conflict", "accurate"),
       (f"3. Poisoned, tagged unverified ({a.copies} copy)", pc, "accurate"),
       (f"4. Poisoned, falsely tagged official ({a.copies} copy)", pc, "false_official")]
for title, cond, reg in ctx:
    print("\n" + title)
    for arm in ["baseline", "audit", "weight", "combined"]:
        r = d[(d.condition == cond) & (d.arm == arm) & (d.regime.isin([reg, "none"]))]
        if r.empty:
            continue
        r = r.iloc[0]
        extra = f"   [audit values: {r.audit_values}]" if arm == "audit" else ""
        print(f"   {arm:9s} -> {r.outcome:9s} | {r.answer[:70]}{extra}")
