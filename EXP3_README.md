# Experiment 3: do the defenses still work when the trust labels lie?

Files in this update (copy them into the SAME folder as run_experiments.py; do NOT overwrite run_experiments.py):
  signet.py (updated, only adds things; Experiments 1 and 2 behave exactly as before)
  run_exp3.py, report_exp3.py, show_question.py

Arms
  baseline   plain RAG, no labels (same prompts as before)
  audit      model sees the passages with NO year/source/score and lists the value each passage states; code flags
             CONFLICT when two passages disagree. Uses no label at all.
  weight     metadata only (retain + weight, same prompts as before)
  combined   weight prompt + the audit's value list
Label regimes: accurate | noisy (20% flipped, seeds 0-4; seed 0 = the draw already used) | false_official (planted
passages tagged official, so metadata cannot catch them)

Outputs (results_exp3): run_log_exp3.csv, exp3_summary.csv, exp3_by_poison_copies.csv, exp3_noisy_by_seed.csv, fig4_exp3.png
