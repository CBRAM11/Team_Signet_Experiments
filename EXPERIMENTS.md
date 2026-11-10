# Experiments index (Team Signet, Trust the Source)

All runs use IRS Form 1040 Instructions (2025 current edition, 2024 prior edition), a TF-IDF top-5 retriever, and
Llama 3.1 8B served locally with Ollama at temperature 0, unless the row says simulated.

| Experiment | What it asks | Code | Results folder | Real model run? |
|---|---|---|---|---|
| 1. Poison propagation | How far does a wrong passage get through plain RAG? | `run_experiments.py`, `report.py` | `results_full/` | Yes |
| 2. Defenses | Does filtering or credibility weighting help, with correct vs 20%-wrong labels? | `run_experiments.py`, `report.py` | `results_full/` | Yes |
| 3. Lying labels | Do the defenses survive forged "official" labels? Does a label-free content audit help? | `run_exp3.py`, `report_exp3.py`, `signet.py` | `results_exp3/` | Yes |
| 4. Quiet in-place edit | With the true passage rewritten in place (no true copy present), can any check that reads no trust label notice? Four detectors: content audit, corpus cross-check, prior-version check, model prior. | `run_exp4.py`, `report_exp4.py`, `signet.py` | `results_exp4/` | Yes |

Other folders (not results):

- `results_pilot/`, `results_exp3_pilot/`: small pilot runs used to check the setup before the full runs.
- `results_sim/`, `results_check/`: simulated backend, plumbing tests only. Not experimental results.
- `exp3_frozen_v1/`: frozen Experiment 3 dataset (questions, ground truth, clean and attacked passages, retrieval sets,
  source PDFs, reference results). Dataset hash:
  `51a35e90c0f908ab14f059a408a3e5f9b35b73b808c7e8aefdcf0fe23a765083`.
  Check it with `python freeze_exp3_dataset.py --verify exp3_frozen_v1` (standard library only).
- `corpus/irs/`: the paragraph-chunked source text. `cur.pdf` and `old.pdf` are the 2025 and 2024 PDFs.
- `llm_cache.jsonl` files: cached model answers, so a re-run only pays for prompts that changed.

Experiment 3 headline (out of 40 poisoned runs): plain RAG 13 planted answers; metadata-only 3 with accurate labels and
32 with forged official labels; content audit flagged 40/40 with 2 false alarms in 31 unchanged cases. The audit is
detection only, and a genuine passage always sat beside the poisoned one (limitation).

Experiment 4 headline (20 questions, Llama 3.1 8B, 40 in-place edit runs): plain RAG repeated the planted number in 40/40.
The content audit caught 0 of 32 purely single-source edits. A prior-version check on the 11 stable-value questions
caught 17/22 edited runs with 0/11 false flags; with the content audit, 22/22 with 1/11 false flag. On the 9
changed-value questions nothing worked (0/18). The corpus cross-check and model-prior check flagged 55% and 70% of clean
runs, so they are not usable here. One model, 20 questions; a rerun on DeepSeek V4.1 Flash is pending. Full write-up:
`reports/Trust_the_Source_Experiment4_Report.docx` (and `.pdf`).

Setup and run steps: `README.md` (Experiments 1 and 2) and `EXP3_README.md` (Experiment 3), `EXP4_README.md` (Experiment 4, including the DeepSeek rerun).
