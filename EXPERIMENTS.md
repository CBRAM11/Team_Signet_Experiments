# Experiments index (Team Signet, Trust the Source)

All runs use IRS Form 1040 Instructions (2025 current edition, 2024 prior edition), a TF-IDF top-5 retriever, and
Llama 3.1 8B served locally with Ollama at temperature 0, unless the row says simulated.

| Experiment | What it asks | Code | Results folder | Real model run? |
|---|---|---|---|---|
| 1. Poison propagation | How far does a wrong passage get through plain RAG? | `run_experiments.py`, `report.py` | `results_full/` | Yes |
| 2. Defenses | Does filtering or credibility weighting help, with correct vs 20%-wrong labels? | `run_experiments.py`, `report.py` | `results_full/` | Yes |
| 3. Lying labels | Do the defenses survive forged "official" labels? Does a label-free content audit help? | `run_exp3.py`, `report_exp3.py`, `signet.py` | `results_exp3/` | Yes |

Other folders (not results):

- `results_pilot/`, `results_exp3_pilot/`: small pilot runs used to check the setup before the full runs.
- `results_sim/`, `results_check/`: simulated backend, plumbing tests only. Not experimental results.
- `exp3_frozen_v1/`: frozen Experiment 3 dataset (questions, ground truth, clean and attacked passages, retrieval sets,
  source PDFs, reference results). Dataset hash:
  `51a35e90c0f908ab14f059a408a3e5f9b35b73b808c7e8aefdcf0fe23a765083`.
  Check it with `python freeze_exp3_dataset.py --verify exp3_frozen_v1` (standard library only).
- `attack_set_v1/`: the same 20 questions and planted passages in the shared signet-harness format
  (`questions.json`, `docs.json`, `files/`). Question-set hash:
  `db408f166027c8850e142b36d2915afabca7db73ab00f699cb2eaa4582e2838c`. Rebuild with `python build_attack_set.py`.
- `corpus/irs/`: the paragraph-chunked source text. `cur.pdf` and `old.pdf` are the 2025 and 2024 PDFs.
- `llm_cache.jsonl` files: cached model answers, so a re-run only pays for prompts that changed.

Experiment 3 headline (out of 40 poisoned runs): plain RAG 13 planted answers; metadata-only 3 with accurate labels and
32 with forged official labels; content audit flagged 40/40 with 2 false alarms in 31 unchanged cases. The audit is
detection only, and a genuine passage always sat beside the poisoned one (limitation).

Setup and run steps: `README.md` (Experiments 1 and 2) and `EXP3_README.md` (Experiment 3).
