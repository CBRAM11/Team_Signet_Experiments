# Experiment 3 attack set, signet-harness format (v1)

The 20 Experiment 3 questions and their planted passages, converted from `exp3_frozen_v1/` into the shared harness format
(`questions.json` + `docs.json` + `files/`). Rebuild it with `python build_attack_set.py` (needs `pypdf`).

**Hash to cite** (sha256 of `questions.json`, printed by `signet check-questions --file attack_set_v1/questions.json`):
`db408f166027c8850e142b36d2915afabca7db73ab00f699cb2eaa4582e2838c`

Source of truth is still `exp3_frozen_v1` (dataset hash `51a35e90c0f908ab14f059a408a3e5f9b35b73b808c7e8aefdcf0fe23a765083`).
This folder only reshapes it.

## What is in it
- `questions.json`: 20 questions, case `planted_copy`. Extra fields kept: `attack`, `target_doc`, `original_value`, `edited_value`,
  `prior_edition_value`, `poisoned_passages`, `conditions` (clean, benign_conflict, minority_poison, majority_poison), `source_pdf_pages`.
- `docs.json`: `irs-2024` (1,042 passages), `irs-2025` (1,158) and `irs-2025-modified` (60 planted copies: 3 per question).
  Passage ids: `irs-2025-p175` is chunk 175 of the 2025 edition. Planted copies keep their Experiment 3 ids (`poison-q01-0`).
- `files/`: page excerpts of the two IRS PDFs that the questions come from, plus a table of which page each question uses.

## Read before using
- `verified` is `false` on purpose. A script checked that each correct value and prior-edition value appears on the PDF page
  and that each planted copy holds the edited value and not the original. A person has not read every question yet.
- `irs-2025-modified` has the same filename, title and ingest metadata as the genuine 2025 edition. So a version check
  (the Track 2 freshness check) cannot tell them apart. That is the single-source gap, not a bug.
- Experiment 3 used its own TF-IDF retrieval, with a genuine copy always retrieved next to the planted one. The harness default
  is hybrid BM25 plus dense retrieval, so results through the harness are not directly comparable with the Experiment 3 numbers.
  The exact Experiment 3 retrieval sets are in `exp3_frozen_v1/retrieval_sets.jsonl`.
- Experiment 3's trust labels (accurate, noisy, forged official) are not in this format. They live in `retrieval_sets.jsonl`.
