# Team Signet: MVP experiments (poison propagation + defenses)

Two experiments, one harness.

- **Experiment 1:** how far a wrong passage gets through a plain RAG pipeline (clean / benign conflict / 1 poisoned / 3 poisoned).
- **Experiment 2:** does a defense help? Hard filter vs retain-and-weight vs source-separated, with correct vs 20%-wrong credibility labels.

Everything is offline except the LLM calls. Requirements: `pip install numpy scikit-learn pandas matplotlib pypdf`.

## IMPORTANT
The `--backend simulated` option is a scripted stand-in, **not a language model**. It only proves the code runs.
Charts from it are stamped "SIMULATED". Only runs with a real model are results.
The bundled demo corpus is **fictional** (all numbers invented).

## Step by step

1. **Test the plumbing (1 minute).**
   `python make_demo_corpus.py && python run_experiments.py --backend simulated --out results_sim && python report.py --results results_sim`
2. **Get real documents.** Download 2 editions of one public IRS instruction document (current year + prior year), e.g. Form 1040 instructions. Then:
   `python pdf_to_corpus.py --out corpus/irs --doc cur.pdf --year 2024 --title "Form 1040 Instructions 2024" --doc old.pdf --year 2023 --title "Form 1040 Instructions 2023"`
3. **Write 20 questions** in `questions.json` (copy `questions_TEMPLATE.json`). For each:
   - `correct`: the value exactly as printed in the current edition (it must appear in that text).
   - `old_value`: the prior edition's value if it legitimately changed, else `null`. Aim for about 12 changed and 8 unchanged.
   - `wrong`: a plausible planted value. No value may be contained inside another (the script checks).
   - Check the spelling by searching the `.txt` files in `corpus/irs`. PDF extraction can split numbers oddly.
4. **First pass with a real model (cheap):**
   `python run_experiments.py --backend anthropic --model <MODEL_ID> --corpus corpus/irs --questions questions.json --out results --quick --max-questions 5`
   Needs `ANTHROPIC_API_KEY`. For a local or HiPerGator model served with vLLM/Ollama use
   `--backend openai --model <name> --base-url http://HOST:PORT/v1`. Read `results/run_log.csv` by hand. Do the answers look sane?
5. **Full run:** same command without `--quick` and `--max-questions` (about 1,700 calls for 20 questions; answers are cached in `results/llm_cache.jsonl`, so a re-run only pays for new prompts).
6. **Report:** `python report.py --results results`. This writes the tables (`exp1_summary.csv`, `exp2_summary.csv`, `exp2_filter_threshold_sweep.csv`) and three charts.
7. **Read failures by hand** (`run_log.csv`, filter on `outcome` = other/attacked) and fix any setup mistake, then do one clean rerun.

## What each outcome means
`correct`, `attacked` (planted value stated), `stale` (prior-year value), `flagged` (model reported a conflict), `other` (neither, or both values unresolved).

## What the numbers mean
- Experiment 1: `poison_in_top5_%` = retrieval success; `attack_success_%` = generation influence (PoisonedRAG's two conditions).
- Experiment 2: `attack_success_%(poisoned)` and `correct_%(clean+benign)` = security and utility; `legit_answer_chunks_dropped_%` = false rejection for the filter; `flagged_%(clean)` = false alarm.

## Limits to state
- Credibility scores use metadata (year, origin tag) that this harness assigns. It tests the *mechanism*. The noisy-label runs show how much it depends on label quality.
- Prompt-level approximations of CAG and MADAM-RAG, not the trained or multi-round originals.
- 20 questions: report counts ("4 of 5"), not precise percentages.
- A poisoned passage with no conflicting source (in-place edit) cannot be caught by a conflict check. This harness plants modified *copies*.
