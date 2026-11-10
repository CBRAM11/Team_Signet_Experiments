# Experiment 4: single-source in-place edit

Plan: Track 1 Next Steps (Oct 23), section 2. Question: can anything catch a quiet edit when no true copy sits next to it?

Copy `run_exp4.py`, `report_exp4.py` and this file into the SAME folder as `signet.py`. Nothing existing is changed.

## Run (Windows, Ollama running with llama3.1:8b)

    mkdir results_exp4
    copy results_exp3\llm_cache.jsonl results_exp4\llm_cache.jsonl
    python run_exp4.py --backend openai --model llama3.1:8b --base-url http://localhost:11434/v1 --corpus corpus\irs --questions questions.json --out results_exp4 --max-questions 3
    (look at the three questions: check "edited" output lines, then delete results_exp4\run_log_exp4.csv and run again without --max-questions)
    python report_exp4.py --results results_exp4

About 520 model calls in total. The clean and benign prompts are reused from the Experiment 3 cache.

## What changes from Experiment 3
The 2025 document text is edited where the best passage states the true value, then re-chunked. Only the edited wording exists
at that spot. Experiment 3 added a modified copy beside the genuine passage.

| Condition | Library | Edit |
|---|---|---|
| clean | 2025 | none (false-flag control) |
| benign_conflict | 2024 + 2025 | none (false-flag control with legitimate edition differences) |
| inplace_1 | 2025 | one passage edited in place (**the test**) |
| inplace_3 | 2025 | edited in place + 2 agreeing copies |
| mixed_inplace_1 / _3 | 2024 + 2025 | same, but the 2024 edition can disagree for a legitimate reason (secondary) |

## Detectors (none reads a trust label)
- D1 content audit: the model lists the value each retrieved passage states; code flags disagreement (as in Experiment 3).
- D2 corpus cross-check: the same audit on 10 more 2025 passages from elsewhere in the library. Flags when none of the retrieved values is confirmed there.
- D3 prior version: the same audit on the 2024 edition. Flags when no retrieved value matches. Applied only to values that normally do not change (no old_value); also logged for all questions as "D3 (all)".
- D4 model prior: the model answers with no passages shown. Flags when its answer is not among the retrieved values. Will not carry over to private military documents.

## Reading the report
- Detection is reported for all in-place runs and for the subset where the true wording is nowhere in the retrieved passages (the pure single-source case).
- False flags come from the clean and benign runs. Coverage is the share of questions where the detector had something to compare against.
- Run with `--backend simulated` only to test the plumbing: those numbers are not results (the report says so).

## Rerun on DeepSeek V4.1 Flash (through OpenRouter)

Same script, same questions, same prompts. Only the model changes. Use a separate output folder so the Llama results are not overwritten.

PowerShell:

    $env:OPENAI_API_KEY = "<your OpenRouter key>"
    python run_exp4.py --backend openai --model deepseek/deepseek-v4.1-flash --base-url https://openrouter.ai/api/v1 --corpus corpus\irs --questions questions.json --out results_exp4_deepseek
    python report_exp4.py --results results_exp4_deepseek

Pilot first with `--max-questions 3` (use `--out results_exp4_deepseek_pilot`). The full run makes about 480 model calls before caching, each about 1,000 input tokens, so it costs well under $1 at the harness price of $0.14 / $0.42 per million tokens.

Settings: temperature 0, max 300 output tokens, the model's default reasoning (the shared harness pins `reasoning: None` for this model, so this matches). The run does not go through the shared harness package, so there is no trace or cost log; the model slug is recorded in run_config_exp4.json. Report these numbers beside the Llama ones, not merged with them.
