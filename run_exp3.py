"""
Experiment 3: do the defenses still work when the trust labels lie?

Same corpus, questions, retriever and model as Experiments 1 and 2. What is new:
  * label regimes: accurate | noisy (20% of tags flipped, several seeds) | false_official (planted passages tagged official)
  * 'audit' arm: content-only check. The model sees the passages WITHOUT year / source / score and lists the value
    each passage states; code then flags a CONFLICT if two passages disagree. Independent of every label.
  * 'combined' arm: the credibility-weighted prompt plus the audit's value list.
  * 'baseline' and 'weight' arms are rerun with the same prompts as before, so cached answers are reused.

  python run_exp3.py --backend openai --model llama3.1:8b --base-url http://localhost:11434/v1 ^
      --corpus corpus\\irs --questions questions.json --out results_exp3
  (add --quick --max-questions 5 for a pilot; --backend simulated to test the plumbing)
Tip: copy results_full\\llm_cache.jsonl into the new --out folder first. Prompts that did not change are then free.
"""
import argparse
import csv
import json
import os
import time

import signet as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="simulated", choices=["simulated", "anthropic", "openai"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--corpus", default="corpus/demo")
    ap.add_argument("--questions", default="questions_demo.json")
    ap.add_argument("--out", default="results_exp3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="label-noise draws (0 = the draw used before)")
    ap.add_argument("--quick", action="store_true", help="accurate + false_official + one noisy seed only")
    ap.add_argument("--max-questions", type=int)
    a = ap.parse_args()
    if a.backend != "simulated" and not a.model:
        ap.error("--model is required for a real backend")

    os.makedirs(a.out, exist_ok=True)
    cur_year, docs = S.load_corpus(a.corpus)
    base = S.build_chunks(docs)
    retr = S.Retriever(base)
    questions = json.load(open(a.questions, encoding="utf-8"))
    if a.max_questions:
        questions = questions[:a.max_questions]
    S.check_questions(questions)
    llm = S.LLM(a.backend, a.model, a.base_url,
                cache_path=None if a.backend == "simulated" else os.path.join(a.out, "llm_cache.jsonl"))

    seeds = a.seeds[:1] if a.quick else a.seeds
    # regime = (name, label_mode, seed). Label-independent arms run once under regime "none".
    regimes = [("accurate", "gold", 0)] + [(f"noisy_s{s}", "noisy", s) for s in seeds] + [("false_official", "false_official", 0)]

    rows = []
    t_start = time.time()
    for qi, q in enumerate(questions):
        for cond in S.CONDITIONS:
            pool = S.build_pool(base, q, cond, cur_year, retr)
            ranked = retr.rank(q["question"], pool, a.k)
            max_sim = max((s for _, s in ranked), default=0.0)
            poison_ranks = [r + 1 for r, (c, _) in enumerate(ranked) if c.poisoned]
            ids = ";".join(f"{r + 1}:{c.cid}" for r, (c, _) in enumerate(ranked))

            def make_items(lm, seed):
                out = []
                for r, (c, sim) in enumerate(ranked):
                    lab = S.label_origin(c, q["id"], cond, lm, a.noise, seed)
                    rel = sim / max_sim if max_sim > 0 else 0.0
                    out.append({"text": c.text, "doc": c.doc, "year": c.year, "origin": lab, "true_origin": c.origin,
                                "cred": S.credibility(sim, max_sim, c, cur_year, lab), "rel": round(rel, 3),
                                "poisoned": c.poisoned, "rank": r + 1})
                return out

            def row(arm, regime, lm, seed, ans, raw, calls0, t0, items, audit_vals, extra_calls=0):
                legit_ans = [it for it in items if not it["poisoned"]
                             and (S._norm(q["correct"]) in S._norm(it["text"])
                                  or (q.get("old_value") and S._norm(q["old_value"]) in S._norm(it["text"])))]
                rows.append({
                    "arm": arm, "regime": regime, "label_mode": lm, "seed": seed,
                    "qid": q["id"], "condition": cond, "changed": int(bool(q.get("old_value"))),
                    "outcome": S.classify(ans, q), "answer": ans,
                    "audit_values": "|".join(audit_vals) if audit_vals else "",
                    "audit_flag": int(S.parse_answer(S.audit_decision(audit_vals)).lower().startswith("conflict")) if audit_vals else "",
                    "poison_in_topk": int(bool(poison_ranks)), "poison_best_rank": min(poison_ranks) if poison_ranks else "",
                    "retrieved": ids,
                    "tags": ";".join(it["origin"][0] for it in items),
                    "poison_tag": ";".join(it["origin"] for it in items if it["poisoned"]),
                    "scores": ";".join(f'{it["cred"]:.2f}' for it in items),
                    "legit_answer_chunks": len(legit_ans),
                    "llm_calls": llm.calls - calls0 + extra_calls, "seconds": round(time.time() - t0, 3),
                    "raw": (raw or "").replace("\n", " ")[:400],
                })

            # ---- label-independent arms ----
            items = make_items("gold", 0)
            c0, t0 = llm.calls, time.time()
            raw = llm.complete(S.SYS, S.fmt_plain(items) + f"\n\nQuestion: {q['question']}", {"task": "concat", "q": q, "chunks": items})
            row("baseline", "none", "-", "", S.parse_answer(raw), raw, c0, t0, items, None)

            c0, t0 = llm.calls, time.time()
            ap_ = "\n\n".join(f"[Passage {i + 1}]\n{it['text']}" for i, it in enumerate(items)) + f"\n\nQuestion: {q['question']}"
            araw = llm.complete(S.SYS_AUDIT, ap_, {"task": "audit", "q": q, "chunks": items})
            avals = S.parse_audit(araw, len(items))
            row("audit", "none", "-", "", S.parse_answer(S.audit_decision(avals)), araw, c0, t0, items, avals)

            # ---- label-dependent arms, one set per regime ----
            for regime, lm, seed in regimes:
                items = make_items(lm, seed)
                c0, t0 = llm.calls, time.time()
                raw = llm.complete(S.SYS_WEIGHT, S.fmt_weighted(items) + f"\n\nQuestion: {q['question']}",
                                   {"task": "weighted", "q": q, "chunks": items})
                row("weight", regime, lm, seed, S.parse_answer(raw), raw, c0, t0, items, None)

                c0, t0 = llm.calls, time.time()
                note = S.fmt_audit_note(avals)
                prompt = (S.fmt_weighted(items) + "\n\nContent check (ignores the labels):\n" + note +
                          f"\n\nQuestion: {q['question']}")
                raw = llm.complete(S.SYS_COMBINED, prompt, {"task": "weighted", "q": q, "chunks": items})
                row("combined", regime, lm, seed, S.parse_answer(raw), raw, c0, t0, items, avals, extra_calls=1)  # +1 = the audit call
        print(f"question {qi + 1}/{len(questions)} done ({time.time() - t_start:.0f}s, {llm.calls} calls)", flush=True)

    path = os.path.join(a.out, "run_log_exp3.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"backend": a.backend, "model": a.model, "k": a.k, "noise": a.noise, "seeds": seeds,
               "n_questions": len(questions), "corpus": a.corpus, "regimes": [r[0] for r in regimes],
               "simulated_warning": "SIMULATED backend: plumbing test only, not experimental results."
               if a.backend == "simulated" else None},
              open(os.path.join(a.out, "run_config_exp3.json"), "w"), indent=2)
    print("wrote", path, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
