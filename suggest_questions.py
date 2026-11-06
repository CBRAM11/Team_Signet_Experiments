"""
Drafts questions.json from two real editions of the same document, so you only have to review it.

  python suggest_questions.py --corpus corpus\\irs --out questions_draft.json

How it works
  1. Finds sentences in the CURRENT edition that contain exactly one dollar amount.
  2. Looks for the same sentence in the PRIOR edition (matching on the words, ignoring the numbers and years).
       amount differs  -> "changed"  (becomes a benign-conflict question; old_value is filled in)
       amount same     -> "stable"
  3. Turns each sentence into a fill-in-the-blank question, plants a believable wrong amount,
     and writes about 12 changed + 8 stable questions spread across the document.
It also writes questions_candidates.csv with every candidate so you can swap any you dislike.
ALWAYS read the draft and delete or edit anything odd (PDF text can be messy).
"""
import argparse
import csv
import json
import os
import re

AMT = re.compile(r"\$\d[\d,]*(?:\.\d+)?")


BAD = re.compile(r"\.{4,}|\b(for example|example|suppose|assume|worksheet line|line \d+[a-z]?\b.*\bline \d+)\b", re.I)


def clean(s):
    s = re.sub(r"(\w)- (\w)", r"\1\2", s)      # "contribu- tions" -> "contributions"
    s = re.sub(r"(\w) -(\w)", r"\1-\2", s)      # "W -2" -> "W-2"
    s = re.sub(r"(\s\d{1,2}\.)+$", "", s)         # trailing "22. 23."
    return s.strip()


def sentences(text):
    t = re.sub(r"\s+", " ", text)
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t):
        s = clean(s.strip())
        if 60 <= len(s) <= 260 and not BAD.search(s):
            yield s


def key(s):
    k = AMT.sub("$X", s.lower())
    k = re.sub(r"\b20\d\d\b", "YYYY", k)
    return re.sub(r"[^a-z$ ]", "", k).strip()


def wrong_for(amount, taken):
    n = float(amount.replace("$", "").replace(",", ""))
    for f in (0.8, 1.25, 0.6, 1.5, 0.7):
        w = n * f
        txt = f"${w:,.0f}" if n == int(n) else f"${w:,.2f}"
        if txt not in taken and txt.replace("$", "").replace(",", "") not in amount.replace(",", ""):
            return txt
    return f"${n + 1234:,.0f}"


def spread(items, n):
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="questions_draft.json")
    ap.add_argument("--changed", type=int, default=12)
    ap.add_argument("--stable", type=int, default=8)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.corpus, "corpus.json")))
    cur_year = meta["current_year"]
    texts = {d["year"]: open(os.path.join(a.corpus, d["file"]), encoding="utf-8").read() for d in meta["documents"]}
    prior_year = max(y for y in texts if y < cur_year)

    def amounts_by_key(text):
        d = {}
        for s in sentences(text):
            am = set(AMT.findall(s))
            if len(am) == 1:
                d.setdefault(key(s), []).append((s, next(iter(am))))
        return d

    old = amounts_by_key(texts[prior_year])
    cur = amounts_by_key(texts[cur_year])

    cands = []
    skipped_ambiguous = 0
    for k, lst in cur.items():
        if len(k.split()) < 8:
            continue
        cur_amts = {a_ for _, a_ in lst}
        if k not in old:
            continue
        old_amts = {a_ for _, a_ in old[k]}
        if len(cur_amts) > 1 or len(old_amts) > 1:   # same wording, several amounts (e.g. per filing status)
            skipped_ambiguous += 1
            continue
        s, amount = lst[0]
        old_amt = next(iter(old_amts))
        status = "stable" if old_amt == amount else "changed"
        cands.append({"status": status, "sentence": s, "correct": amount,
                      "old_value": old_amt if status == "changed" else None})
    print(f"skipped {skipped_ambiguous} ambiguous sentences (same wording, several amounts)")

    changed = spread([c for c in cands if c["status"] == "changed"], a.changed)
    stable = spread([c for c in cands if c["status"] == "stable"], a.stable)

    with open("questions_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["status", "correct", "old_value", "sentence"])
        w.writeheader()
        w.writerows(cands)

    qs = []
    for i, c in enumerate(changed + stable, 1):
        blank = c["sentence"].replace(c["correct"], "_____")
        taken = {c["correct"], c["old_value"] or ""}
        qs.append({
            "id": f"q{i:02d}",
            "question": f'According to the {cur_year} instructions, what amount fills the blank? "{blank}"',
            "correct": c["correct"],
            "wrong": wrong_for(c["correct"], taken),
            "old_value": c["old_value"],
        })
    json.dump(qs, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"candidates found: {sum(c['status']=='changed' for c in cands)} changed, "
          f"{sum(c['status']=='stable' for c in cands)} stable")
    print(f"wrote {a.out} with {len(qs)} questions ({len(changed)} changed, {len(stable)} stable)")
    print("wrote questions_candidates.csv (all candidates). Review the draft before running the experiment.")


if __name__ == "__main__":
    main()
