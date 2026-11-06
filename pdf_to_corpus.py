"""
Turn real public PDFs (e.g. IRS instructions) into a corpus folder the harness can read.

  python pdf_to_corpus.py --out corpus/irs \
      --doc path/to/i1040gi_2024.pdf  --year 2024 --title "Form 1040 Instructions 2024" \
      --doc path/to/i1040gi_2023.pdf  --year 2023 --title "Form 1040 Instructions 2023"

The newest year is treated as the current edition. Both are tagged origin=official.
"""
import argparse
import json
import os

from pypdf import PdfReader

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--doc", action="append", required=True)
ap.add_argument("--year", action="append", type=int, required=True)
ap.add_argument("--title", action="append", required=True)
a = ap.parse_args()
assert len(a.doc) == len(a.year) == len(a.title), "give --doc, --year and --title the same number of times"

os.makedirs(a.out, exist_ok=True)
docs = []
for path, year, title in zip(a.doc, a.year, a.title):
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    fn = f"{title.lower().replace(' ', '_')}.txt"
    # blank line between pages so paragraph-aware chunking has boundaries to use
    open(os.path.join(a.out, fn), "w", encoding="utf-8").write("\n\n".join(pages))
    docs.append({"file": fn, "title": title, "year": year, "origin": "official"})
    print(f"{title}: {len(pages)} pages, {sum(len(p.split()) for p in pages)} words")

json.dump({"current_year": max(a.year), "documents": docs}, open(os.path.join(a.out, "corpus.json"), "w"), indent=2)
print("wrote", os.path.join(a.out, "corpus.json"))
