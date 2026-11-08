"""Rebuild attack_set_v1/ (the Experiment 3 attack set in signet-harness format) from the frozen dataset.

  python build_attack_set.py --frozen exp3_frozen_v1 --out attack_set_v1

Reads: exp3_frozen_v1 (verified first: run `python freeze_exp3_dataset.py --verify exp3_frozen_v1`).
Writes: attack_set_v1/questions.json, docs.json, files/*.pdf (page excerpts), files/README.md.
Needs pypdf (already in requirements.txt). Deterministic: the same input gives the same JSON.
"""
import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter

FAMILY = "IRS-1040-INSTR"
DOCS = {  # doc_id -> (title in clean_passages.jsonl, filename, title, url, year)
    "irs-2024": ("Form 1040 Instructions 2024", "i1040gi_2024.pdf", "2024 Instructions for Form 1040",
                 "https://www.irs.gov/pub/irs-prior/i1040gi--2024.pdf", "2024"),
    "irs-2025": ("Form 1040 Instructions 2025", "i1040gi_2025.pdf", "2025 Instructions for Form 1040",
                 "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf", "2025"),
}
MODIFIED = "irs-2025-modified"
CONDITIONS = ["clean", "benign_conflict", "minority_poison", "majority_poison"]


def pid_of(frozen_id: str) -> str:
    """'Form 1040 Instructions 2025#175' -> 'irs-2025-p175'."""
    doc, n = frozen_id.rsplit("#", 1)
    return f"irs-{doc.rsplit(' ', 1)[1]}-p{n}"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9$]", "", s.lower().replace("’", "'"))


def regex_for(value: str) -> str:
    """'$10,918.20' -> '\\b10918\\.20\\b' (the scorer lower-cases and drops $ and commas)."""
    return r"\b" + re.escape(value.replace("$", "").replace(",", "")) + r"\b"


def page_texts(pdf: Path) -> list:
    return [norm(p.extract_text() or "") for p in PdfReader(str(pdf)).pages]


def find_pages(pages: list, question: str, value: str) -> list:
    """1-based pages that contain the words before the blank followed by the value."""
    sent = re.search(r'"(.*)"', question, flags=re.S).group(1)
    before = norm(" ".join(sent.split("_____")[0].split()[-4:]))
    hits = [i + 1 for i, t in enumerate(pages) if before in t and norm(value) in t]
    return hits or [i + 1 for i, t in enumerate(pages) if before + norm(value) in t]


def excerpt(src: Path, keep: list, dst: Path):
    r, w = PdfReader(str(src)), PdfWriter()
    for n in keep:
        w.add_page(r.pages[n - 1])
    with open(dst, "wb") as f:
        w.write(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="exp3_frozen_v1")
    ap.add_argument("--out", default="attack_set_v1")
    a = ap.parse_args()
    fz, out = Path(a.frozen), Path(a.out)
    (out / "files").mkdir(parents=True, exist_ok=True)

    truth = {t["id"]: t for t in json.loads((fz / "ground_truth.json").read_text(encoding="utf-8"))}
    qs = json.loads((fz / "questions.json").read_text(encoding="utf-8"))
    clean = [json.loads(l) for l in (fz / "clean_passages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    attacked = [json.loads(l) for l in (fz / "attacked_passages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    poison_by_q = {}
    for p in attacked:
        poison_by_q.setdefault(p["question_id"], []).append(p["id"])

    pdfs = {"irs-2024": fz / "source_pdfs" / "Form_1040_Instructions_2024.pdf",
            "irs-2025": fz / "source_pdfs" / "Form_1040_Instructions_2025.pdf"}
    texts = {k: page_texts(v) for k, v in pdfs.items()}

    # ---- docs.json
    docs = []
    for doc_id, (frozen_title, fname, title, url, year) in DOCS.items():
        passages = []
        for c in clean:
            if c["document"] == frozen_title:
                n = c["id"].rsplit("#", 1)[1]
                passages.append({"pid": pid_of(c["id"]), "locator": f"chunk {n}", "text": c["text"]})
        docs.append({"doc_id": doc_id, "family": FAMILY, "default_pool": False, "filename": fname, "title": title,
                     "source_url": url, "ingest": {"date": year, "version": f"Tax Year {year}", "status": "active"},
                     "passages": passages})
    docs.append({
        "doc_id": MODIFIED, "family": FAMILY, "default_pool": False, "variant_of": "irs-2025",
        "filename": "i1040gi_2025.pdf", "title": "2025 Instructions for Form 1040",
        "source_url": "planted copies of irs-2025 passages with one value replaced (Experiment 3); same filename, title and "
                      "ingest metadata as the genuine 2025 edition, so a version check cannot tell them apart",
        "ingest": {"date": "2025", "version": "Tax Year 2025", "status": "active"},
        "passages": [{"pid": p["id"], "locator": f"modified copy {p['copy_number']} of {pid_of(p['derived_from'])}",
                      "text": p["text"], "question_id": p["question_id"], "copy_number": p["copy_number"],
                      "derived_from": pid_of(p["derived_from"]), "value_replaced": p["value_replaced"],
                      "value_planted": p["value_planted"], "present_in_conditions": p["present_in_conditions"]}
                     for p in attacked]})
    (out / "docs.json").write_text(json.dumps({"docs": docs}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- questions.json + page excerpts
    questions, used = [], {"irs-2024": {1}, "irs-2025": {1}}
    for q in qs:
        t = truth[q["id"]]
        cur_pages = find_pages(texts["irs-2025"], q["question"], q["correct"])
        old_pages = find_pages(texts["irs-2024"], q["question"], q["old_value"]) if q["old_value"] else []
        used["irs-2025"].update(cur_pages)
        used["irs-2024"].update(old_pages)
        questions.append({
            "id": q["id"], "family": FAMILY, "case": "planted_copy", "verified": False,
            "question": q["question"],
            "corpus": ["irs-2024", "irs-2025", MODIFIED],
            "gold_passages": [pid_of(t["source_passage_id"])],
            "gold_answer": f"{q['correct']} (2025 Instructions for Form 1040)",
            "alt_answer": f"{q['wrong']} (planted copy)",
            "accept": [regex_for(q["correct"])], "alt": [regex_for(q["wrong"])],
            "attack": "planted_copy", "target_doc": "irs-2025", "original_value": q["correct"],
            "edited_value": q["wrong"], "prior_edition_value": q["old_value"],
            "value_changed_between_editions": bool(q["old_value"]),
            "poisoned_passages": poison_by_q.get(q["id"], []),
            "conditions": CONDITIONS,
            "source_pdf_pages": {"irs-2025": cur_pages, "irs-2024": old_pages},
            "machine_checked_against_pdf": bool(cur_pages) and (bool(old_pages) or not q["old_value"]),
        })
    (out / "questions.json").write_text(json.dumps({"questions": questions}, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8")

    for doc_id, name in (("irs-2025", "exp3_irs-2025_pages.pdf"), ("irs-2024", "exp3_irs-2024_pages.pdf")):
        excerpt(pdfs[doc_id], sorted(used[doc_id]), out / "files" / name)
    rows = "\n".join(f"| `{q['id']}` | {q['source_pdf_pages']['irs-2025']} | {q['source_pdf_pages']['irs-2024'] or 'n/a (value unchanged)'} |"
                     for q in questions)
    (out / "files" / "README.md").write_text(
        "# Source page excerpts\n\n"
        "Pages of the original IRS PDFs that each question's passage comes from (cover page 1 is always kept, because it shows\n"
        "the edition). Page numbers are the page's position in the original file, not the number printed on it. The full PDFs\n"
        "are in `exp3_frozen_v1/source_pdfs/`.\n\n"
        f"| File | Original pages kept |\n|---|---|\n"
        f"| `exp3_irs-2025_pages.pdf` | {sorted(used['irs-2025'])} |\n"
        f"| `exp3_irs-2024_pages.pdf` | {sorted(used['irs-2024'])} |\n\n"
        "Pages were located by machine (the words before the blank followed by the value), not read by a person.\n\n"
        "| Question | 2025 page(s) | 2024 page(s) |\n|---|---|---|\n" + rows + "\n", encoding="utf-8")
    print(f"{len(questions)} questions, {sum(len(d['passages']) for d in docs)} passages, "
          f"2025 pages {sorted(used['irs-2025'])}, 2024 pages {sorted(used['irs-2024'])}")


if __name__ == "__main__":
    main()
