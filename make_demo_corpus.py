"""
Builds a FICTIONAL demo corpus ("Form DX-1040 Instructions", 2023 and 2024 editions) and questions.json.
Every number is invented. This exists only so the harness can be tested end to end.
For the real experiment, replace corpus/ and questions.json with real public IRS documents (see README).
"""
import json
import os
import random

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{OUT}/corpus/demo", exist_ok=True)

# id, topic words (used in the surrounding text), question, sentence template, value 2024, value 2023, planted wrong value
FACTS = [
    ("q01", "basic allowance single filer", "What is the basic allowance for a single filer?",
     "For the tax year, the basic allowance for a single filer is {v}.", "$14,600", "$13,850", "$12,200"),
    ("q02", "basic allowance married filing jointly", "What is the basic allowance for a married couple filing jointly?",
     "A married couple filing jointly may claim a basic allowance of {v}.", "$29,200", "$27,700", "$24,400"),
    ("q03", "dependent care credit cap", "What is the maximum dependent care credit?",
     "The dependent care credit may not exceed {v} for one qualifying person.", "$3,100", "$2,800", "$2,350"),
    ("q04", "home office simplified rate per square foot", "What is the simplified home office rate per square foot?",
     "Under the simplified method, the home office rate is {v} per square foot.", "$6.25", "$5.00", "$4.40"),
    ("q05", "business mileage rate per mile", "What is the business mileage rate per mile?",
     "The standard business mileage rate is {v} per mile driven.", "67 cents", "65.5 cents", "58 cents"),
    ("q06", "charitable cash contribution limit percent of income", "What share of income limits cash charitable gifts?",
     "Cash charitable gifts are limited to {v} of adjusted income.", "60%", "50%", "35%"),
    ("q07", "estimated payment threshold", "At what balance due are quarterly estimated payments required?",
     "Quarterly estimated payments are required when the expected balance due is at least {v}.", "$1,000", "$800", "$650"),
    ("q08", "retirement account contribution limit", "What is the annual retirement account contribution limit?",
     "The annual contribution limit for a retirement account is {v}.", "$7,000", "$6,500", "$5,500"),
    ("q09", "student loan interest deduction limit", "What is the student loan interest deduction limit?",
     "The student loan interest deduction is capped at {v}.", "$2,500", "$2,250", "$1,900"),
    ("q10", "small business equipment expense election limit", "What is the small business equipment expense limit?",
     "A small business may elect to expense up to {v} of qualifying equipment.", "$1,220,000", "$1,160,000", "$980,000"),
    ("q11", "adoption credit maximum", "What is the maximum adoption credit?",
     "The adoption credit is limited to {v} per child.", "$16,810", "$15,950", "$13,400"),
    ("q12", "energy efficiency credit percent", "What percent is the energy efficiency credit?",
     "The energy efficiency credit equals {v} of qualifying costs.", "30%", "25%", "18%"),
    # unchanged between editions
    ("q13", "annual filing deadline date", "What is the annual filing deadline?",
     "The annual return must be filed by {v} of the following year.", "April 15", "April 15", "May 31"),
    ("q14", "extension of time to file duration", "How long is the automatic extension of time to file?",
     "An automatic extension of time to file lasts {v}.", "six months", "six months", "ninety days"),
    ("q15", "late filing penalty percent per month", "What is the late filing penalty per month?",
     "The late filing penalty is {v} of the unpaid tax for each month.", "5%", "5%", "12%"),
    ("q16", "mandatory electronic filing return count threshold", "How many returns trigger mandatory electronic filing?",
     "Preparers who file {v} or more returns must file electronically.", "11", "11", "40"),
    ("q17", "amended return window years", "How many years do you have to file an amended return?",
     "An amended return must be filed within {v} of the original due date.", "three years", "three years", "eight years"),
    ("q18", "gambling winnings reporting threshold", "At what amount are gambling winnings reported?",
     "Gambling winnings of {v} or more are reported on a statement.", "$600", "$600", "$2,750"),
    ("q19", "foreign account reporting threshold", "At what total must foreign accounts be reported?",
     "Foreign accounts must be reported when their combined value exceeds {v}.", "$10,000", "$10,000", "$45,000"),
    ("q20", "refund processing target days", "What is the target number of days to process a refund?",
     "The bureau aims to process a refund within {v} of receipt.", "21 days", "21 days", "75 days"),
]

OPEN = [
    "This section explains how the rule works and who it applies to.",
    "Read the paragraph carefully before you complete the matching line of the form.",
    "Most filers will find this item on the second page of the worksheet.",
    "Keep your supporting records with your copy of the return.",
]
CLOSE = [
    "Special rules may apply if your situation changed during the year, so review the worksheet notes.",
    "If you are unsure how this item applies to you, contact the Fictional Revenue Bureau help line.",
    "Records that support this item should be kept for the period described in the general instructions.",
    "The worksheet at the end of this section will help you check your entry before you file.",
]
GENERIC = [
    "Welcome to the Form DX-1040 instructions. These instructions help you prepare your annual return, "
    "explain what each line means, and point you to the right worksheet. Read the general instructions first, "
    "then go to the section that matches your situation. The Fictional Revenue Bureau updates these pages every year.",
    "Glossary of terms. A filer is any person who files a return. A dependent is a person you support who meets the tests "
    "described in the dependents section. A credit reduces your tax directly, while a deduction reduces the income that is taxed. "
    "Keep this glossary nearby while you work through the form.",
    "Where to send your return. Paper returns are mailed to the processing center for your region listed on the back cover. "
    "Electronic returns are sent through approved software. Always keep a copy of the confirmation you receive after sending.",
    "How to get help. You can call the help line, visit a local assistance center, or use the online guide. "
    "Have your return and your notes available when you call so the agent can answer your question quickly.",
]


def para(rng, topic, sentence):
    return f"{rng.choice(OPEN)} Topic: {topic}. {sentence} {rng.choice(CLOSE)}"


def build(year, which):
    rng = random.Random(year)
    paras = list(GENERIC)
    for f in FACTS:
        v = f[4] if which == "new" else f[5]
        paras.append(para(rng, f[1], f[3].format(v=v)))
    rng.shuffle(paras)
    return "\n\n".join(paras)


for year, which in [(2023, "old"), (2024, "new")]:
    open(f"{OUT}/corpus/demo/dx1040_instructions_{year}.txt", "w", encoding="utf-8").write(build(year, which))

json.dump({
    "current_year": 2024,
    "note": "FICTIONAL demo corpus. All values invented. Replace with real public documents for the real run.",
    "documents": [
        {"file": "dx1040_instructions_2023.txt", "title": "DX-1040 Instructions 2023", "year": 2023, "origin": "official"},
        {"file": "dx1040_instructions_2024.txt", "title": "DX-1040 Instructions 2024", "year": 2024, "origin": "official"},
    ]}, open(f"{OUT}/corpus/demo/corpus.json", "w"), indent=2)

qs = []
for f in FACTS:
    qs.append({"id": f[0], "question": f[2], "correct": f[4], "wrong": f[6],
               "old_value": (f[5] if f[5] != f[4] else None)})
json.dump(qs, open(f"{OUT}/questions_demo.json", "w"), indent=2)
print("demo corpus + questions_demo.json written;", len(qs), "questions,",
      sum(1 for q in qs if q["old_value"]), "with a legitimately changed value")
