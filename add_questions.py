import json
qs = json.load(open("questions.json", encoding="utf-8"))
P = 'According to the 2025 instructions, what amount fills the blank? "'
new = [
 ("q21", "This includes any amount (up to _____) of a one-time QCD to a split-interest entity (SIE).", "$54,000", "$43,200", "$53,000"),
 ("q22", "The person had gross income that was equal to or more than the exemption amount for that year or _____ for 2025.", "$5,200", "$4,160", "$5,050"),
 ("q23", "You can exclude from income the smaller of the amount of the premiums paid or _____.", "$3,000", "$2,400", None),
 ("q24", "In addition to any other penalties, the law imposes a penalty of _____ for filing a frivolous return.", "$5,000", "$4,000", None),
 ("q25", "A gift or bequest from a foreign individual or foreign estate (including foreign persons related to that foreign individual or foreign estate) totaling more than _____.", "$100,000", "$80,000", None),
 ("q26", "If you and your spouse are filing jointly and both of you were eligible educators, the maximum deduction is _____.", "$600", "$480", None),
 ("q27", "You can ask the IRS to figure the taxable part for you for a _____ fee.", "$1,000", "$800", None),
 ("q28", "If you are married filing jointly and either you or your spouse had wages or RRTA compensation of more than _____, your employer may have withheld Additional Medicare Tax even if you do not owe the tax.", "$200,000", "$160,000", None),
]
for i, s, c, w, o in new:
    qs.append({"id": i, "question": P + s + '"', "correct": c, "wrong": w, "old_value": o})
json.dump(qs, open("questions.json", "w", encoding="utf-8"), indent=2)
print(len(qs), "questions saved;", sum(1 for q in qs if q["old_value"]), "changed")
