import sys
import re
import pypdf

sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader("resources/Thesis Submitted Revised 29 June PDF 2.pdf")

loanword_matches = []
for idx, page in enumerate(reader.pages):
    txt = page.extract_text()
    if any(k in txt.lower() for k in ["loanword", "loan word", "borrowed", "borrowing", "adaptation", "epenthesis"]):
        for line in txt.split("\n"):
            if any(k in line.lower() for k in ["loan", "borrow", "epenth", "arabic", "hausa", "english"]):
                loanword_matches.append(f"P{idx+1}: {line}")

print(f"Total matching lines: {len(loanword_matches)}")
for l in loanword_matches[:30]:
    print(l)
