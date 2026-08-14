import pypdf

reader = pypdf.PdfReader("resources/Thesis Submitted Revised 29 June PDF 2.pdf")
print("Total pages:", len(reader.pages))

# Search for tone, loanwords, orthography, phonology in the Table of contents or headings
for p in range(5, 18):
    text = reader.pages[p].extract_text()
    for line in text.split('\n'):
        if any(w in line.lower() for w in ['tone', 'loan', 'orthograph', 'borrow', 'phonol', 'vowel', 'consonant', 'chapter']):
            print(f"P{p+1}: {line}")
