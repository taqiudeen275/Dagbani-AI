import os
import re

thesis_path = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/Thesis Submitted Revised 29 June PDF 2_extracted.txt"
with open(thesis_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(f"Thesis length: {len(text)} characters")

# Find table of contents or main sections
toc_matches = re.findall(r'(?:TABLE OF CONTENTS|Table of Contents|Contents).*?(?:CHAPTER|Chapter|1\.)', text[:15000], re.DOTALL)
if toc_matches:
    print("Found TOC snippet:")
    print(toc_matches[0][:2000])

# Search for specific terms
keywords = ["orthography", "phonology", "tone", "vowel harmony", "radio", "oral literature", "drum", "lunsi", "corpus", "dataset", "folktales"]
for kw in keywords:
    count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE))
    print(f"Keyword '{kw}': {count} occurrences")
