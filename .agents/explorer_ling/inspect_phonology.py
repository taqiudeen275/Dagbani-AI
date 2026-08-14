import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's extract sections 2.1 through 2.4
print("=== THESIS CHAPTER 2 PHONOLOGY CONTENT ===")
print(text[:15000]) # First 15000 chars

with open(".agents/explorer_ling/thesis_ch2_extracted_phonology_part1.txt", "w", encoding="utf-8") as out:
    out.write(text[:50000])

with open(".agents/explorer_ling/thesis_ch2_extracted_phonology_part2.txt", "w", encoding="utf-8") as out:
    out.write(text[50000:100000])

with open(".agents/explorer_ling/thesis_ch2_extracted_phonology_part3.txt", "w", encoding="utf-8") as out:
    out.write(text[100000:])

print("Saved split parts 1, 2, 3")
