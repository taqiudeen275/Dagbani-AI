import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total length: {len(text)}")

# Find all occurrences of 2.2, 2.3, 2.4, etc.
for m in re.finditer(r'(2\.[1-9](\.\d+)?[^\n]+)', text):
    print(f"Pos {m.start()}: {m.group(1)}")

# Print 2.2.1 Consonant system
pos_2_2_1 = text.find("2.2.1")
if pos_2_2_1 != -1:
    print("\n--- 2.2.1 START ---")
    print(text[pos_2_2_1:pos_2_2_1+4000])

# Print 2.3 Vowel system
pos_2_3 = text.find("2.3 The Vowel")
if pos_2_3 != -1:
    print("\n--- 2.3 START ---")
    print(text[pos_2_3:pos_2_3+8000])
