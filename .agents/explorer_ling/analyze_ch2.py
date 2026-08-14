import re

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters in thesis ch2: {len(text)}")

# Let's inspect sections 2.2 to 2.4
print("\n--- Summary of Headings ---")
for line in text.split("\n"):
    if re.match(r'^\s*2\.\d+(\.\d+)?', line):
        print(line)
