import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def print_section(title, filename, regex_pattern, max_chars=4000):
    print(f"================== {title} ==================")
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(regex_pattern, content, re.IGNORECASE | re.DOTALL)
    if match:
        snippet = match.group(0)[:max_chars]
        print(snippet)
    else:
        print("Pattern not found!")

# Let's inspect vowel system and vowel harmony in thesis ch2
print_section("VOWEL SYSTEM & HARMONY", ".agents/explorer_ling/thesis_ch2_extracted_phonology_part1.txt", r"2\.3\s+The Vowel System of Dagbani.*?(?=2\.4\s+The Syntactic)", max_chars=8000)

