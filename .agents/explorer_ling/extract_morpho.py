import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for Tone in thesis ch2
for m in re.finditer(r'(tone|tonal|pitch|mora|downstep)', text, re.IGNORECASE):
    idx = m.start()
    print("--- MATCH ON TONE ---")
    print(text[max(0, idx-100):min(len(text), idx+200)].replace('\n', ' '))

# Let's search for morphophonology, noun classes, pronouns
pos_syn = text.find("2.4 The Syntactic and Verbal Systems of Dagbani")
if pos_syn != -1:
    print("\n=== SYNTACTIC AND MORPHOLOGICAL SYSTEM (2.4) ===")
    print(text[pos_syn:pos_syn+8000])

