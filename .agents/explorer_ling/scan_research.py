import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def scan_file_for_keywords(filename, keywords):
    print(f"\n==================== SCANNING {filename} ====================")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    for kw in keywords:
        matches = [m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE)]
        print(f"Keyword '{kw}': {len(matches)} occurrences")
        for idx in matches[:3]:
            start = max(0, idx - 150)
            end = min(len(text), idx + 350)
            print(f"--- [Match around pos {idx}] ---")
            print(text[start:end].replace('\n', ' '))

keywords = ['tone', 'tonal', 'orthograph', 'BGL', 'downstep', 'syllab', 'phonem', 'vowel harmony', 'G2P', 'grapheme', 'loanword', 'Hausa', 'Arabic']
for fn in [
    '.agents/explorer_ling/building_dagbani_lm.txt',
    '.agents/explorer_ling/building_dagbani_lm_tts.txt',
    '.agents/explorer_ling/breaking_low_resource_barr.txt',
    '.agents/explorer_ling/grandmasters_snippet.txt'
]:
    scan_file_for_keywords(fn, keywords)
