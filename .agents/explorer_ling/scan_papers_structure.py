import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/building_dagbani_lm.txt", "r", encoding="utf-8") as f:
    lm_txt = f.read()

print("=== BUILDING DAGBANI LM (Outline & Sections) ===")
for line in lm_txt.split("\n"):
    if len(line.strip()) > 0 and (line.strip().startswith("1") or line.strip().startswith("2") or line.strip().startswith("3") or line.strip().startswith("4") or line.strip().startswith("5") or line.strip().startswith("6") or line.strip().startswith("7") or line.strip().startswith("8") or "Section" in line or "Chapter" in line or line.isupper()):
        print(line[:100])

print("\n" + "="*50 + "\n")
with open(".agents/explorer_ling/building_dagbani_lm_tts.txt", "r", encoding="utf-8") as f:
    tts_txt = f.read()

print("=== BUILDING DAGBANI LM AND TTS (Outline & Sections) ===")
for line in tts_txt.split("\n"):
    if len(line.strip()) > 0 and (line.strip().startswith("1") or line.strip().startswith("2") or line.strip().startswith("3") or line.strip().startswith("4") or line.strip().startswith("5") or line.strip().startswith("6") or line.strip().startswith("7") or line.strip().startswith("8") or "Section" in line or "Chapter" in line or line.isupper()):
        print(line[:100])
