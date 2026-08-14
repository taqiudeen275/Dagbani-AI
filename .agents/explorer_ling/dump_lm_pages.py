import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/building_dagbani_lm.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's print pages 1 to 4
pages = text.split("--- [PAGE ")
for p in pages[1:5]:
    print(f"=== PAGE {p[:10]} ===")
    print(p[:3000])

with open(".agents/explorer_ling/building_dagbani_lm_tts.txt", "r", encoding="utf-8") as f:
    text2 = f.read()

pages2 = text2.split("--- [PAGE ")
for p in pages2[1:5]:
    print(f"=== TTS DOC PAGE {p[:10]} ===")
    print(p[:3000])
