import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/building_dagbani_lm.txt", "r", encoding="utf-8") as f:
    text = f.read()

pages = text.split("--- [PAGE ")
for p in pages[1:4]:
    print(f"=== LM DOC PAGE {p[:10]} ===")
    print(p)
