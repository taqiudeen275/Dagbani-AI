import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/grandmasters_snippet.txt", "r", encoding="utf-8") as f:
    text = f.read()

pages = text.split("--- [PAGE ")
for p in pages:
    if len(p) > 0 and (p.startswith("50") or p.startswith("51") or p.startswith("52") or p.startswith("53")):
        print(f"=== GM PAGE {p[:5]} ===")
        print(p)
