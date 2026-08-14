import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

pos_2_4_5 = text.find("2.4.5")
if pos_2_4_5 != -1:
    print(text[pos_2_4_5:pos_2_4_5+15000])
else:
    print("2.4.5 not found, printing from page 80 onwards")
    pages = text.split("--- [PAGE ")
    for p in pages:
        if any(p.startswith(str(num)) for num in range(80, 102)):
            print(f"=== PAGE {p[:5]} ===")
            print(p[:2000])

