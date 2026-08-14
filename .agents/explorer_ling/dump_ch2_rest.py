import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

pos_2_4_2 = text.find("2.4.2 The Verb")
print(text[pos_2_4_2:pos_2_4_2+15000])

