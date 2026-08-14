import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/thesis_ch2_phonology.txt", "r", encoding="utf-8") as f:
    text = f.read()

pos_2_3_1 = text.find("2.3.1 Vowel Harmony")
pos_2_4 = text.find("2.4 The Syntactic")

print("=== 2.3.1 VOWEL HARMONY & SURROUNDING SECTIONS ===")
print(text[pos_2_3_1:pos_2_4])
