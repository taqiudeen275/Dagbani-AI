import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/grandmasters_snippet.txt", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("2.6 Linguistic Affiliation")
if pos == -1:
    pos = text.find("Linguistic Affiliation")
if pos == -1:
    pos = text.find("Gur language")

print(text[pos:pos+10000])
