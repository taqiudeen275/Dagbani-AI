drum_path = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/Grandmasters_of_the_drum_Special_Issue_6_extracted.txt"
with open(drum_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(f"Drum text length: {len(text)} chars")
lines = text.split("\n")
print("\n--- FIRST 60 LINES ---")
for l in lines[:60]:
    if l.strip():
        print(l)
