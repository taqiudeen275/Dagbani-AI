drum_path = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/Grandmasters_of_the_drum_Special_Issue_6_extracted.txt"
with open(drum_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

import re

queries = [
    r"samban",
    r"lunsi",
    r"recording|recorded|audio|tape|fieldwork",
    r"transcription|transcribe|orthography",
    r"praise poetry|panegyric",
    r"drum history|drum language|acoustic|rhythm"
]

out_path = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/drum_matches.txt"
with open(out_path, "w", encoding="utf-8") as out:
    for q in queries:
        matches = re.findall(rf".{{0,60}}{q}.{{0,60}}", text, re.IGNORECASE)
        out.write(f"\n=== Query: {q} ({len(matches)} matches) ===\n")
        for m in matches[:8]:
            out.write("  ... " + m.strip().replace("\n", " ") + " ...\n")

print("Done")
