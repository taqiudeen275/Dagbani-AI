import os
import re

base_dir = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm"
files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]

queries = [
    r"VITS",
    r"FastSpeech",
    r"Matcha",
    r"MMS-TTS|mms-tts|Massively Multilingual Speech",
    r"Coqui",
    r"Orpheus",
    r"XTTS",
    r"HiFi-GAN|BigVGAN|vocoder|neural vocoder",
    r"phonemizer|grapheme|phoneme|G2P|Epitran",
    r"tone|tonal|downstep|high-high|pitch",
    r"BPE|Byte-Pair|SentencePiece|byte-fallback|subword fertility|fertility",
    r"vocabulary expansion|vocab expansion",
    r"LoRA|QLoRA|Continual Pre-training|domain adaptation|Aya|Llama|Mistral|Gemma",
    r"Common Voice|Mozilla",
    r"Bible|JW300|jw\.org|bible\.com|Biblica|Open\.Bible|BibleTTS",
    r"Wikimedia|Wikipedia|Wikidata",
    r"LoresLM|LoResLM",
    r"drum|drummer|drumming|Lunsi|oral literature|Namoo",
    r"radio|broadcast|Simli|Diamond|Justice|Zaa|Tamale",
    r"UGSpeechData|WAXAL|GhanaNLP|Khaya|Spell4Wiki"
]

print(f"Found {len(files)} text files to search.")
results = {}

for q in queries:
    pat = re.compile(q, re.IGNORECASE)
    results[q] = []
    for fname in files:
        fpath = os.path.join(base_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if pat.search(line):
                results[q].append((fname, idx+1, line.strip()[:160]))

summary_path = os.path.join(base_dir, "search_summary.txt")
with open(summary_path, "w", encoding="utf-8") as out:
    for q, hits in results.items():
        out.write(f"\n==================== QUERY: {q} (Total hits: {len(hits)}) ====================\n")
        for fname, lno, snippet in hits[:15]: # show up to 15 snippets
            out.write(f"[{fname}:{lno}] {snippet}\n")

print(f"Wrote search summary to {summary_path}")
