import re
import os

files = [
    "Building Dagbani Language Model and TTS_extracted.txt",
    "Building Dagbani Language Models_extracted.txt",
    "35_breaking_the_low_resource_barr_extracted.txt",
    "APSIPA2025_P208_extracted.txt"
]

base_dir = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm"

for fname in files:
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace sequences of whitespace/newlines while preserving page breaks and section paragraphs
        # First protect page markers
        pages = content.split("--- PAGE ")
        cleaned_pages = []
        for p in pages:
            if not p.strip():
                continue
            lines = p.split("\n")
            header = lines[0]
            body = "\n".join(lines[1:])
            # Replace lone newlines with spaces, but keep double newlines
            body_clean = re.sub(r'(?<!\n)\n(?!\n)', ' ', body)
            body_clean = re.sub(r' +', ' ', body_clean)
            cleaned_pages.append(f"--- PAGE {header}\n{body_clean}")
        
        clean_path = os.path.join(base_dir, fname.replace("_extracted.txt", "_clean.txt"))
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(cleaned_pages))
        print(f"Cleaned {fname} -> {clean_path}")
