import pypdf
import os
import json

def extract_pdf(pdf_path):
    print(f"=== Extracting {pdf_path} ===")
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text += f"\n--- PAGE {i+1} ---\n" + page_text
    return text

files = [
    "research/Building Dagbani Language Model and TTS.pdf",
    "research/Building Dagbani Language Models.pdf",
    "resources/2026.loreslm-1.54.pdf",
    "resources/35_breaking_the_low_resource_barr.pdf",
    "resources/APSIPA2025_P208.pdf",
    "resources/Thesis Submitted Revised 29 June PDF 2.pdf",
    "resources/Grandmasters_of_the_drum_Special_Issue_6.pdf"
]

for f in files:
    full_path = os.path.join("d:/ATS Tech/Dagbani AI", f)
    if os.path.exists(full_path):
        out_name = os.path.join("d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm", os.path.basename(f).replace(".pdf", "_extracted.txt"))
        txt = extract_pdf(full_path)
        with open(out_name, "w", encoding="utf-8") as out_f:
            out_f.write(txt)
        print(f"Wrote {len(txt)} chars to {out_name}")
    else:
        print(f"File not found: {full_path}")
