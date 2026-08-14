import pypdf
import os

def extract_pdf(pdf_path, out_txt, start_page=0, end_page=None):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    end = total if end_page is None else min(end_page, total)
    print(f"Extracting {pdf_path} (pages {start_page+1} to {end})...")
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in range(start_page, end):
            text = reader.pages[p].extract_text()
            f.write(f"\n--- [PAGE {p+1}] ---\n")
            f.write(text if text else "")
    print(f"Saved {out_txt} ({os.path.getsize(out_txt)} bytes)")

def main():
    # 1. Building Dagbani Language Models.pdf
    extract_pdf("research/Building Dagbani Language Models.pdf", ".agents/explorer_ling/building_dagbani_lm.txt")

    # 2. Building Dagbani Language Model and TTS.pdf
    extract_pdf("research/Building Dagbani Language Model and TTS.pdf", ".agents/explorer_ling/building_dagbani_lm_tts.txt")

    # 3. Breaking the low-resource barrier
    extract_pdf("resources/35_breaking_the_low_resource_barr.pdf", ".agents/explorer_ling/breaking_low_resource_barr.txt")

    # 4. Thesis - locate actual Chapter 2 (let's scan for Chapter 2 heading beyond page 25)
    thesis_path = "resources/Thesis Submitted Revised 29 June PDF 2.pdf"
    reader = pypdf.PdfReader(thesis_path)
    ch1_p = None
    ch2_p = None
    ch3_p = None
    for p in range(15, len(reader.pages)):
        txt = reader.pages[p].extract_text()
        if "CHAPTER TWO" in txt and ("Phonological" in txt or "PHONOLOGICAL" in txt):
            ch2_p = p
            print(f"Thesis Chapter 2 body found at page {p+1}")
        if "CHAPTER THREE" in txt and ch2_p is not None and ch3_p is None:
            ch3_p = p
            print(f"Thesis Chapter 3 body found at page {p+1}")
            break

    if ch2_p:
        extract_pdf(thesis_path, ".agents/explorer_ling/thesis_ch2_phonology.txt", start_page=ch2_p, end_page=(ch3_p if ch3_p else ch2_p + 60))
        # Also extract Chapter 1
        extract_pdf(thesis_path, ".agents/explorer_ling/thesis_ch1_intro.txt", start_page=18, end_page=ch2_p)

    # 5. Grandmasters of the drum - extract introductory chapters / linguistic analysis
    extract_pdf("resources/Grandmasters_of_the_drum_Special_Issue_6.pdf", ".agents/explorer_ling/grandmasters_snippet.txt", start_page=0, end_page=60)

if __name__ == "__main__":
    main()
