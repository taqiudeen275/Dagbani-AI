import pypdf
import os
import sys

def main():
    thesis_path = "resources/Thesis Submitted Revised 29 June PDF 2.pdf"
    if not os.path.exists(thesis_path):
        print(f"File not found: {thesis_path}")
        return

    reader = pypdf.PdfReader(thesis_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")

    start_p2 = None
    start_p3 = None

    for idx in range(total_pages):
        text = reader.pages[idx].extract_text()
        if "CHAPTER TWO" in text and ("Phonological" in text or "PHONOLOGICAL" in text):
            start_p2 = idx
            print(f"Chapter 2 found at PDF page {idx+1}")
        if "CHAPTER THREE" in text and start_p2 is not None and start_p3 is None:
            start_p3 = idx
            print(f"Chapter 3 found at PDF page {idx+1}")
            break

    if start_p2 is not None:
        end_idx = start_p3 if start_p3 is not None else start_p2 + 70
        output_txt = f".agents/explorer_ling/thesis_chapter2.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            for p in range(start_p2, end_idx):
                f.write(f"=== PAGE {p+1} ===\n")
                f.write(reader.pages[p].extract_text() + "\n\n")
        print(f"Saved Chapter 2 ({end_idx - start_p2} pages) to {output_txt}")

if __name__ == "__main__":
    main()
