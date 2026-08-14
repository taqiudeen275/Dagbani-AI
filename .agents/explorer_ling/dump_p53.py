import sys
import pypdf

sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader("resources/Thesis Submitted Revised 29 June PDF 2.pdf")
print("=== PAGE 72/73 (THESIS BODY PAGE 53/54) ===")
print(reader.pages[52].extract_text())
print("\n" + "="*50 + "\n")
print(reader.pages[53].extract_text())
