import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_ling/breaking_low_resource_barr.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("=== BREAKING LOW RESOURCE BARRIER FOR DAGBANI ASR (ICLR 2023) ===")
print(text[:8000])
