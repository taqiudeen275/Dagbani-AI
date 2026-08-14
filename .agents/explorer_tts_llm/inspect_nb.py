import json

nb_path = "d:/ATS Tech/Dagbani AI/resources/dagbani_whisper_asr_colab_v2.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

out_path = "d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/notebook_dump.txt"
with open(out_path, "w", encoding="utf-8") as out:
    out.write(f"Total cells: {len(nb.get('cells', []))}\n")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        out.write(f"\n==================== CELL {i+1} ({cell_type}) ====================\n")
        out.write(source + "\n")

print(f"Successfully dumped notebook to {out_path}")
