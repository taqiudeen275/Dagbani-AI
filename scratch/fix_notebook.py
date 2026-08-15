import json

notebook_path = 'notebooks/Dagbani_AI_Phase1_Phase2_Kaggle_T4x2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split at "  \"language_info\": {"
split_point = '  "language_info": {'
if split_point in content:
    cells_part = content.split(split_point)[0]
    tail = '''  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
'''
    new_content = cells_part + tail
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

# Validate
data = json.load(open(notebook_path, 'r', encoding='utf-8'))
print(f"SUCCESS: Notebook has {len(data['cells'])} cells and valid JSON!")
