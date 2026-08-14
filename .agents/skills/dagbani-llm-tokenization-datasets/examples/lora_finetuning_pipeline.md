# Step-by-Step LoRA / QLoRA Instruction Fine-Tuning Pipeline for Dagbani

This example demonstrates how to fine-tune a foundation LLM (e.g., LLaMA-3.1-8B-Instruct) on Dagbani question-answering and translation datasets using `llm_lora_finetuner.py`.

---

## 1. Prepare Instruction Dataset (`dagbani_instructions.jsonl`)

Format instruction pairs in JSON Lines with `system`, `instruction`, and `response`:

```json
{"system": "A nyɛla Dagbanli AI sɔŋda.", "instruction": "Wula ka n-ni tooi gu n kpaŋkpaŋ ka chɛ binnɛma?", "response": "Zaŋmi tiim din tuhi binnɛma n-fari kpaŋkpaŋ maa saha kam ka lahi gbibi pukparilim sodoligu viɛnyɛla."}
{"system": "A nyɛla Dagbanli AI sɔŋda.", "instruction": "Bɔ n-nyɛ Yaa-Naa tuma Dagbaŋ?", "response": "Yaa-Naa n-nyɛ Dagbaŋ Naa kpɛma ŋun gbibi nam, salo alaafeei, mini kaya ni ta'ada zaa."}
{"system": "A nyɛla Dagbanli AI sɔŋda.", "instruction": "Translate to English: Ti bɔrimi ni ti zaŋ alaafeei baŋsim n-wuhi salo.", "response": "We want to teach the people health knowledge."}
```

---

## 2. Launch QLoRA Fine-Tuning via CLI

Run parameter-efficient fine-tuning on a single 16GB–24GB GPU:

```bash
python ../scripts/llm_lora_finetuner.py \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-file data/dagbani_instructions.jsonl \
    --output-dir checkpoints/dagbani_llama_lora/ \
    --lora-r 64 \
    --lora-alpha 64 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --quantization 4bit
```

---

## 3. Interactive Inference with Trained LoRA Adapter

Test generated responses in Python using PEFT:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_id = "checkpoints/dagbani_llama_lora/"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)

prompt = """<|start_header_id|>system<|end_header_id|>
A nyɛla Dagbanli AI sɔŋda.<|eot_id|><|start_header_id|>user<|end_header_id|>
Wula ka bɛ kɔri kpaŋkpaŋ Tamale?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Model Response:\n", response)
```
