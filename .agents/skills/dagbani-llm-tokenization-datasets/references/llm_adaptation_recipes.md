# LLM Adaptation Recipes for Dagbani: Continual Pre-Training & LoRA/QLoRA

This reference document outlines the hyperparameter configurations, dataset curriculum, and parameter-efficient fine-tuning (PEFT) protocols for adapting Large Language Models to Dagbani.

---

## 1. Continual Pre-Training (CPT) vs Instruction Fine-Tuning

| Training Stage | Primary Objective | Dataset Composition | Compute Footprint | Target Loss / Metric |
|---|---|---|---|---|
| **Continual Pre-Training (CPT)** | Teach the LLM native Dagbani grammar, vocabulary, world knowledge, and morphology. | 50M–200M tokens (Wikipedia, Bible, Transcriptions, Panegyrics) | 4x–8x A100 GPUs (100–300 GPU hours) | Perplexity $< 8.5$ on held-out Dagbani text |
| **Instruction Tuning (SFT)** | Adapt the model to follow instructions, answer questions, translate, and adhere to cultural norms. | 25k–100k instruction-response pairs | Single 16GB–24GB GPU (RTX 4090 / T4 / A10) | CHRF++ $> 52.0$, Dagbani-MMLU $> 75\%$ |

---

## 2. Bilingual Curriculum Strategy

To prevent **catastrophic forgetting** of base reasoning, math, and coding capabilities during continual pre-training:
- **Ratio**: Interleave **70% Dagbani text** with **30% high-quality English educational text** (SlimPajama / Cosmopedia / OpenWebMath).
- **Sequence Length**: Pack texts into continuous 4,096-token chunks separated by EOS tokens with cross-document attention masking.

---

## 3. Production QLoRA Fine-Tuning Specification

For instruction tuning Meta-Llama-3.1-8B-Instruct or Cohere Aya-23-8B on consumer or budget cloud GPUs:

### Hyperparameter Configuration:

```python
LORA_CONFIG = {
    "r": 64,                           # High rank to capture nuanced morphology
    "lora_alpha": 64,                  # Scaling factor alpha/r = 1.0
    "lora_dropout": 0.05,              # Prevents overfitting on small datasets
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",      # NormalFloat 4-bit
    "bnb_4bit_use_double_quant": True, # Secondary quantization for memory reduction
    "bnb_4bit_compute_dtype": "bfloat16"
}

TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
    "warmup_steps": 100,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "optimizer": "paged_adamw_8bit",
    "num_train_epochs": 3,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 250,
    "save_total_limit": 3
}
```

---

## 4. Culturally Grounded Chat Template Specification

Standard conversational template for Dagbani instruction-tuned models:

```json
{
  "system_prompt": "A nyɛla Dagbanli AI sɔŋda ŋun mali yiko, zaɣa mini baŋsim zaŋ kpa Dagbaŋ kaya ni ta'ada, yɛltɔɣa taɣimalisi, pukparilim, mini alaafeei polo. Saɣisibu kam zaŋmi Dagbanli din lu n-doli BGL sodoligu n-ti salo.",
  "conversation_format": "<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{model_response}<|eot_id|>"
}
```

---

## 5. Downstream Evaluation Protocols

1. **Machine Translation**:
   - English $\to$ Dagbani & Dagbani $\to$ English on GhanaNLP / FLORES-200.
   - Evaluated using **CHRF++** (character n-grams) and **spBLEU**.
2. **Dagbani Culturally-Grounded MMLU**:
   - 500 multi-choice questions covering Yaa-Naa chieftaincy history, traditional festivals (Damba, Bugum), proverbs (*Yɛltɔɣa taɣimalisi*), and local agronomy (shea nut, yam, maize).
