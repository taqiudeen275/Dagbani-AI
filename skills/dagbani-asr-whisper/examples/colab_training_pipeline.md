# End-to-End Google Colab Fine-Tuning Pipeline Walkthrough

**Document Version**: 1.0.0  
**Target Platform**: Google Colab Free / Pro (NVIDIA T4 / V100 / A100 GPU)  
**Notebook Reference**: `resources/dagbani_whisper_asr_colab_v2.ipynb`  

---

## 1. Complete Workflow Architecture

```
1. Environment Setup & GPU Check
   ├── PyTorch, Transformers, PEFT, BitsAndBytes, Datasets, Evaluate, SoundFile
   └── Introspect VRAM allocation (T4 16GB)
2. Data Ingestion & Metadata Parsing
   ├── Load audio from SciDB / Common Voice / GhanaNLP
   ├── Ingest transcripts, apply Dagbani UTF-8 NFC Normalization
   └── Validate duration boundaries [0.5s, 30.0s]
3. Speaker-Disjoint Partitioning
   └── Partition speakers 90% Train / 5% Validation / 5% Test
4. Model Quantization & LoRA Adapter Setup
   ├── Load Whisper Medium in 8-bit precision (BitsAndBytes)
   └── Inject LoRA rank decomposition (r=16, alpha=32)
5. Custom Data Collator & Trainer Loop
   ├── Lazy dynamic batch feature extraction (80 log-mel bins)
   ├── Seq2SeqTrainer with Cosine LR schedule and FP16 mixed precision
   └── Periodic evaluation and checkpoint saving
6. Test Evaluation & Model Card Publishing
   ├── Compute WER, CER, Normalized WER, and Glyph Precision/Recall
   └── Export LoRA adapter weights to Hugging Face Hub
```

---

## 2. Step-by-Step Code Execution

### Step 1: Environment & Dependency Installation
```python
!pip install -q --upgrade pip
!pip install -q \
    transformers==4.38.2 \
    datasets==2.18.0 \
    peft==0.9.0 \
    bitsandbytes==0.42.0 \
    accelerate==0.27.2 \
    evaluate==0.4.1 \
    jiwer==3.0.3 \
    soundfile==0.12.1 \
    librosa==0.10.1
```

### Step 2: Speaker-Disjoint Split Strategy
```python
import random
from collections import defaultdict

def create_speaker_disjoint_splits(records, train_ratio=0.90, val_ratio=0.05, seed=42):
    random.seed(seed)
    speaker_map = defaultdict(list)
    for rec in records:
        speaker_map[rec["speaker_id"]].append(rec)
        
    speakers = list(speaker_map.keys())
    random.shuffle(speakers)
    
    n_speakers = len(speakers)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)
    
    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train:n_train + n_val])
    test_speakers = set(speakers[n_train + n_val:])
    
    train_records = [r for s in train_speakers for r in speaker_map[s]]
    val_records = [r for s in val_speakers for r in speaker_map[s]]
    test_records = [r for s in test_speakers for r in speaker_map[s]]
    
    return train_records, val_records, test_records
```

### Step 3: Model Loading, 8-Bit Quantization & LoRA Injection
```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id, language=None, task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

# Prepare model for 8-bit quantized training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 4,718,592 || all params: 768,576,000 || trainable%: 0.6139%
```

### Step 4: Training Arguments & Trainer Execution
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-dagbani-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=225,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=["tensorboard"]
)

# Disable forced tokens to avoid foreign language bias
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
```
