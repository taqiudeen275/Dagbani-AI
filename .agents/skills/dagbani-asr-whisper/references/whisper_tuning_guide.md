# Dagbani Whisper Fine-Tuning Recipe & Hyperparameter Optimization Guide

**Document Version**: 1.0.0  
**Target Architectures**: OpenAI Whisper Small ($244\text{M}$), Medium ($769\text{M}$), Large-v3 ($1550\text{M}$)  
**Hardware Baselines**: NVIDIA T4 (16GB), RTX 3090/4090 (24GB), A100 (40GB/80GB)  

---

## 1. Executive Strategy for Low-Resource Fine-Tuning

Full fine-tuning of Whisper on small African language datasets ($< 50\text{ hours}$) suffers from two major pathologies:
1. **Catastrophic Overfitting & Forgetting**: Dense full fine-tuning rapidly destroys pre-trained acoustic priors and English/multilingual representations.
2. **VRAM Exhaustion**: Whisper Medium full fine-tuning requires $> 28\text{ GB}$ VRAM, exceeding standard single-GPU cloud environments.

### 1.1 Parameter-Efficient Fine-Tuning (PEFT) with LoRA
Low-Rank Adaptation freezes base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects low-rank decomposition matrices $\Delta W = \frac{\alpha}{r}(B \times A)$:
- Trainable parameter ratio: **$0.62\%$** (~4.8M parameters on Whisper Medium).
- Training memory requirement: Reduced by **$65\%$** (fits comfortably in 10.5 GB VRAM in 8-bit mode).
- Generalization: Prevents catastrophic forgetting while allowing the decoder to adapt to Dagbani syntax and BGL orthography.

---

## 2. Hyperparameter Configuration Matrix

| Hyperparameter | Pilot / Debug Run | Production Fine-Tuning (T4 16GB) | Production Cluster (A100 80GB) |
| :--- | :--- | :--- | :--- |
| **Base Model** | `openai/whisper-small` | `openai/whisper-medium` | `openai/whisper-large-v3` |
| **Quantization** | 8-bit (`bitsandbytes`) | 8-bit (`bitsandbytes`) | None (BF16) or 8-bit |
| **Precision** | `fp16` | `fp16` | `bf16` |
| **LoRA Rank ($r$)** | 16 | **16** | 32 |
| **LoRA Alpha ($\alpha$)** | 32 | **32** | 64 |
| **LoRA Dropout** | 0.05 | **0.05** | 0.05 |
| **Target Modules** | `["q_proj", "v_proj"]` | `["q_proj", "v_proj", "out_proj"]` | `["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]` |
| **Per-Device Train Batch** | 4 | **4** | 16 |
| **Gradient Accumulation** | 2 | **4** | 2 |
| **Effective Batch Size** | 8 | **16** | **32** |
| **Peak Learning Rate** | $1 \times 10^{-4}$ | **$1 \times 10^{-4}$** | $5 \times 10^{-5}$ |
| **Learning Rate Schedule** | Linear with warmup | **Cosine with warmup** | Cosine with warmup |
| **Warmup Steps** | 50 | **500** | 1000 |
| **Weight Decay** | 0.01 | **0.01** | 0.01 |
| **Gradient Checkpointing** | True | **True** | True |
| **Max Training Steps** | 1000 | **3000 - 5000** | 10000 |
| **Evaluation Frequency** | Every 100 steps | **Every 500 steps** | Every 500 steps |
| **Save Total Limit** | 2 | **3** | 5 |
| **Metric for Best Model** | `wer` (min) | **`wer` (min)** | `wer` (min) |

---

## 3. Tokenizer & Generation Configuration for Dagbani

### 3.1 Language ID Handling
Dagbani (`dag`) is not pre-registered in Whisper's multilingual language table. Forcing an unrelated language token (such as English `<|en|>` or Hausa `<|ha|>`) biases the decoder language model toward foreign phonotactics.

```python
# Optimal Dagbani Generation Config
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.task = "transcribe"
model.generation_config.language = None
model.generation_config.max_length = 225
model.generation_config.no_speech_threshold = 0.6
```

### 3.2 UTF-8 Byte-Fallback Verification
Whisper's BPE vocabulary natively supports arbitrary UTF-8 byte streams. Dagbani special characters decompose into multi-byte tokens without loss:
- `ɛ` (`U+025B`): `[0xC9, 0x9B]`
- `ɔ` (`U+0254`): `[0xC9, 0x94]`
- `ɣ` (`U+0263`): `[0xC9, 0xA3]`
- `ŋ` (`U+014B`): `[0xC5, 0x8B]`
- `ʒ` (`U+0292`): `[0xCA, 0x92]`

---

## 4. Training Stability & Memory Management

### 4.1 Lazy Dynamic Feature Extraction
Pre-computing log-mel spectrograms for $>10,000$ audio clips during dataset initialization exhausts host RAM ($>16\text{ GB}$). Implement dynamic on-the-fly extraction in `DataCollatorSpeechSeq2SeqWithPadding`:
```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract features dynamically only for current batch
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad label token IDs
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Mask padding in loss computation with -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Drop leading BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch
```

### 4.2 Gradient Checkpointing with 8-bit LoRA
When using `BitsAndBytes` 8-bit quantization with LoRA and `gradient_checkpointing=True`, PyTorch requires enabling input gradients before passing tensors to frozen layers:
```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
```
