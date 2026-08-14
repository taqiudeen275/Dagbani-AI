# Dagbani Automatic Speech Recognition (ASR) & Whisper Fine-Tuning Playbook

**Document ID**: `DAG-KB-ASR-001`  
**Version**: `1.0.0` (Production Playbook)  
**Target Architectures**: OpenAI Whisper (Small, Medium, Large-v3), Wav2Vec 2.0 / XLS-R, MMS  
**Primary Hardware**: NVIDIA T4 (16GB), A10G (24GB), A100 (40GB/80GB), Edge ARM / Mobile  

---

## 1. Architectural Overview & Design Philosophy

Automatic Speech Recognition (ASR) for Dagbani requires overcoming three core low-resource hurdles:
1. **Acoustic Scarcity**: Available transcribed speech totals ~10 to ~80 hours across disparate recording environments.
2. **Orthographic Special Glyphs**: The 1998 Bureau of Ghana Languages (BGL) orthography contains non-ASCII characters (`ɛ`, `ɔ`, `ɣ`, `ŋ`, `ʒ`) that must be preserved without producing `<unk>` tokens.
3. **Conversational Lenition & Pitch Homophony**: Western (Tomosili) speakers frequently lenite consonants, while tone remains unwritten in text.

OpenAI Whisper's sequence-to-sequence Transformer architecture provides an optimal foundation for Dagbani by jointly modeling acoustic log-mel filterbanks and autoregressive language priors. When adapted via **8-bit Low-Rank Adaptation (LoRA)** and **lazy batch feature extraction**, Whisper Medium achieves state-of-the-art transcription accuracy on consumer GPUs without memory exhaustion.

```
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ Raw Audio File  │ ──► │  Audio Preprocessing  │ ──► │ Whisper Log-Mel Audio │ ──► │ Autoregressive BPE   │
│ (WAV/MP3/OGG)   │     │ (16kHz, VAD, Slicing) │     │ Spectrogram (80/128b) │     │ Decoder (BGL Tokens) │
└─────────────────┘     └───────────────────────┘     └───────────────────────┘     └──────────────────────┘
                                                                  │                             │
                                                      ┌───────────┴───────────┐     ┌───────────┴──────────┐
                                                      │ Transformer Encoder   │ ──► │ Transformer Decoder  │
                                                      │ (LoRA Q, V, Out Proj) │     │ (LoRA Cross-Attn)    │
                                                      └───────────────────────┘     └──────────────────────┘
```

---

## 2. Audio Preprocessing Pipeline

### 2.1 Format Standardization & Resampling
- **Sampling Rate ($f_s$)**: Uniform $16,000\text{ Hz}$ ($16\text{ kHz}$).
- **Channels**: Mono ($1\text{ channel}$). Multichannel audio is downmixed:
  $$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$$
- **Bit Depth**: 16-bit Linear PCM (`int16`) normalized to 32-bit floating point (`float32` $\in [-1.0, 1.0]$).
- **Peak Normalization**:
  $$x_{\text{norm}}[n] = \frac{x[n]}{\max(|x[n]|) + 10^{-7}}$$

### 2.2 Log-Mel Spectrogram Parameters

| Parameter | Whisper Tiny / Base / Small / Medium | Whisper Large-v3 | Unit / Formula |
| :--- | :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | $16,000\text{ Hz}$ | $16,000\text{ Hz}$ | Samples / sec |
| **Filterbank Bins ($N_{\text{mels}}$)** | **80 channels** | **128 channels** | Mel scale channels |
| **Window Length ($N_{\text{fft}}$)** | 400 samples ($25.0\text{ ms}$) | 400 samples ($25.0\text{ ms}$) | $T_{\text{win}} = N_{\text{fft}} / f_s$ |
| **Hop Length ($N_{\text{hop}}$)** | 160 samples ($10.0\text{ ms}$) | 160 samples ($10.0\text{ ms}$) | $T_{\text{hop}} = N_{\text{hop}} / f_s$ |
| **Window Function** | Periodic Hann | Periodic Hann | $w[n] = 0.5 - 0.5\cos(2\pi n / N)$ |
| **Window Context** | Fixed 30.0 seconds | Fixed 30.0 seconds | 3000 frames |
| **Input Feature Shape** | $(80, 3000)$ | $(128, 3000)$ | $(\text{Channels}, \text{Frames})$ |

### 2.3 Voice Activity Detection (VAD) & Segmentation
1. **Duration Filtering**:
   - Minimum duration: $0.5\text{ seconds}$ (filters out impulse noise, empty clicks).
   - Maximum duration: $30.0\text{ seconds}$ (matches Whisper context window).
2. **Silero VAD Architecture**:
   - Threshold $= 0.50$, minimum speech duration $= 250\text{ ms}$, minimum silence duration $= 400\text{ ms}$, speech padding $= 100\text{ ms}$.
   - Continuous audio $>30.0\text{s}$ is sliced at natural VAD silence boundaries with a $200\text{ ms}$ cross-fade.

### 2.4 Data Augmentation & SpecAugment
To prevent catastrophic overfitting on $<50$ hours of training speech:
- **Frequency Masking**: $F = 27$ maximum Mel frequency channels masked ($m_F = 2$).
- **Time Masking**: $T = 100$ frames (1.0 second) masked ($p = 0.05$ max ratio).
- **Speed Perturbation**: Perturbations at $\{0.9, 1.0, 1.1\}\times$ speed without modifying text labels.
- **Additive Noise**: SNR between $12\text{ dB}$ and $28\text{ dB}$ using ambient West African market/outdoor background audio.

---

## 3. Whisper Model Configurations & Parameter Matrix

| Model Checkpoint | Total Params | Encoder Layers | Decoder Layers | Hidden Dim ($d_{\text{model}}$) | Attention Heads | Mel Bins | Inference VRAM | 8-Bit LoRA VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `whisper-small` | $244\text{ M}$ | 12 | 12 | 768 | 12 | 80 | ~3.0 GB | ~6.5 GB |
| `whisper-medium` | **$769\text{ M}$** | **24** | **24** | **1024** | **16** | **80** | **~6.5 GB** | **~10.5 GB (T4 Optimal)** |
| `whisper-large-v3`| $1550\text{ M}$| 32 | 32 | 1280 | 20 | 128 | ~12.0 GB | ~15.5 GB (A100/L4) |

---

## 4. Parameter-Efficient Fine-Tuning (PEFT): 8-Bit LoRA Recipe

Full parameter fine-tuning of Whisper Medium requires $>28\text{ GB}$ VRAM and destabilizes pre-trained multilingual acoustic embeddings. Low-Rank Adaptation (LoRA) injects trainable rank decomposition matrices into frozen base weights:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(B \times A)$$

### 4.1 Production LoRA Hyperparameter Specification

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                                              # Rank parameter
    lora_alpha=32,                                     # Scaling factor (alpha / r = 2.0)
    target_modules=["q_proj", "v_proj", "out_proj"],   # Target attention projections
    lora_dropout=0.05,                                 # Regularization dropout
    bias="none",                                       # Freeze bias vectors
)
```

- **Trainable Weights**: ~4.8M parameters (~$0.62\%$ of Whisper Medium).
- **Extended Target Modules (for deep adaptation)**: `["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]`.

### 4.2 BitsAndBytes 8-Bit Quantization Pipeline

```python
import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium",
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 5. Tokenization, Special Glyphs & Unforced Decoding

### 5.1 Byte-Fallback Encoding for BGL Glyphs
Whisper's BPE tokenizer operates directly on raw UTF-8 bytes. The 5 non-ASCII BGL characters decompose into multi-byte sequences without producing `<unk>` tokens:

| Glyph | Unicode Code Point | UTF-8 Hex Bytes | Tokenizer Round-Trip Status |
| :---: | :---: | :---: | :---: |
| **`ɛ`** | `U+025B` | `0xC9 0x9B` | Exact match (2 byte tokens) |
| **`ɔ`** | `U+0254` | `0xC9 0x94` | Exact match (2 byte tokens) |
| **`ɣ`** | `U+0263` | `0xC9 0xA3` | Exact match (2 byte tokens) |
| **`ŋ`** | `U+014B` | `0xC5 0x8B` | Exact match (2 byte tokens) |
| **`ʒ`** | `U+0292` | `0xCA 0x92` | Exact match (2 byte tokens) |

### 5.2 Unforced Language Decoding Configuration
Dagbani (`dag`) is not pre-registered in Whisper's 99 language tokens. Forcing a foreign language token (e.g. `<|en|>`, `<|ha|>`) biases acoustic cross-attention toward foreign phonotactics.

```python
# Configure generation config for unforced Dagbani transcription
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.config.forced_decoder_ids = None
```

---

## 6. Custom Lazy Feature Extraction Data Collator

Pre-computing log-mel spectrograms across 15,000+ audio clips exhausts host RAM ($>16\text{ GB}$). The production pipeline implements dynamic, lazy batch extraction:

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Lazy dynamic feature extraction on raw 16kHz audio waveforms
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]

        batch_inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        input_features = batch_inputs.input_features

        # 2. Dynamic token label padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3. Mask padding positions with -100 to ignore in Cross-Entropy loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 4. Remove leading BOS token if duplicated
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }
```

---

## 7. Production Training Hyperparameters & Training Loop

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | `openai/whisper-medium` | 769M Transformer checkpoint |
| **Per-Device Train Batch Size** | `4` | Mini-batch per GPU |
| **Gradient Accumulation Steps** | `4` | Effective Batch Size $= 16$ |
| **Peak Learning Rate** | `1e-4` | Optimal convergence rate for 8-bit LoRA |
| **LR Scheduler** | `cosine` | Cosine annealing with warmup |
| **Warmup Steps** | `500` | Smooth initial gradient trajectory |
| **Max Training Steps** | `5000` (~10 epochs on 20h corpus) | Total optimizer steps |
| **Weight Decay** | `0.01` | AdamW L2 regularization |
| **Precision** | `fp16` (mixed precision) | Accelerates tensor operations |
| **Gradient Checkpointing** | `True` | Trades compute for ~40% VRAM reduction |
| **Evaluation Strategy** | Every 500 steps | Computes validation WER/CER |
| **Generation Max Length** | `225 tokens` | Caps autoregressive output length |
| **Metric for Best Model** | `wer` (lower is better) | Automates best checkpoint checkpointing |

---

## 8. Evaluation Framework, Metrics & Error Taxonomy

### 8.1 Word Error Rate (WER) & Character Error Rate (CER)
$$\text{WER} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Reference Words}} \times 100\%$$
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$

### 8.2 Strict vs. Normalized Evaluation
1. **Strict WER**: Raw string equality (penalizes punctuation, capitalization, whitespace differences).
2. **Normalized WER**: Standardizes text via NFC normalization, lowercasing, and removal of non-spoken punctuation:

```python
import re
import unicodedata

def normalize_dagbani_eval(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"').lower()
    cleaned = [ch for ch in text if unicodedata.category(ch).startswith("L") or ch in {"-", "'", " "}]
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()
```

### 8.3 Special Glyph Recall Metric
Evaluates specific recall for Dagbani special letters (`ɛ`, `ɔ`, `ɣ`, `ŋ`, `ʒ`):
$$\text{Recall}_c = \frac{\sum_{i} \min(N_{\text{ref}}^{(i)}(c), N_{\text{pred}}^{(i)}(c))}{\sum_{i} N_{\text{ref}}^{(i)}(c)}$$

### 8.4 Common Error Taxonomy & Engineering Mitigations

| Failure Mode | Root Cause | Observable Symptom | Engineering Mitigation |
| :--- | :--- | :--- | :--- |
| **Silent Chunk Hallucination** | Autoregressive LM prior runs unconditioned in background noise | Model repeats *"N nyɛla... N nyɛla..."* on silence | 1. Silero VAD pre-filtering.<br>2. `no_speech_threshold=0.6`.<br>3. `condition_on_previous_text=False`. |
| **Dialectal Lenition Miss** | Western speakers lenite $/bg/ \rightarrow [\upsilon]$, $/gs/ \rightarrow [x]$ | Word substitution errors on *kɔbga* vs *kɔwa* | Include balanced training data from Tamale and Yendi; apply SpecAugment. |
| **Special Glyph Dropping** | Model predicts standard ASCII Latin letters (`e, o, g`) | `paɣa` transcribed as `paga` | Target LoRA adapters to MLP blocks (`fc1`, `fc2`) to reinforce multi-byte transitions. |
| **English Code-Switching** | Mixed English technical terms in daily speech | Phonetic mangling of *school*, *hospital* | Include code-switched training examples; leverage Whisper multilingual priors. |

---

## 9. Production Inference & Edge Deployment

### 9.1 CTranslate2 & Faster-Whisper Optimization
CTranslate2 provides an optimized C++ inference runtime utilizing cuBLAS and INT8 quantization:

```python
from faster_whisper import WhisperModel

# Load INT8-quantized model on GPU
model = WhisperModel(
    "ats-tech/dagbani-whisper-medium-ct2",
    device="cuda",
    compute_type="int8_float16"
)

segments, info = model.transcribe(
    "audio_sample.wav",
    beam_size=5,
    language=None,
    task="transcribe",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=400)
)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

- **Speedup**: $4.2\times$ faster than standard PyTorch Hugging Face pipeline.
- **Memory Footprint**: $1.9\text{ GB}$ VRAM for Whisper Medium.

### 9.2 Whisper.cpp for Mobile & Offline Android Devices
- Standalone C/C++ engine compiled for ARM NEON and Apple Silicon.
- Model quantized to `Q4_0` or `Q5_0` GGUF format ($<500\text{ MB}$ file size for Whisper Small).
- Runs in real-time ($<200\text{ ms}$ latency per 3-second audio chunk) on standard MediaTek / Qualcomm mobile chipsets without internet connectivity.

---

## 10. References & Empirical Authorities
1. Ibrahim, A., et al. (2023). *Breaking the Low-Resource Barrier for Dagbani ASR: From Data Collection to ASR Modeling*. AfricaNLP @ ICLR 2023.
2. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI.
3. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
4. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
5. Silero Team. (2024). *Silero VAD: Pre-trained Enterprise-Grade Voice Activity Detector*.
