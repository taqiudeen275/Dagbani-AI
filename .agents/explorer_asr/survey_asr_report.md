# Dagbani Automatic Speech Recognition (ASR) & Whisper Architecture Specification Report

**Document ID**: `DAGBANI-ASR-SPEC-2026-V1`  
**Author**: Domain Expert & Specification Miner (ASR & Whisper Architecture)  
**Target Milestone**: M0 (Survey & Forensic Analysis) & M1 (Knowledge Base Synthesis)  
**Working Directory**: `.agents/explorer_asr/`  
**Status**: Comprehensive & Authoritative  

---

## 1. Executive Summary & Forensic Exploration Scope

This document presents an exhaustive, forensic exploration and technical specification extraction of Automatic Speech Recognition (ASR) architectures, OpenAI Whisper fine-tuning pipelines, audio preprocessing protocols, and speech datasets tailored for the **Dagbani** language (*Dagbanli* / *Dagbane*), a Gur language spoken by over 3 million people in Northern Ghana and Togo.

The analysis is synthesized directly from authoritative primary project resources:
1. `resources/dagbani_whisper_asr_colab_v2.ipynb` (Production Colab Fine-Tuning Notebook v2).
2. `resources/35_breaking_the_low_resource_barr.pdf` (Ibrahim et al., AfricaNLP @ ICLR 2023: *Breaking the Low-Resource Barrier for Dagbani ASR: From Data Collection to ASR Modeling*).
3. `resources/APSIPA2025_P208.pdf` (Zhou, Ito, & Nose, APSIPA ASC 2025: *Improving Speech-to-Speech Translation for Low-Resource Languages via Transfer Learning*).
4. `research/Building Dagbani Language Model and TTS.pdf` (Architecting Speech-Centric Large Language Models and Generative Voice Systems for Low-Resource African Languages).
5. `resources/Thesis Submitted Revised 29 June PDF 2.pdf` (Sheini 2021, University of Bayreuth: *Dagbani English & Dagbani Phonology/Grammar*).
6. `resources/dagbani_llm_tts_strategy.html` (Interactive Architectural Strategy Blueprint).
7. `research/research_source.txt` (External Linked Research Repositories).

---

## 2. Audio Specifications & Preprocessing Pipeline

ASR performance in low-resource environments depends strictly on precise acoustic feature extraction, standardized raw audio ingestion, and robust noise handling.

### 2.1 Audio Ingestion & Format Standards
To interface cleanly with neural acoustic models (Whisper, Wav2Vec 2.0, XLS-R, MMS):
- **Native Sampling Rate**: $16,000\text{ Hz}$ ($16\text{ kHz}$). All input audio formats (WAV, MP3, OGG, FLAC) must be resampled to $16\text{ kHz}$ mono upon loading.
- **Channel Configuration**: Single-channel ($1\text{ Channel}$ / Mono). Multichannel signals are downmixed via arithmetic averaging across channels:
  $$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^C x_c[n]$$
- **Bit Depth & Encoding**: Signed 16-bit Linear PCM (`int16`, little-endian) or normalized 32-bit floating point (`float32` in the range $[-1.0, 1.0]$).
- **Target Bitrate**: Minimum 64 kbps (CBR/VBR) for MP3/OGG inputs; uncompressed PCM WAV for raw ingestion.
- **Dynamic Range & Normalization**: Peak normalization to $-1.0\text{ dBFS}$ or RMS energy normalization:
  $$x_{\text{norm}}[n] = \frac{x[n]}{\max(|x[n]|) + \epsilon}$$

### 2.2 Log-Mel Spectrogram Extraction Parameters
Whisper maps raw waveform vectors directly into log-magnitude Mel-scale spectrogram representations.

| Parameter | Whisper (Tiny / Base / Small / Medium / Large-v1 / Large-v2) | Whisper Large-v3 | Unit / Formula |
| :--- | :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | $16,000\text{ Hz}$ | $16,000\text{ Hz}$ | Samples per second |
| **Mel Filterbank Channels ($N_{\text{mels}}$)** | **80 channels** | **128 channels** | Filterbank count |
| **STFT Window Size ($N_{\text{fft}}$)** | 400 samples ($25.0\text{ ms}$) | 400 samples ($25.0\text{ ms}$) | $T_{\text{win}} = \frac{N_{\text{fft}}}{f_s}$ |
| **Hop Length ($N_{\text{hop}}$)** | 160 samples ($10.0\text{ ms}$) | 160 samples ($10.0\text{ ms}$) | $T_{\text{hop}} = \frac{N_{\text{hop}}}{f_s}$ |
| **Window Function** | Periodic Hann Window | Periodic Hann Window | $w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N}\right)$ |
| **Frequency Range ($f_{\min} - f_{\max}$)** | $0\text{ Hz} - 8000\text{ Hz}$ | $0\text{ Hz} - 8000\text{ Hz}$ | Nyquist limit at $16\text{ kHz}$ |
| **Spectrogram Frames per 30s** | **3000 frames** | **3000 frames** | $\frac{30.0 \times 16000}{160} = 3000$ |
| **Spectrogram Tensor Shape** | $(80, 3000)$ | $(128, 3000)$ | $(\text{Channels}, \text{Time Frames})$ |
| **Log Compression** | $\log_{10}(\max(M, 10^{-5}))$ scaled to $[-1, 1]$ | $\log_{10}(\max(M, 10^{-5}))$ scaled to $[-1, 1]$ | Dynamic range bounding |

### 2.3 Noise Handling, Voice Activity Detection (VAD) & Segmentation
1. **Duration Filtering Bounds**:
   - $\text{Duration}_{\min} = 0.5\text{ seconds}$ (filters out click artifacts and empty audio files).
   - $\text{Duration}_{\max} = 30.0\text{ seconds}$ (matches Whisper's fixed 30-second context window).
   - Any clip $> 30.0\text{s}$ must be segmented prior to feature extraction to prevent silent tail truncation.
2. **Voice Activity Detection (VAD)**:
   - **Silero VAD v4/v5**: High-precision, low-latency deep neural network for speech boundary detection.
   - **Chunking Thresholds**: Speech threshold $= 0.5$, minimum speech duration $= 250\text{ ms}$, minimum silence duration $= 400\text{ ms}$, speech pad $= 100\text{ ms}$.
   - **Long-Form Audio Strategy**: Split continuous audio into utterances $\le 28\text{s}$ at natural silent pauses (VAD-detected boundaries) with a $200\text{ ms}$ cross-fade overlap.
3. **Environmental Noise Mitigation**:
   - Low-pass filter cut-off at $7.6\text{ kHz}$; high-pass DC blocker at $50\text{ Hz}$.
   - Spectral gating / stationary background noise suppression for field-recorded mobile samples.

### 2.4 Data Augmentation & SpecAugment for Low-Resource Speech
Given the low-resource constraint of Dagbani speech (~10 to ~50 hours available), acoustic data augmentation prevents catastrophic overfitting:
1. **SpecAugment Parameters**:
   - **Frequency Masking**: $F = 27$ (mask up to 27 consecutive Mel frequency channels, with $m_F = 2$ masks).
   - **Time Masking**: $T = 100$ (mask up to 100 consecutive time frames / 1 second, with $p = 0.05$ max ratio).
   - **Time Warping**: Warping parameter $W = 5$ frames.
2. **Waveform-Level Augmentation**:
   - **Speed Perturbation**: Perturbation factors $\{0.9, 1.0, 1.1\}$ (alters pitch and tempo without changing semantic label).
   - **Additive Noise**: Gaussian noise with SNR between $10\text{ dB}$ and $30\text{ dB}$, and ambient West African outdoor background noise simulation.
   - **Gain Perturbation**: Random gain adjustment between $-6\text{ dB}$ and $+6\text{ dB}$.

---

## 3. Dagbani Linguistic & Orthographic Foundations for ASR

Understanding Dagbani phonology, dialectal variations, and writing systems is crucial for designing tokenizers, acoustic models, and metric evaluators.

### 3.1 1998 Bureau of Ghana Languages (BGL) Orthography
The modern Dagbani writing system uses the Latin script augmented with 5 special International African Institute / African Reference Alphabet characters:

$$\mathcal{A}_{\text{Dagbani}} = \{a, b, ch, d, e, \varepsilon, f, g, gb, \gamma, h, i, j, k, kp, l, m, n, ny, \eta, \eta m, o, \jmath, p, r, s, sh, t, u, v, w, y, z, \text{\textyogh} / \text{\textroundcap}\dots\}$$

#### Specific Special Glyphs & Unicode Code Points:
| Grapheme (Lower) | Grapheme (Upper) | Unicode (Lower) | Unicode (Upper) | IPA Value | Phonetic Description | Dagbani Example | English Meaning |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **`ɛ`** | **`Ɛ`** | `U+025B` | `U+0190` | `[ɛ]` | Open-mid front unrounded vowel | *nyɛla* | is / to be |
| **`ɔ`** | **`Ɔ`** | `U+0254` | `U+0186` | `[ɔ]` | Open-mid back rounded vowel | *dɔɣim* | family / birth |
| **`ɣ`** | **`Ɣ`** | `U+0263` | `U+0194` | `[ɣ]` / `[x]` | Voiced velar fricative | *paɣa* | woman / wife |
| **`ŋ`** | **`Ŋ`** | `U+014B` | `U+014A` | `[ŋ]` | Voiced velar nasal | *ŋun* | who / whoever |
| **`ʒ`** | **`Ʒ`** | `U+0292` | `U+01B7` | `[ʒ]` | Voiced postalveolar fricative | *ʒɛm* | despise / disrespect |

#### Digraphs & Double Articulations:
- **`ch`** (`[tʃ]`): Voiceless postalveolar affricate (e.g. *chama* - go).
- **`gb`** (`[ɡ͡b]` / `[db]`): Voiced labial-velar plosive (e.g. *gballi* - grave / zana mat).
- **`kp`** (`[k͡p]` / `[tp]`): Voiceless labial-velar plosive (e.g. *kpariba* - farmers).
- **`ŋm`** (`[ŋ͡m]` / `[nm]`): Labial-velar nasal (e.g. *ŋmariga* - star).
- **`ny`** (`[ɲ]`): Palatal nasal (e.g. *nyɛ* - see / be).
- **`sh`** (`[ʃ]`): Voiceless postalveolar fricative (e.g. *shikuru* - school).

### 3.2 Phonological Processes & Acoustic Variations
1. **Advanced Tongue Root ([ATR]) Vowel Harmony**:
   - **[+ATR] Set**: $/i, e, o, u/$
   - **[-ATR] Set**: $/ɛ, ɔ, ɨ \text{ (or } ɪ/\ə), a/$
   - Harmony operates primarily root-to-suffix ($/i/$ triggers $[+ATR]$ in suffix: *dirigu* `[dirigu]` vs $[-ATR]$ *gɔrigu* `[gɔrgʊ]`).
   - Suffix-to-root harmony occurs with mid vowels $/e, o/$ (*dɔr-tɨ* vs *dor-o*).
2. **Consonant Alternations & Allophony**:
   - $/d/$ and $/r/$: $[r]$ is an intervocalic allophone of $/d/$ (*daan-doli* vs *dori*).
   - Velar lenition: $/g/ \to [ɣ] \to [ʔ]$ intervocalically; $/g/ \to [x]$ before voiceless consonants.
   - Labial-velar coronalization: $/kp, gb, ŋm/ \to [tp, db, nm]$ before front vowels.
3. **Tone System & Pitch Mechanics**:
   - Dagbani possesses 2 level tones: **High (H)** and **Low (L)**, with systematic **downstep (!H)**.
   - Orthography **omits all tone marks**. Thus, the acoustic model must infer semantic meaning from lexical context for tone-differentiated homographs:
     - *gballi* `[ɡbálːɪ́]` (High-High) = "grave"
     - *gballi* `[ɡbálːɪ̀]` (High-Low) = "zana mat"

### 3.3 Dialectal Variations (Western / Tomosili vs Eastern / Katindu / Nayahili)
| Feature / Word | Western (Tomosili - Tamale, Savelugu) | Eastern (Katindu / Nayahili - Yendi) | Linguistic Shift |
| :--- | :--- | :--- | :--- |
| **Speech / Word** | *yɛltɔɣa* | *yɛltoɣa* | Vowel quality shift ($ɔ \leftrightarrow o$) |
| **Hundred** | *kɔwa* (`[kòʋá]`) | *kɔbga* (`[kɔ̀bgá]`) | $/bg/ \to [ʋ]$ coalescence |
| **Talk / Speak** | *tɔxɨ* (`[tɔ́xɨ́]`) | *tɔɣsɨ* (`[tɔ́ɣsɨ́]`) | $/gs/ \to [x]$ coalescence |
| **Child** | *bia* | *bia* | Identical root |
| **Acoustic Impact** | High lenition, frequent vowel coalescence | Strict velar plosive retention | ASR must handle multi-dialect pronunciations |

---

## 4. Whisper Model Architecture & Fine-Tuning Recipes

### 4.1 Whisper Architecture Zoo & Parameter Matrix
OpenAI Whisper is an encoder-decoder Transformer trained on 680,000 hours of weakly supervised multilingual and multitask audio data.

| Checkpoint | Total Params | Encoder Layers | Decoder Layers | Attention Heads | Embedding Dim ($d_{\text{model}}$) | Mel Filterbank Bins | Target VRAM (Inference) | Target VRAM (8-bit LoRA Train) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper Tiny** | $39\text{ M}$ | 4 | 4 | 6 | 384 | 80 | ~1.0 GB | ~2.5 GB |
| **Whisper Base** | $74\text{ M}$ | 6 | 6 | 8 | 512 | 80 | ~1.5 GB | ~3.8 GB |
| **Whisper Small** | $244\text{ M}$ | 12 | 12 | 12 | 768 | 80 | ~3.0 GB | ~6.5 GB |
| **Whisper Medium** | **$769\text{ M}$** | **24** | **24** | **16** | **1024** | **80** | **~6.5 GB** | **~10.5 GB (T4 Recommended)** |
| **Whisper Large-v3** | $1550\text{ M}$ | 32 | 32 | 20 | 1280 | 128 | ~12.0 GB | ~15.5 GB (A100/L4 Recommended) |

### 4.2 Architectural Baselines: Wav2Vec 2.0 vs MMS vs Whisper
- **Wav2Vec 2.0 (XLSR-53 / XLS-R-300M)**:
  - Acoustic encoder with CTC (Connectionist Temporal Classification) linear head.
  - Demonstrated by Ibrahim et al. (2023) on Dagbani: achieves **$34.7\%$ WER** (XLSR-53) and **$37.7\%$ WER** (XLS-R-300M) after 9 epochs on 9.5 hours of audio.
  - Limitation: Frame-level conditional independence assumption; cannot generate autoregressive language modeling context or punctuation naturally.
- **Meta MMS (Massively Multilingual Speech - 300M / 1B)**:
  - Supports CTC-based transcription across 1,400+ languages.
- **OpenAI Whisper (Medium / Small)**:
  - Autoregressive Encoder-Decoder architecture.
  - Jointly models acoustic features and autoregressive target language sequence.
  - Superior transcription fluency, natural punctuation handling, and robustness to conversational acoustic noise.

### 4.3 Low-Resource Adaptation Strategy: LoRA & QLoRA
Full fine-tuning of Whisper Medium ($769\text{M}$ parameters) requires $> 28\text{ GB}$ of VRAM (exceeding standard NVIDIA T4 16GB limits) and risks severe catastrophic forgetting on small Dagbani datasets ($< 50\text{ hours}$).

#### Parameter-Efficient Fine-Tuning (PEFT) with LoRA:
Low-Rank Adaptation freezes base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$
where $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$ and $B \in \mathbb{R}^{d \times r}$ is initialized to zero.

#### Production LoRA Hyperparameter Configuration:
```python
lora_config = LoraConfig(
    r=16,                                              # Rank parameter (16 yields optimal capacity vs memory)
    lora_alpha=32,                                     # Scaling alpha (typically 2 * r)
    target_modules=["q_proj", "v_proj", "out_proj"],   # Critical query, value, and output projections
    lora_dropout=0.05,                                 # Regularization dropout
    bias="none",                                       # Do not train biases
)
```
- **Trainable Parameters**: ~4.8M parameters (~$0.62\%$ of Whisper Medium's 769M weights).
- **Expanded Target Modules for Deep Adaptation**: `["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]`.

#### Quantization Options:
- **8-Bit LLM.int8() (BitsAndBytes)**:
  - `BitsAndBytesConfig(load_in_8bit=True)`
  - Outliers in activation matrices are retained in full FP16 precision, while non-outlier matrix products are quantized to 8-bit integers.
  - Perfectly stable on NVIDIA T4 (16GB VRAM) with memory footprint $\approx 4.2\text{ GB}$ for model weights.
- **4-Bit QLoRA (NF4)**:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`
  - Memory footprint $\approx 2.4\text{ GB}$. Ideal for consumer GPUs (8GB VRAM RTX 2070/3060/4060).

### 4.4 Tokenizer & Language ID Strategy for Dagbani
Whisper uses a byte-level Byte-Pair Encoding (BPE) vocabulary ($50,257$ core tokens $+ 1,608$ special tokens $= 51,865$).

#### Handling Non-ASCII Dagbani Glyphs (`ɛ`, `ɔ`, `ɣ`, `ŋ`, `ʒ`):
1. **Native Byte-Fallback Mechanism**:
   - Because Whisper's tokenizer operates at the raw UTF-8 byte level, all Dagbani special characters decompose into multi-byte sequences without producing `<unk>` (unknown token) symbols:
     - `ɛ` (`U+025B`) $\to$ bytes `0xC9 0x9B` (2 tokens)
     - `ɔ` (`U+0254`) $\to$ bytes `0xC9 0x94` (2 tokens)
     - `ɣ` (`U+0263`) $\to$ bytes `0xC9 0xA3` (2 tokens)
     - `ŋ` (`U+014B`) $\to$ bytes `0xC5 0x8B` (2 tokens)
     - `ʒ` (`U+0292`) $\to$ bytes `0xCA 0x92` (2 tokens)
2. **Tokenizer Round-Trip Validation**:
   - Input: `"n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"`
   - Encoded IDs $\to$ Decoded text $\equiv$ Exact character match.
3. **Language Token Configuration**:
   - Dagbani (`dag`) is **not** one of the 99 languages natively pre-registered in Whisper (`<|en|>`, `<|fr|>`, `<|ha|>`, `<|yo|>`, `<|sw|>`).
   - **Recommended Strategy (No Forced Language)**:
     ```python
     model.generation_config.forced_decoder_ids = None
     model.generation_config.suppress_tokens = []
     model.generation_config.task = "transcribe"
     ```
     By omitting a forced language token, the model decoder learns Dagbani character sequences directly conditioned on cross-attention audio features without being biased by foreign language token priors.

### 4.5 Training Hyperparameters Optimization Matrix
| Hyperparameter | Pilot / Debug Run | Production Fine-Tuning (T4 16GB) | Production Cluster (A100 80GB) |
| :--- | :--- | :--- | :--- |
| **Base Model** | `openai/whisper-small` | `openai/whisper-medium` | `openai/whisper-large-v3` |
| **Precision** | `fp16` | `fp16` | `bf16` |
| **Quantization** | 8-bit (`bitsandbytes`) | 8-bit (`bitsandbytes`) | None (Full FP16/BF16) or 8-bit |
| **Per-Device Train Batch** | 4 | 4 | 16 |
| **Gradient Accumulation** | 2 | 4 | 2 |
| **Effective Batch Size** | 8 | **16** | **32** |
| **Peak Learning Rate** | $1 \times 10^{-4}$ | **$1 \times 10^{-4}$** | $5 \times 10^{-5}$ |
| **LR Scheduler** | Linear with warmup | **Cosine with warmup** | Cosine with warmup |
| **Warmup Steps** | 50 | **500** | 1000 |
| **Weight Decay** | 0.01 | 0.01 | 0.01 |
| **Gradient Checkpointing** | True | True | True |
| **Evaluation Strategy** | Every 100 steps | Every 500 steps | Every 500 steps |
| **Generation Max Length** | 225 tokens | 225 tokens | 225 tokens |
| **Predict with Generate** | True | True | True |
| **Metric for Best Model** | `wer` (lower is better) | `wer` | `wer` |

---

## 5. Forensic Breakdown of `dagbani_whisper_asr_colab_v2.ipynb`

A cell-by-cell architectural audit of the official production fine-tuning notebook:

```
dagbani_whisper_asr_colab_v2.ipynb
├── Cell 00 [MD]  : Title & Architectural Changelog
├── Cell 01-03    : Environment Setup, Pinned Pip Installs & GPU Introspection
├── Cell 04-05    : Global Hyperparameter Configurations & Pilot Switches
├── Cell 06-07    : HuggingFace Hub Authentication (Colab Secrets integration)
├── Cell 08-10    : SciDB Ingestion Engine & Unzipping with Duplication Guards
├── Cell 11-12    : Excel Workbook (`Dagbani.xlsx`) Header & Metadata Parser
├── Cell 13-14    : Dagbani UTF-8 Text Normalization & Orthography Diagnostics
├── Cell 15-16    : Audio Duration Filtering & Persistent JSON Disk Caching
├── Cell 17-18    : Speaker-Disjoint Split Partitioning (90/5/5%)
├── Cell 19-20    : WhisperProcessor & Byte-Fallback Tokenizer Round-Trip Tests
├── Cell 21-22    : Lazy Batch-Time Feature Extraction Architecture
├── Cell 23-24    : 8-Bit Model Quantization & LoRA Adapter Injection
├── Cell 25-26    : Custom Data Collator (`DataCollatorSpeechSeq2SeqWithPadding`) & Metrics
├── Cell 27-29    : Seq2SeqTrainer Execution & Auto-Resume Checkpoint Controller
├── Cell 30-33    : Test Set Evaluation, Sample Predictions & Special Glyph Precision/Recall
├── Cell 34-36    : Model Card Generation & Automated HuggingFace Hub Publishing
└── Cell 37-38    : Interactive Gradio Demo Deployment
```

### Key Architectural Invariants Discovered in Colab v2:
1. **Lazy Feature Extraction (Cells 21-22, 25-26)**:
   - Pre-computing log-mel spectrogram features across 10,000+ clips via `dataset.map` consumes $> 16\text{ GB}$ RAM, crashing Google Colab environments.
   - The notebook implements a dynamic collation pattern: `DataCollatorSpeechSeq2SeqWithPadding` computes `processor.feature_extractor` on raw audio waveforms dynamically *only for the current mini-batch*.
2. **Strict Speaker Disjointness (Cells 17-18)**:
   - Partitions unique speaker IDs (`SPEAKER_ID`) randomly into train ($90\%$), validation ($5\%$), and test ($5\%$) sets.
   - Prevents acoustic memorization of speaker vocal tract timbre, ensuring genuine generalized word error rate evaluation.
3. **Special Glyph Audit Metric (Cell 33)**:
   - Evaluates specific recall and precision for Dagbani distinctive letters (`ɛ`, `ɔ`, `ɣ`, `ŋ`, `ʒ`) on held-out predictions:
     $$\text{Recall}_c = \frac{\sum \min(N_{\text{truth}}(c), N_{\text{pred}}(c))}{\sum N_{\text{truth}}(c)}$$
     $$\text{Precision}_c = \frac{\sum \min(N_{\text{truth}}(c), N_{\text{pred}}(c))}{\sum N_{\text{pred}}(c)}$$

---

## 6. Dagbani Speech Corpora & Benchmark Catalog

| Dataset Name | Source / Institution | Hours | Speakers | Format | Transcriptions / Content | License / Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GhanaNLP Dagbani ASR Corpus** | GhanaNLP & Dagbani Wikimedians (AfricaNLP 2023) | **9.57 hrs** (9h 34m) | 22 (15M / 7F) | WAV / OGG 16kHz | 11,207 utterances, 48,636 total tokens, 7,222 unique tokens. Wikipedia articles & cultural texts. | **CC0 (Public Domain)** / Wikimedia Commons & Khaya |
| **SciDB Dagbani Speech Corpus** | SciDB Repository | ~15-25 hrs | Multi-speaker | MP3 16kHz + `Dagbani.xlsx` | Sentence-aligned read speech with demographic speaker metadata. | Research Use |
| **Mozilla Common Voice (Dagbani)** | Mozilla Foundation | Variable (Community) | Crowd | MP3 48kHz | Crowdsourced sentence readings validated by 2+ community upvotes. | **CC0** / HuggingFace `mozilla-foundation/common_voice_17_0` |
| **WAXAL Spontaneous Corpus** | WAXAL Project / CMU | Spontaneous | Diverse | WAV 16kHz | Image-prompted unscripted dialogues capturing natural code-switching. | Academic Research |
| **Dagbani Bible Audio** | Faith Comes By Hearing / Open.Bible | ~80 hrs | Dramatized / Single | MP3 44.1kHz | Complete New Testament audio recordings aligned with chapter/verse text. | Non-commercial Research |

---

## 7. Evaluation Framework, Error Metrics & Failure Modes

### 7.1 Metric Formulation
1. **Word Error Rate (WER)**:
   $$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Reference Words}}$$
2. **Character Error Rate (CER)**:
   $$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
3. **Normalized vs Strict Metric Calculation**:
   - **Strict WER**: Measures exact orthographic string equality (sensitive to punctuation, capitalization, and alternate spelling variants).
   - **Normalized WER**: Standardizes text via NFC normalization, lowercasing, and removal of non-spoken punctuation before Levenshtein alignment:
     ```python
     def normalize_dagbani_text(text: str) -> str:
         text = unicodedata.normalize("NFC", str(text))
         text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
         text = text.replace("“", '"').replace("”", '"').lower()
         cleaned = [ch for ch in text if unicodedata.category(ch).startswith("L") or ch in {"-", "'", " "}]
         return re.sub(r"\s+", " ", "".join(cleaned)).strip()
     ```
4. **Phoneme Error Rate (PER)**:
   - For orthographically fluid languages like Dagbani, Grapheme-to-Phoneme (G2P) conversion to IPA sequences eliminates false penalty inflations caused by dialectal spelling variants (*yɛltɔɣa* vs *yɛltoɣa*).

### 7.2 Primary Failure Modes & Forensic Mitigations
| Failure Mode | Root Cause | Observable Symptom | Engineering Mitigation |
| :--- | :--- | :--- | :--- |
| **Silent Chunk Hallucination** | Autoregressive language model prior generates repetitive phrases in silent audio. | Whisper repeats phrases like *"N nyɛla... N nyɛla..."* during background pauses. | 1. Implement Silero VAD before decoding.<br>2. Set `no_speech_threshold=0.6`.<br>3. Set `condition_on_previous_text=False`. |
| **Orthographic Inconsistency** | Non-standardized spelling across writers (*yɛltɔɣa* vs *yɛltoɣa*, *kɔbga* vs *kɔwa*). | Model predicts phonetically correct text penalized by strict WER. | 1. Apply standardized G2P text normalization.<br>2. Report both Strict WER and Normalized WER. |
| **Tone Confusion** | Tone is unmarked in text but distinguishes lexical meaning (*gballi* grave vs mat). | Semantic misinterpretation in downstream translation/LLM tasks. | 1. Ingest acoustic pitch contours ($F_0$).<br>2. Train downstream LLM with contextual disambiguation. |
| **Dialectal Acoustic Lenition** | Tomosili speakers elide consonants ($/bg/ \to [ʋ]$, $/gs/ \to [x]$). | Insertion/deletion errors in Western dialect speakers. | 1. Ensure balanced training data from Tamale and Yendi.<br>2. SpecAugment frequency/time masking. |
| **English Code-Switching** | Loanwords and mixed English in daily conversation (*school*, *hospital*, *doctor*). | Model phonetically transcribes English words using Dagbani orthography. | 1. Include code-switched training examples.<br>2. Multi-task pretraining on English-Dagbani parallel pairs. |

---

## 8. Production Inference, Optimization & Edge Deployment

### 8.1 CTranslate2 & `faster-whisper`
- **Engine Architecture**: C++ inference engine using cuBLAS and oneDNN with INT8 / FP16 tensor quantization.
- **Speedup**: $4\times$ faster than vanilla PyTorch HuggingFace pipelines with $70\%$ reduced memory footprint.
- **Production Snippet**:
  ```python
  from faster_whisper import WhisperModel
  
  # Load quantized INT8 model on GPU or CPU
  model = WhisperModel("ats-tech/dagbani-whisper-asr", device="cuda", compute_type="int8_float16")
  segments, info = model.transcribe("audio_sample.wav", beam_size=5, language=None, task="transcribe")
  for segment in segments:
      print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
  ```

### 8.2 `whisper.cpp` for Mobile & Edge Devices (Android / iOS / Raspberry Pi)
- Zero-dependency, pure C/C++ implementation of Whisper.
- Quantized weight formats: `Q4_0`, `Q5_0`, `Q8_0` allowing Whisper Small/Medium to run locally on ARM NEON / Apple Silicon processors in real-time ($< 250\text{ ms}$ latency per chunk).

### 8.3 Real-Time Streaming Architecture
- **Chunked Streaming Pipeline**:
  1. Capture microphone audio stream into a $16\text{ kHz}$ circular buffer.
  2. Emit VAD speech triggers every $1.0\text{s}$.
  3. Run local agreement heuristic across overlapping $3.0\text{s}$ windows: commit transcription prefix once two consecutive windows agree on decoded text.

---

## 9. Specification Miner Matrix Tables

### 9.1 Features Discovered Matrix
| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Audio Preproc | 16kHz Resampling | Converts arbitrary multi-format audio into uniform 16kHz mono PCM | Audio buffer / file path | Normalized `float32` array $[-1.0, 1.0]$ | Raises `RuntimeError` on corrupted codecs | Colab v2 / Audio specs |
| 2 | Audio Preproc | Log-Mel Extraction | 80/128-channel Mel filterbank with 25ms window and 10ms hop | $16\text{kHz}$ PCM waveform | Tensor shape $(B, 80/128, 3000)$ | Truncates $> 30\text{s}$, zero-pads $< 30\text{s}$ | Whisper Architecture |
| 3 | Audio Preproc | Duration Filtering | Filters out clips outside $[0.5\text{s}, 30.0\text{s}]$ and caches to JSON | File path & duration cache | Filtered record list | Skips and logs decode errors | Colab v2 Cell 15-16 |
| 4 | Text Normalization | Dagbani Normalizer | NFC normalization, lowercasing, punctuation stripping, preserving `ɛ, ɔ, ɣ, ŋ, ʒ` | Raw string | Sanitized Dagbani string | Removes empty rows post-cleaning | Colab v2 Cell 13-14 |
| 5 | Dataset Split | Speaker-Disjoint Split | Partitions dataset by unique speaker IDs (90/5/5%) | List of metadata records | `DatasetDict(train, val, test)` | Ensures 0 speaker overlap | Colab v2 Cell 17-18 |
| 6 | Tokenization | Byte-Fallback BPE | Decomposes Dagbani special characters into multi-byte UTF-8 tokens | UTF-8 text string | Integer token IDs $\in [0, 51865]$ | Verified exact round-trip match | Colab v2 Cell 19-20 |
| 7 | Model Architecture | 8-bit LoRA PEFT | Injects low-rank adapters into $W_0$ in 8-bit quantized base model | Quantized Whisper weights | Trainable adapter weights | Reduces VRAM from 28GB to 10.5GB | Colab v2 Cell 23-24 |
| 8 | Training Loop | Lazy Batch Collation | Extracts log-mel features and pads labels dynamically per batch | Raw batch audio records | PyTorch batch tensor dict | Avoids host RAM exhaustion | Colab v2 Cell 25-26 |
| 9 | Evaluation | Special Glyph Audit | Computes recall and precision for `ɛ, ɔ, ɣ, ŋ, ʒ` on held-out predictions | Ground truth & predictions | Per-character recall/precision % | Highlights orthographic drift | Colab v2 Cell 33 |
| 10 | Deployment | Faster-Whisper INT8 | Optimized C++ inference using CTranslate2 and INT8 quantization | $16\text{kHz}$ Audio waveform | Fast transcription segments | 4x faster than PyTorch | Inference specs |

### 9.2 Edge Cases & Observed Behavior
| # | Feature | Input | Observed Behavior | Mitigation / Solution |
|---|---------|-------|-------------------|----------------------|
| 1 | Whisper Processor | Special glyph round-trip: `"n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"` | Encodes to byte tokens and decodes back to exact original string | Use native Whisper tokenizer without vocabulary modification |
| 2 | Audio Loader | Audio longer than 30 seconds ($> 30.0\text{s}$) | Whisper feature extractor silently truncates at 30.0s, dropping trailing text | Filter or VAD-chunk audio to $\le 30.0\text{s}$ prior to training |
| 3 | Audio Loader | Audio shorter than 0.5 seconds ($< 0.5\text{s}$) | May contain clicks, breath noise, or trigger CTC/loss instabilities | Filter out all audio $< 0.5\text{s}$ via `MIN_DURATION_SEC` |
| 4 | Language ID | Forcing unsupported language token (e.g. `<|en|>`) | Biases language model decoder toward English phonology and vocabulary | Set `forced_decoder_ids = None` and `task = "transcribe"` |
| 5 | Metadata Ingestion | Missing Excel columns or corrupted rows | Crashes naive positional parsers | Use explicit dictionary-mapped header parser (`openpyxl`) |
| 6 | Decoder Generation | Silence / continuous background noise | Model hallucinates repetitive phrases (*"N nyɛla..."*) | Set `no_speech_threshold = 0.6` and condition on VAD |
| 7 | Orthography Metric | Dialect spelling variations (*yɛltɔɣa* vs *yɛltoɣa*) | Exact WER reports false positive error penalty | Report both Strict WER and Normalized / Phoneme Error Rate |
