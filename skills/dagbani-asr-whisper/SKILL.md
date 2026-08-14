---
name: dagbani-asr-whisper
description: Production-grade ASR fine-tuning toolkit, audio preprocessor, Whisper LoRA/QLoRA trainer, and speech evaluation metrics for Dagbani (Dagbanli).
---

# Dagbani ASR & Whisper Fine-Tuning Skill

The `dagbani-asr-whisper` skill provides an end-to-end Automatic Speech Recognition (ASR) fine-tuning, inference, and evaluation pipeline for the Dagbani language (*Dagbanli*, ISO 639-3: `dag`). It enables adapting OpenAI Whisper (Tiny, Base, Small, Medium, Large-v3) to low-resource Dagbani speech using Parameter-Efficient Fine-Tuning (LoRA/PEFT), 8-bit quantization (`bitsandbytes`), Voice Activity Detection (VAD) audio chunking, and special glyph precision/recall tracking.

---

## 1. Capabilities & Core Modules

1. **Audio Preprocessor & Feature Extraction (`scripts/audio_preprocessor.py`)**:
   - Resamples multi-format audio (WAV, MP3, FLAC, OGG) to 16 kHz single-channel mono PCM.
   - Computes 80-channel (Whisper Small/Medium) and 128-channel (Whisper Large-v3) log-magnitude Mel spectrograms.
   - Energy and Silero-style Voice Activity Detection (VAD) segmentation for long-form speech (>30s).
   - Duration filtering (`0.5s` to `30.0s`) and persistent JSON metadata caching.

2. **Whisper Dagbani Fine-Tuning Engine (`scripts/whisper_dagbani_trainer.py`)**:
   - Parameter-Efficient Fine-Tuning (PEFT) via LoRA ($r=16, \alpha=32$, targeting `q_proj, v_proj, out_proj`).
   - 8-bit Quantization (`load_in_8bit=True`) enabling Whisper Medium fine-tuning on a single 16GB GPU (NVIDIA T4 / V100 / RTX 3090).
   - Lazy dynamic batch collation (`DataCollatorSpeechSeq2SeqWithPadding`) preventing host RAM exhaustion.
   - Byte-fallback tokenizer configuration preventing `<unk>` degradation on Dagbani extended glyphs (`ɛ, ɔ, ŋ, ɣ, ʒ`).

3. **ASR Evaluation & Special Character Audit (`scripts/evaluate_asr.py`)**:
   - Word Error Rate (WER) and Character Error Rate (CER).
   - Normalized WER (applying NFC normalization, lowercase, and spoken-punctuation sanitization).
   - Special glyph precision, recall, and F1-score for Dagbani characters: `ɛ, ɔ, ŋ, ɣ, ʒ`.

---

## 2. Directory Structure

```
skills/dagbani-asr-whisper/
├── SKILL.md                             # Main skill documentation and usage instructions
├── references/
│   ├── whisper_tuning_guide.md          # LoRA/QLoRA recipes, hyperparams & VRAM requirements
│   ├── audio_preprocessing_spec.md      # 16kHz standards, log-mel filterbanks, VAD chunking
│   └── evaluation_metrics.md            # WER, Normalized WER, CER & special glyph recall
├── examples/
│   ├── colab_training_pipeline.md       # Step-by-step training pipeline walkthrough
│   └── inference_transcription.md       # CLI and Python inference code samples
└── scripts/
    ├── audio_preprocessor.py            # Audio resampling, VAD chunking & log-mel generator
    ├── whisper_dagbani_trainer.py       # Fine-tuning engine with LoRA & 8-bit PEFT
    └── evaluate_asr.py                  # Evaluation tool for WER, CER & glyph metrics
```

---

## 3. Quick Start & CLI Usage

### 3.1 Audio Preprocessing & Chunking
```bash
# Ingest and resample a directory of raw audio recordings
python skills/dagbani-asr-whisper/scripts/audio_preprocessor.py \
  --input-dir ./raw_audio/ \
  --output-dir ./processed_16k/ \
  --manifest ./manifest.json \
  --target-sr 16000

# VAD chunking for a long audio recording
python skills/dagbani-asr-whisper/scripts/audio_preprocessor.py \
  --input-file ./long_interview.wav \
  --chunk-vad \
  --max-duration 28.0 \
  --output-dir ./chunks/
```

### 3.2 Whisper LoRA Fine-Tuning
```bash
# Launch fine-tuning with 8-bit LoRA on Whisper Medium
python skills/dagbani-asr-whisper/scripts/whisper_dagbani_trainer.py \
  --base-model openai/whisper-medium \
  --train-manifest ./train_manifest.json \
  --val-manifest ./val_manifest.json \
  --output-dir ./checkpoints/whisper-dagbani-medium-lora \
  --per-device-batch 4 \
  --grad-accum 4 \
  --learning-rate 1e-4 \
  --max-steps 3000 \
  --quant-8bit \
  --fp16
```

### 3.3 ASR Evaluation & Glyph Precision/Recall
```bash
# Evaluate hypothesis transcriptions against reference ground truth
python skills/dagbani-asr-whisper/scripts/evaluate_asr.py \
  --reference-file ./references.txt \
  --hypothesis-file ./predictions.txt \
  --output-report ./eval_summary.json

# Run built-in self tests
python skills/dagbani-asr-whisper/scripts/evaluate_asr.py --self-test
```

---

## 4. Python API Usage

```python
from skills.dagbani_asr_whisper.scripts.audio_preprocessor import AudioPreprocessor
from skills.dagbani_asr_whisper.scripts.evaluate_asr import ASREvaluator

# 1. Preprocess and validate an audio file
preprocessor = AudioPreprocessor(target_sr=16000)
waveform, sr = preprocessor.load_audio("sample.mp3")
mel_spec = preprocessor.compute_log_mel(waveform) # (80, 3000)

# 2. Evaluate transcriptions
evaluator = ASREvaluator()
refs = ["n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"]
hyps = ["n nyela be paghanga mini o ka zhem sheli"]

metrics = evaluator.evaluate(refs, hyps)
print(f"Normalized WER: {metrics['normalized_wer']:.2%}")
print(f"Special Glyph Recall: {metrics['special_glyph_metrics']['overall_recall']:.2%}")
```

---

## 5. Architectural Invariants for Low-Resource Dagbani ASR

- **Language ID Suppression**: Do NOT force foreign language tokens (e.g. `<|en|>`). Set `forced_decoder_ids = None` and `task = "transcribe"`.
- **Special Glyph Round-Trip**: The native byte-fallback BPE tokenizer round-trips all Dagbani glyphs (`ɛ, ɔ, ŋ, ɣ, ʒ`) cleanly without out-of-vocabulary (`<unk>`) token loss.
- **Speaker Disjointness**: Always partition evaluation datasets strictly by speaker ID to prevent acoustic memorization and ensure genuine generalization.
