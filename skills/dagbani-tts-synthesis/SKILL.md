---
name: dagbani-tts-synthesis
description: >-
  Production-grade Text-to-Speech (TTS) synthesis and acoustic modeling suite for Dagbani (Dagbanli, ISO 639-3: dag).
  Provides end-to-end VITS, Coqui XTTS-v2, Matcha-TTS, and Meta MMS-TTS architectures, custom G2P phonemizer with
  tone-tier tagging, HiFi-GAN/BigVGAN neural vocoder fine-tuning, dataset preprocessing, and batch/real-time inference.
---

# Dagbani TTS Synthesis Skill

The `dagbani-tts-synthesis` skill provides a complete, production-ready framework for building, training, fine-tuning, and deploying neural Text-to-Speech (TTS) models for Dagbani (*Dagbanli*, ISO 639-3: `dag`).

Dagbani is a Mabia / Gur language with unique acoustic characteristics:
- **Two-level register tone system** (High [H] and Low [L]) with automatic and grammatical downstep (!H), producing critical lexical minimal pairs (e.g., *gballi* [ɡbálːí] 'grave' vs. *gballi* [ɡbálːì] 'zana mat').
- **Advanced Tongue Root ([±ATR]) vowel harmony** and phonemic length contrasts across 11 vowels (6 short, 5 long) and 5 nasalized vowels.
- **Co-articulated labial-velar stops** (`kp` [k͡p], `gb` [ɡ͡b], `ŋm` [ŋ͡m]), palatal nasals (`ny` [ɲ]), velar fricatives (`ɣ` [ɣ]), and voiced post-alveolar fricatives (`ʒ` [ʒ]).
- **Moraic tone-bearing coda nasals** (`m, n, ŋ`) that function as independent prosodic Tone-Bearing Units (TBUs).

---

## Architecture Overview

```
                           ┌──────────────────────────────────────────────┐
                           │          Dagbani Input Text (BGL/ASCII)       │
                           └──────────────────────────────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │     dagbani_phonemizer.py                    │
                           │  - Unicode NFC Normalization                 │
                           │  - ASCII to BGL Disambiguation               │
                           │  - Multi-char Digraph Tokenizer              │
                           │  - Contextual Allophonic Mutation            │
                           │  - Tone Diacritic Restoration (H, L, !H)     │
                           │  - Moraic Coda Segmentation                  │
                           └──────────────────────────────────────────────┘
                                                  │
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │ Phoneme Token IDs + Tone & Duration Sequence │
                           └──────────────────────────────────────────────┘
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
       ┌──────────────────────────────┐                       ┌──────────────────────────────┐
       │   VITS End-to-End Synthesis  │                       │   Matcha-TTS / FastSpeech 2  │
       │  - Posterior Encoder         │                       │  - Optimal Transport Flow    │
       │  - Normalizing Flows         │                       │  - Mel-Spectrogram Predictor │
       │  - Stochastic Duration Pred  │                       └──────────────────────────────┘
       │  - Integrated HiFi-GAN Dec   │                                       │
       └──────────────────────────────┘                                       ▼
                      │                                       ┌──────────────────────────────┐
                      │                                       │    BigVGAN / HiFi-GAN Vocoder│
                      │                                       │    (Snake Anti-Aliased Acts) │
                      │                                       └──────────────────────────────┘
                      │                                                       │
                      └───────────────────────────┬───────────────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │ 24 kHz / 48 kHz High-Fidelity Audio │
                               └─────────────────────────────────────┘
```

---

## Directory Structure

```
skills/dagbani-tts-synthesis/
├── SKILL.md                              # This specification file
├── references/
│   ├── vits_architecture_guide.md        # VITS, XTTS-v2, Matcha-TTS & MMS specifications
│   ├── phonemizer_pipeline.md            # G2P, tone integration, moraic prosody rules
│   └── vocoder_finetuning.md             # HiFi-GAN & BigVGAN vocoder training recipes
├── examples/
│   ├── vits_training_example.md          # Single-speaker & multi-speaker VITS pipeline
│   └── voice_synthesis_workflow.md       # CLI and Python API inference runbooks
└── scripts/
    ├── dagbani_phonemizer.py             # Rule-based & lexical G2P phonemizer CLI/API
    ├── prepare_tts_dataset.py            # Audio hygiene, silence trimming, alignment
    └── synthesize_tts.py                 # Multi-backend TTS inference runner
```

---

## Quick Start Guide

### 1. Phonemizing Dagbani Text

Transform orthographic text (BGL Unicode or ASCII fallback) into IPA phoneme symbols with tone markers and token IDs:

```bash
# Direct CLI conversion
python scripts/dagbani_phonemizer.py --text "O biɛla Yendi zúŋɔ" --format json

# Interactive or file-based conversion
python scripts/dagbani_phonemizer.py --input-file raw_sentences.txt --output-file phonemized.tsv
```

Example Python API usage:
```python
from scripts.dagbani_phonemizer import DagbaniPhonemizer

phonemizer = DagbaniPhonemizer()
result = phonemizer.phonemize("O biɛla Yendi", return_token_ids=True)
print(result["ipa"])        # "o b j ɛ́ l á j é n d ì"
print(result["token_ids"])  # [18, 1, 23, 27, 28, 14, 27, 29, 21, 27, 16, 4, 11, 28]
```

### 2. Preparing Audio & Text Datasets

Process raw recordings (e.g. WAXAL, Common Voice, BibleTTS) into standard LJSpeech/VITS format with EBU R128 loudness normalization and silence trimming:

```bash
python scripts/prepare_tts_dataset.py \
    --audio-dir raw_audio/ \
    --transcripts raw_transcripts.tsv \
    --output-dir dataset_vits/ \
    --sample-rate 24000 \
    --target-lufs -23.0 \
    --min-duration 1.0 \
    --max-duration 12.0
```

### 3. Synthesizing Speech

Generate WAV audio from text using a pre-trained checkpoint:

```bash
python scripts/synthesize_tts.py \
    --text "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam." \
    --model-type vits \
    --checkpoint checkpoints/vits_dagbani_v2.pt \
    --output-wav output/sample_01.wav \
    --speaker-id 0 \
    --noise-scale 0.667 \
    --length-scale 1.0
```

---

## Key Technical Decisions for Dagbani TTS

1. **Deterministic Digraph Preservation**: Never segment `kp`, `gb`, `ŋm`, `ny`, `ch`, `sh` as separate single letters. They are mapped to unitary phoneme IDs `/k͡p/`, `/ɡ͡b/`, `/ŋ͡m/`, `/ɲ/`, `/t͡ʃ/`, `/ʃ/`.
2. **Moraic Coda Tone Tiers**: Syllable-final nasals (`/m/, /n/, /ŋ/`) bear independent tone mora. The phonemizer outputs explicit high/low tone tokens attached to codas.
3. **BigVGAN Vocoder Adaptation**: BigVGAN with Snake activations avoids metallic robotic artifacts common in tonal languages during rapid pitch inflections and downdrift.
