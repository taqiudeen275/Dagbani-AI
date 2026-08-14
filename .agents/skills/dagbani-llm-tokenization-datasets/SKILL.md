---
name: dagbani-llm-tokenization-datasets
description: >-
  Production-grade Large Language Model (LLM) tokenization, continual pre-training, parameter-efficient fine-tuning (LoRA/QLoRA),
  and dataset curation suite for Dagbani (Dagbanli, ISO 639-3: dag). Includes custom Byte-Level BPE tokenizer training
  preserving digraphs and BGL glyphs, text hygiene/cleaning pipelines, and Hugging Face LoRA instruction tuning recipes.
---

# Dagbani LLM Tokenization, Datasets & Adaptation Skill

The `dagbani-llm-tokenization-datasets` skill provides a complete, production-ready framework for training custom subword tokenizers, curating high-quality text datasets, and fine-tuning foundation Large Language Models (LLaMA-3.1, Aya-23, Mistral, Gemma-2) for Dagbani (*Dagbanli*, ISO 639-3: `dag`).

---

## Technical Challenges Addressed

1. **Catastrophic Subword Fertility**: Off-the-shelf LLM tokenizers (e.g. LLaMA-3, Mistral) have high fertility on Dagbani (**3.82 to 4.25 tokens/word**), fragmenting Dagbani letters (`ɛ`, `ɔ`, `ŋ`, `ɣ`, `ʒ`) into individual raw bytes and destroying semantic root morphemes.
2. **Agglutinative Nominal & Verbal Morphology**: Dagbani nominal class prefixes/suffixes (*wab-gu* -> *wab-ri*) and verbal aspect markers require morpheme-aligned subword merges.
3. **Data Scarcity & Quality Control**: Native digitized text is scarce (~50M tokens). Continual Pre-Training (CPT) and instruction tuning require rigorous deduplication, NFC normalization, script filtering, and bilingual curriculum balancing (70% Dagbani : 30% English).

---

## Architecture Overview

```
                          ┌───────────────────────────────────────────────┐
                          │   Raw Dagbani Text / Dumps (Wikipedia, BGL)   │
                          └───────────────────────────────────────────────┘
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │             dataset_cleaner.py                │
                          │  - Unicode NFC Normalization                  │
                          │  - BGL Extended Glyph Validation (ɛ, ɔ, ŋ, ɣ, ʒ)│
                          │  - Min/Max Length & Non-Latin Filter          │
                          │  - MinHash / Exact Deduplication              │
                          └───────────────────────────────────────────────┘
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │            Curated Text Corpus                │
                          └───────────────────────────────────────────────┘
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
       ┌──────────────────────────────┐                       ┌──────────────────────────────┐
       │  train_dagbani_tokenizer.py  │                       │    llm_lora_finetuner.py     │
       │  - Byte-Level BPE (8k–16k)   │                       │  - 4-bit / 8-bit QLoRA       │
       │  - Digraph Preservation      │                       │  - Rank r=64, Alpha=64       │
       │  - Low Fertility (1.28 tok/w)│                       │  - All Linear Target Modules │
       │  - HuggingFace Export        │                       │  - Chat Template Formatting  │
       └──────────────────────────────┘                       └──────────────────────────────┘
                      │                                                       │
                      ▼                                                       ▼
       ┌──────────────────────────────┐                       ┌──────────────────────────────┐
       │   Expanded Tokenizer Repo    │                       │  Adapted Dagbani LLM Adapter │
       │ (tokenizer.json, merges.txt) │                       │     (LoRA Safetensors)       │
       └──────────────────────────────┘                       └──────────────────────────────┘
```

---

## Directory Structure

```
skills/dagbani-llm-tokenization-datasets/
├── SKILL.md                              # This specification file
├── references/
│   ├── tokenization_fertility_guide.md   # Subword fertility benchmarks, byte-fallback analysis
│   ├── llm_adaptation_recipes.md         # QLoRA, continual pre-training, bilingual curriculum
│   └── dataset_curation_standards.md     # Cleaning rules, deduplication, NFC, licensing
├── examples/
│   ├── tokenizer_training_example.md     # Custom BPE tokenizer training walkthrough
│   └── lora_finetuning_pipeline.md       # Step-by-step instruction tuning on Dagbani QA/chat
└── scripts/
    ├── train_dagbani_tokenizer.py        # Byte-Level BPE tokenizer training script
    ├── dataset_cleaner.py                # Text normalization, validation, and deduplicator
    └── llm_lora_finetuner.py             # PyTorch/HuggingFace LoRA/QLoRA fine-tuner CLI
```

---

## Quick Start Guide

### 1. Cleaning and Normalizing Raw Corpora

Clean raw text files, normalize Unicode to NFC, remove corrupted characters, and deduplicate lines:

```bash
python scripts/dataset_cleaner.py \
    --input-file raw_dagbani_corpus.txt \
    --output-file cleaned_corpus.txt \
    --min-words 3 \
    --max-words 256 \
    --deduplicate
```

### 2. Training a Custom Dagbani BPE Tokenizer

Train an 8,000-vocabulary Byte-Level BPE tokenizer with digraphs preserved as atomic subwords:

```bash
python scripts/train_dagbani_tokenizer.py \
    --corpus-file cleaned_corpus.txt \
    --vocab-size 8000 \
    --output-dir tokenizers/dagbani_bpe_8k/ \
    --preserve-digraphs
```

### 3. Fine-Tuning LLMs with LoRA / QLoRA

Fine-tune Meta-Llama-3.1-8B-Instruct on Dagbani instruction/chat datasets:

```bash
python scripts/llm_lora_finetuner.py \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-file dagbani_instructions.jsonl \
    --output-dir checkpoints/dagbani_llama_lora/ \
    --lora-r 64 \
    --lora-alpha 64 \
    --learning-rate 2e-4 \
    --epochs 3 \
    --quantization 4bit
```

---

## Core Linguistic Rules Enforced

1. **BGL Extended Unicode Glyphs**: Explicitly accommodates `ɛ` (U+025B), `ɔ` (U+0254), `ŋ` (U+014B), `ɣ` (U+0263), `ʒ` (U+0292), and glottal apostrophes `'` (U+0027) without stripping or corrupting them during cleaning.
2. **Atomic Digraph Merging**: Pre-tokenization rules prioritize merging `kp`, `gb`, `ŋm`, `ny`, `ch`, `sh` early in BPE merges.
3. **Embedding Weight Initialization**: When extending base LLM vocabularies, new embeddings are initialized using subword average projections to prevent catastrophic initial loss spikes.
