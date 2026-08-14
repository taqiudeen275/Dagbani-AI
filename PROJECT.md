# Project: Dagbani AI Knowledge Base & Antigravity Skills Suite

## Architecture
- **Linguistics & Orthography Layer**: Vowel harmony (+/- ATR), consonants, tones, IPA, 1998 Bureau of Ghana Languages (BGL) orthography, G2P conversion rules.
- **ASR Layer**: Whisper fine-tuning (Whisper Small/Medium/Large-v3), LoRA/PEFT, CTC/hybrid models, WER/CER evaluation, audio preprocessing.
- **TTS Layer**: Acoustic models (VITS, FastSpeech2, Matcha-TTS, MMS-TTS), phonemizer pipelines, duration modeling, vocoding.
- **LLM & Tokenization Layer**: Byte-level / BPE tokenization, vocabulary fertility analysis, pretraining & instruction fine-tuning, low-resource transfer learning.
- **Dataset & Benchmark Layer**: Common Voice, Dagbani Bible, LoresLM corpora, Wikipedia/Wikimedia, Drumming texts, benchmark protocols.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Phonology & Orthography Guide | Comprehensive phonology, vowel harmony (+/- ATR), tone system, BGL orthography, G2P rules | M1 | Thesis, Drumming, Papers 1-2 |
| 2 | ASR Whisper Playbook | Whisper architecture, fine-tuning recipes, LoRA params, data augmentation, Colab & cluster setups | M1 | APSIPA2025, Colab notebook, Papers 1-2 |
| 3 | TTS Acoustic Playbook | End-to-end TTS architectures (VITS, FastSpeech2, MMS), dataset alignment, phoneme representation | M1 | Strategy HTML, Building TTS Paper |
| 4 | LLM Pretraining & Finetuning Guide | Tokenizer fertility, byte-fallback, LLaMA/Mistral/Gemma adaptation, LoRA configs, alignment | M1 | LoresLM, Strategy HTML, Thesis |
| 5 | Datasets & Benchmarks Catalog | Catalog of all Dagbani speech & text datasets, splits, preprocessing pipelines, quality audits | M1 | All 10 Sources |
| 6 | Skill: `dagbani-linguistics` | Antigravity skill with SKILL.md, references, examples, G2P converter, orthography normalizer scripts | M2, M3 | Linguistics Specs |
| 7 | Skill: `dagbani-asr-whisper` | Antigravity skill with SKILL.md, references, examples, Whisper fine-tuning & evaluation scripts | M2, M3 | ASR Specs |
| 8 | Skill: `dagbani-tts-synthesis` | Antigravity skill with SKILL.md, references, examples, phonemizer, dataset prep & inference scripts | M2, M3 | TTS Specs |
| 9 | Skill: `dagbani-llm-tokenization-datasets` | Antigravity skill with SKILL.md, references, examples, BPE trainer, dataset cleaner, LoRA scripts | M2, M3 | LLM Specs |
| 10 | Executable Tooling & Demos | Working, tested, and self-contained Python scripts for G2P, Whisper training, TTS prep, BPE tokenization | M3 | All Specs |
| 11 | Comprehensive Verification & Auditing | Unit tests, stress tests, review gate, adversarial challenges, forensic integrity audit | M4 | Quality & Integrity Specs |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Survey & Document Analysis | Survey all 10 documents, extract detailed phonetic rules, model hyperparams, datasets | none | DONE |
| 1 | Master Knowledge Base Synthesis | 5 deep-dive guides in `knowledge/` | M0 | DONE |
| 2 | Antigravity Skills Suite | 4 skills in `.agents/skills/` and `skills/` | M0, M1 | DONE |
| 3 | Executable Tooling & Codebase | Functional Python scripts in each skill's `scripts/` dir | M2 | DONE |
| 4 | Verification, Testing & Audit | Reviews, adversarial tests, forensic audit, final validation | M1, M2, M3 | DONE |

## Code Layout
```
d:/ATS Tech/Dagbani AI/
├── ORIGINAL_REQUEST.md
├── PROJECT.md
├── .agents/
│   ├── orchestrator_1/
│   │   ├── BRIEFING.md
│   │   ├── DISPATCH.md
│   │   ├── plan.md
│   │   ├── progress.md
│   │   └── GATE_STATUS.md
│   └── skills/
│       ├── dagbani-linguistics/
│       ├── dagbani-asr-whisper/
│       ├── dagbani-tts-synthesis/
│       └── dagbani-llm-tokenization-datasets/
├── skills/                      # Mirrored production skills
│   ├── dagbani-linguistics/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   ├── examples/
│   │   └── scripts/
│   ├── dagbani-asr-whisper/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   ├── examples/
│   │   └── scripts/
│   ├── dagbani-tts-synthesis/
│   │   ├── SKILL.md
│   │   ├── references/
│   │   ├── examples/
│   │   └── scripts/
│   └── dagbani-llm-tokenization-datasets/
│       ├── SKILL.md
│       ├── references/
│       ├── examples/
│       └── scripts/
├── knowledge/
│   ├── dagbani_phonology_orthography_guide.md
│   ├── dagbani_asr_whisper_playbook.md
│   ├── dagbani_tts_acoustic_playbook.md
│   ├── dagbani_llm_pretraining_finetuning_guide.md
│   └── dagbani_datasets_and_benchmarks_catalog.md
├── research/
└── resources/
```
