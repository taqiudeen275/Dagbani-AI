# Project Plan: Comprehensive Dagbani AI Knowledge Base & Antigravity Skills

## Objective
Synthesize 10 Dagbani research materials into:
1. 5 In-Depth Knowledge Base Playbooks in `knowledge/`.
2. 4 Production-Grade Antigravity Skills in `.agents/skills/` (and mirrored in `skills/`).
3. Fully functional and verified Python scripts, G2P, text normalizers, training harnesses, and datasets.
4. Comprehensive verification and testing.

## Execution Phases & Milestones

### Phase 0: Survey & Deep Content Mining (Current)
- Dispatch 3 Explorers / Spec Miners:
  - `explorer_ling`: Deep analysis of Dagbani phonology, ATR vowel harmony, tones, 1998 BGL orthography vs modern revisions, G2P rules, IPA mappings from Thesis, Drumming, and related papers.
  - `explorer_asr`: Deep analysis of ASR models, Whisper fine-tuning notebooks, APSIPA 2025 paper, low-resource speech strategies, LoRA configs, audio preprocessing, eval metrics (WER, CER).
  - `explorer_tts_llm`: Deep analysis of TTS acoustic models (VITS, FastSpeech2, Matcha-TTS, MMS), Dagbani LLM tokenization (BPE, byte-level, fertility), pretraining & LoRA fine-tuning, datasets (Common Voice, Bible, LoresLM, Wikimedia, Drum history).

### Milestone 1: Master Knowledge Base Synthesis
- Generate 5 comprehensive, highly structured markdown guides in `knowledge/`:
  1. `knowledge/dagbani_phonology_orthography_guide.md`
  2. `knowledge/dagbani_asr_whisper_playbook.md`
  3. `knowledge/dagbani_tts_acoustic_playbook.md`
  4. `knowledge/dagbani_llm_pretraining_finetuning_guide.md`
  5. `knowledge/dagbani_datasets_and_benchmarks_catalog.md`

### Milestone 2: Antigravity Skills Suite Architecture & Implementation
- Build 4 complete Antigravity skills in `.agents/skills/` and mirror in `skills/`:
  1. `dagbani-linguistics`: SKILL.md, references (phoneme inventory, tone rules, orthography), examples, scripts (G2P engine, orthography normalizer, syllabifier).
  2. `dagbani-asr-whisper`: SKILL.md, references (Whisper fine-tuning, LoRA configs, data curation), examples, scripts (audio pipeline, Whisper trainer/evaluator, inference script).
  3. `dagbani-tts-synthesis`: SKILL.md, references (acoustic models, phoneme duration, vocoder), examples, scripts (phonemizer pipeline, VITS dataset prep, synthesis demo).
  4. `dagbani-llm-tokenization-datasets`: SKILL.md, references (tokenizers, data cleaning, prompt templates), examples, scripts (BPE tokenizer trainer, dataset scrubber, LoRA finetuner).

### Milestone 3: Implementation of Working Tooling & Codebase
- Ensure all scripts in `scripts/` of each skill are production-quality, syntax-checked, fully documented, and executable.

### Milestone 4: Verification, Review, Adversarial Stress Testing & Forensic Integrity Audit
- Reviewers inspect completeness, correctness, and adherence to Dagbani linguistic truth.
- Challengers test scripts, tokenization fertility, edge-case orthography, phoneme conversions, and pipeline invocations.
- Forensic Auditor validates authenticity and zero shortcuts.
- Update documentation and prepare final synthesis report.
