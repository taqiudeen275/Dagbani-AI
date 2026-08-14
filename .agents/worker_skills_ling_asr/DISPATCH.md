# DISPATCH LOG

## 2026-08-14T21:05:57Z
You are the Linguistics & ASR Skills Implementer for the Dagbani AI project.
Your working directory is: d:/ATS Tech/Dagbani AI/.agents/worker_skills_ling_asr/
Project Root: d:/ATS Tech/Dagbani AI
Read ORIGINAL_REQUEST.md: d:/ATS Tech/Dagbani AI/ORIGINAL_REQUEST.md
Read PROJECT.md: d:/ATS Tech/Dagbani AI/PROJECT.md

Your inputs:
1. `d:/ATS Tech/Dagbani AI/.agents/explorer_ling/survey_linguistics_report.md`
2. `d:/ATS Tech/Dagbani AI/.agents/explorer_asr/survey_asr_report.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your mission:
Build 2 complete, production-ready Antigravity skills in `skills/` and mirror them to `.agents/skills/`:

1. `skills/dagbani-linguistics/` (and `.agents/skills/dagbani-linguistics/`):
   - `SKILL.md`: Valid YAML frontmatter (`name: dagbani-linguistics`, `description: ...`), comprehensive usage instructions, prompt instructions, and tool integration guides.
   - `references/`:
     - `phonology_matrix.md`: Full IPA consonants, vowels, ATR harmony rules, tone registers.
     - `orthography_bgl.md`: 1998 BGL standard, ASCII conversion tables, digraph handling.
     - `morphophonology_rules.md`: Nasal assimilation, lenition, hiatus elision, tone sandhi.
   - `examples/`:
     - `g2p_conversion.md`: Step-by-step G2P examples for complex Dagbani words, loanwords, and phrases.
     - `orthography_normalization.md`: Examples of noisy/ASCII text normalization.
   - `scripts/`:
     - `dagbani_g2p.py`: Production-grade Grapheme-to-Phoneme converter for Dagbani (BGL and ASCII input -> IPA output with ATR harmony and tone support). CLI and importable module.
     - `orthography_normalizer.py`: Normalizer for Dagbani text (Unicode NFC, ASCII transliteration <-> BGL conversion, punctuation and whitespace hygiene).
     - `dagbani_syllabifier.py`: Syllable parser identifying onsets, nuclei, codas, moraic nasals, and tone-bearing units.

2. `skills/dagbani-asr-whisper/` (and `.agents/skills/dagbani-asr-whisper/`):
   - `SKILL.md`: Valid YAML frontmatter (`name: dagbani-asr-whisper`, `description: ...`), comprehensive Whisper fine-tuning workflows, inference commands, and error handling.
   - `references/`:
     - `whisper_tuning_guide.md`: Whisper Small/Medium/Large-v3 LoRA/QLoRA recipes, hyperparams, batching.
     - `audio_preprocessing_spec.md`: 16kHz sampling, 80/128 log-mel, VAD segmentation, SpecAugment.
     - `evaluation_metrics.md`: Standard WER, Normalized WER, CER, special glyph recall (ɛ, ɔ, ŋ, ɣ, ʒ).
   - `examples/`:
     - `colab_training_pipeline.md`: Full workflow walkthrough of Whisper fine-tuning on Dagbani audio.
     - `inference_transcription.md`: CLI and Python script examples for transcribing Dagbani audio files.
   - `scripts/`:
     - `audio_preprocessor.py`: Audio loading, resample to 16kHz mono, VAD chunking, log-mel feature extraction.
     - `whisper_dagbani_trainer.py`: Fine-tuning script with LoRA/PEFT, BitsAndBytes 8-bit quantization, custom data collator, and evaluation loop.
     - `evaluate_asr.py`: Evaluation tool computing WER, CER, Normalized WER, and special character precision/recall.

Ensure all Python scripts are fully functional, include docstrings, type hints, CLI `argparse` entry points, and can run self-tests/dry-runs cleanly.
Mirror all files created in `skills/` to `.agents/skills/`.
