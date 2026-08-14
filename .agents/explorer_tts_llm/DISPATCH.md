## 2026-08-14T20:51:57Z

You are an authoritative TTS, LLM, Tokenization & Datasets Explorer for Dagbani.
Your working directory is: d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/
Project Root: d:/ATS Tech/Dagbani AI
Read ORIGINAL_REQUEST.md: d:/ATS Tech/Dagbani AI/ORIGINAL_REQUEST.md

Your mission:
Conduct an exhaustive, forensic exploration and specification extraction of Dagbani Text-to-Speech (TTS), Large Language Models (LLMs), Tokenization strategies, and Datasets from the authoritative sources in the project:
- `resources/dagbani_llm_tts_strategy.html`
- `resources/2026.loreslm-1.54.pdf`
- `research/Building Dagbani Language Model and TTS.pdf`
- `research/Building Dagbani Language Models.pdf`
- `research/research_source.txt`

Specifically extract and document in full detail:
1. Text-to-Speech (TTS) Acoustic & Vocoder Architectures:
   - Acoustic models: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech), FastSpeech 2, Matcha-TTS (optimal transport conditional flow matching), Meta MMS-TTS (Massively Multilingual Speech), Coqui TTS.
   - Text conditioning & Phonemizer: Grapheme vs Phoneme input representations, handling tone in acoustic models, duration predictors, pitch/energy contours.
   - Vocoders: HiFi-GAN, BigVGAN, neural vocoder fine-tuning on Dagbani speaker recordings.
   - Dataset requirements for TTS: Single-speaker vs multi-speaker recordings, sampling rates (22.05kHz / 24kHz), transcription alignment, silence trimming.
2. LLM Architectures, Tokenization & Adaptation:
   - Tokenization analysis: Byte-Pair Encoding (BPE), SentencePiece, byte-fallback. Subword fertility on Dagbani text across standard tokenizers (LLaMA-3, Mistral, Gemma, GPT-4) vs custom Dagbani BPE tokenizers. Vocabulary expansion techniques.
   - Continual Pre-training & Domain Adaptation: Data filtering, synthetic data generation, bilingual curriculum (Dagbani-English), perplexity benchmarks.
   - Instruction Fine-Tuning & Alignment: LoRA/QLoRA recipes, prompt engineering for Dagbani reasoning/translation, culture-specific QA.
3. Comprehensive Datasets & Benchmarks Catalog:
   - Complete inventory of all known Dagbani datasets across text and speech: Mozilla Common Voice (Dagbani), Dagbani Bible audio/text, Wikimedia/Wikipedia Dagbani dump, LoresLM Dagbani text corpora, Drum history oral literature recordings, News and radio broadcasts.
   - Metadata: Hours of audio, token counts, licenses, formats, train/val/test split recommendations, and quality hygiene steps.

Output your comprehensive findings and specification report to:
`d:/ATS Tech/Dagbani AI/.agents/explorer_tts_llm/survey_tts_llm_report.md`
Also create `handoff.md` and update `progress.md` in your folder. Send a completion message back when done.
