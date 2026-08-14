# Original User Request

## Initial Request — 2026-08-14T20:50:36Z

Deeply review, analyze, and synthesize all research papers, academic theses, notebooks, HTML strategies, and reference materials on the Dagbani language, building a comprehensive, modular suite of 4 production-ready Antigravity skills and an exhaustive knowledge base for Dagbani linguistics, speech recognition (ASR), speech synthesis (TTS), and large language models (LLMs).

Working directory: d:/ATS Tech/Dagbani AI
Integrity mode: demo

## Requirements

### R1. Deep Multi-Resource Analysis & Cross-Referencing
Perform an exhaustive, rigorous analysis across all 10 research artifacts and resources in `research/` and `resources/`:
1. `research/Building Dagbani Language Model and TTS.pdf`
2. `research/Building Dagbani Language Models.pdf`
3. `research/research_source.txt` (including linked Gemini deep research sessions)
4. `resources/2026.loreslm-1.54.pdf` (Low-resource LM methods & benchmarks)
5. `resources/35_breaking_the_low_resource_barr.pdf` (Techniques for low-resource languages)
6. `resources/APSIPA2025_P208.pdf` (Speech processing & acoustic modeling)
7. `resources/Grandmasters_of_the_drum_Special_Issue_6.pdf` (Acoustic, tonal, cultural, and drum language linguistic analysis)
8. `resources/Thesis Submitted Revised 29 June PDF 2.pdf` (Comprehensive academic thesis on Dagbani phonology, syntax, morphology, and computational linguistics)
9. `resources/dagbani_llm_tts_strategy.html` (Strategic architecture for Dagbani LLMs and TTS)
10. `resources/dagbani_whisper_asr_colab_v2.ipynb` (End-to-end Whisper fine-tuning pipeline, data preprocessing, and evaluation)

Extract every core concept, linguistic rule, phonetic inventory, orthographic convention, dataset specification, model architecture, training hyperparameter, evaluation metric, and failure mode.

### R2. Production-Grade Modular Antigravity Skills Generation
Generate 4 complete, modular Antigravity skills under both `.agents/skills/` (and `skills/`) in the project repository, each containing a compliant `SKILL.md` (with valid YAML frontmatter `name` and `description`), `references/`, `examples/`, and helper `scripts/`:
1. **`dagbani-linguistics`**: Complete Dagbani phonology (vowels, consonants, tone system, vowel harmony), orthography (standard alphabet, special characters like ɛ, ɔ, ɣ, ŋ, ʒ), grammar, morphology, syntax, and cultural/drum acoustic linguistics.
2. **`dagbani-asr-whisper`**: End-to-end Dagbani Automatic Speech Recognition, Whisper fine-tuning recipes, audio preprocessing (resampling, silence removal, normalization), dataset schemas, WER/CER evaluation, and Colab/GPU execution workflows.
3. **`dagbani-tts-synthesis`**: Text-to-Speech synthesis architecture for Dagbani (FastSpeech2, VITS, Coqui TTS, Tacotron2), phonemization/grapheme-to-phoneme (G2P) pipelines, tone modeling, voice dataset recording protocols, and audio quality assessment (MOS).
4. **`dagbani-llm-tokenization-datasets`**: Low-resource LLM adaptation for Dagbani, byte-pair encoding (BPE) / SentencePiece tokenization efficiency (fertility rates), pretraining and instruction fine-tuning datasets, synthetic data generation strategies, LoRA/QLoRA recipes, and bilingual English-Dagbani translation benchmarks.

### R3. Exhaustive Knowledge Base & Master Synthesis Reports
Create a structured knowledge base in `knowledge/` synthesizing all findings into permanent, deep-dive reference documents:
- `knowledge/dagbani_phonology_orthography_guide.md`
- `knowledge/dagbani_asr_whisper_playbook.md`
- `knowledge/dagbani_tts_acoustic_playbook.md`
- `knowledge/dagbani_llm_pretraining_finetuning_guide.md`
- `knowledge/dagbani_datasets_and_benchmarks_catalog.md`

## Acceptance Criteria

### Coverage & Analysis Rigor
- [ ] All 10 source documents across `research/` and `resources/` are thoroughly examined with zero unanalyzed documents.
- [ ] No placeholder text, high-level hand-waving, or generic descriptions; all linguistic phonemes, tone patterns, model hyperparameters, and tokenization considerations are documented with concrete data and examples.

### Antigravity Skill Compliance
- [ ] All 4 skills (`dagbani-linguistics`, `dagbani-asr-whisper`, `dagbani-tts-synthesis`, `dagbani-llm-tokenization-datasets`) are created with valid `SKILL.md` frontmatter (`name`, `description`).
- [ ] Each skill contains at least one substantive reference file in `references/`, realistic examples in `examples/`, and executable Python/shell scripts in `scripts/` (e.g., G2P converters, audio preprocessors, tokenizer testing scripts).
- [ ] Skills are positioned in `.agents/skills/` and mirrored in `skills/` for maximum portability.

### Knowledge Base & Documentation
- [ ] Complete set of 5 master synthesis documents created in `knowledge/`.
- [ ] Includes actionable code snippets, dataset formatting templates (JSONL / HuggingFace Dataset schemas), training loss curves / benchmark tables from the papers, and evaluation rubrics.
