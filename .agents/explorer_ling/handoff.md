# Handoff Report: Dagbani Linguistics & Phonology Specification Mining

**Author**: `explorer_ling` (Linguistics & Phonology Specification Miner)  
**Recipient**: `parent` (Project Orchestrator, ID: `c9f60174-8731-412c-b5ff-fbaf09343052`)  
**Date**: 2026-08-14T21:05:00Z  
**Type**: Hard Handoff (Task Complete)

---

## 1. Observation

Directly extracted and analyzed from project resources:
- `resources/Thesis Submitted Revised 29 June PDF 2.pdf`: 359 pages. Chapter 2 ("The Phonological and Grammatical Systems of Dagbani", pp. 22–82) and Chapter 7 ("Influence of Dagbani Phonological Features on Dagomba's Spoken English", pp. 239–289) provide authoritative analyses of Dagbani phonology, consonants, vowels, vowel harmony, syllable structure, and metrical tense.
  - Section 2.2.1: Consonant inventory and allophony: velar stops `/k, ɡ/` become `[tʃ, dʒ]` before front vowels; `/s, z/` become `[ʃ, ʒ]`; `/kp, gb, ŋm/` become labial-coronal `[tp, db, nm]` before front vowels (Hudu 2010:14).
  - Debuccalization: `/s/` debuccalizes to `[h]` intervocalically when non-clustered (e.g. *biisi* $\rightarrow$ [bíhí]), while clustered `/s/` (*duunsi*) is blocked.
  - Section 2.3: 11 phonemic vowels (6 short /i, e, ɨ, a, o, u/, 5 long /iː, eː, aː, oː, uː/), with [±ATR] pairs and resolution of the high front [-ATR] controversy (Hudu 2016).
  - Section 2.3.1: Vowel harmony rules, height-conditioned trigger requirements, and opaque blockers `{l, s, r}` preventing progressive [+ATR] spread from root `/i/` (e.g. *pílɨ́*).
  - Section 2.4.4: Metrical time-depth particles: *dí* (hodiernal/today), *sá* (hesternal/yesterday), *dáá* (pre-hesternal/remote).
- `resources/Grandmasters_of_the_drum_Special_Issue_6.pdf`: Section 2.6 (pp. 51–53) provides Table 2.1 (Consonants) and Table 2.2 (Vowels), documenting standard orthography and digraphs (`kp, gb, ŋm, ny, ch, sh`).
- `research/Building Dagbani Language Models.pdf` (pp. 1–3) & `research/Building Dagbani Language Model and TTS.pdf` (pp. 1–4): Document 2-level tone system (High, Low, Downstep !H), tonal downstep terracing, moraic coda nasals (`sɔ́ŋ`), and noun tonal melodies (`H-H, H-H-H, H-L, L-H, L-L-H`).
- `resources/35_breaking_the_low_resource_barr.pdf` (GhanaNLP, ICLR 2023): Documents the 1998 Bureau of Ghana Languages (BGL) orthography (`ɛ, ɔ, ŋ, ɣ, ʒ`, digraphs `kp, gb, ŋm, ny, ch`) and tone minimal pairs (*gballi* [H-H] 'grave' vs *gballi* [H-L] 'zana mat').

---

## 2. Logic Chain

1. **Phoneme Inventory Extraction**: Cross-referencing Shaibu (2020), Hudu (2010, 2016), Olawsky (1999), and Alhassan (2021) establishes a definitive 27+ consonant system and an 11-vowel system (6 short, 5 long).
2. **ATR Harmony Modeling**: The debate over high front [-ATR] was resolved by Hudu (2016), showing that Dagbani has no surface `[ɪ]` (merges into `[ɨ]`), and that harmony operates under strict height constraints with alveolar blockers `{l, s, r}`.
3. **Suprasegmentals & Prosody**: Dagbani requires moraic analysis because syllable-final nasals (`/m, n, ŋ/`) act as active Tone-Bearing Units. Pitch declination follows automatic downstep terracing.
4. **Orthography Normalization**: Standard BGL text omits tone and uses extended Latin characters (`ɛ, ɔ, ŋ, ɣ, ʒ`), whereas informal and legacy data uses ASCII fallbacks (`e, o, ng, gh, zh`). G2P and tokenization systems must support bidirectional conversion.
5. **Morphophonological G2P Pipeline**: Rules for nasal prefix assimilation (`/N-/`), intervocalic lenition (`/d/ -> [r]`, `/g/ -> [ɣ]/[ʔ]`, `/s/ -> [h]`), cluster epenthesis (`[ɨ]/[ʊ]`), and hiatus elision (`$V_1\#V_2 \rightarrow V_2$`) form a deterministic pipeline ready for implementation.

---

## 3. Caveats

- **Tone in Text Corpora**: Standard digital corpora (e.g. Wikipedia, Common Voice transcriptions) lack tone markings. ASR models can train on tone-agnostic text, but high-quality TTS requires a tone-annotated phonetic lexicon or neural prosody prediction.
- **Dialectal Variation**: Tomosili (Western, Tamale) and Nayahili (Eastern, Yendi) exhibit subtle differences in root vowel ATR specifications and loanword liquid realizations (`r` vs `l`/`d`).

---

## 4. Conclusion

The linguistic, phonological, and orthographic specifications for Dagbani are fully mined, verified, and synthesized. All requirements have been consolidated into `d:/ATS Tech/Dagbani AI/.agents/explorer_ling/survey_linguistics_report.md`. This artifact provides complete reference specifications to construct:
1. The `dagbani-linguistics` Antigravity skill (G2P, ATR harmony checker, tokenizer).
2. The `dagbani-asr-whisper` skill (text normalizers and acoustic token mappings).
3. The `dagbani-tts-synthesis` skill (phonetic dictionary, tone annotations, moraic duration).
4. The `dagbani-llm-tokenization-datasets` skill (vocabulary retention for digraphs and extended characters).
5. The master knowledge document `knowledge/dagbani_phonology_orthography_guide.md`.

---

## 5. Verification Method

- **Specification Report File**: Inspect `d:/ATS Tech/Dagbani AI/.agents/explorer_ling/survey_linguistics_report.md`.
- **Text Extraction Files**: Inspect `.agents/explorer_ling/thesis_ch2_phonology.txt`, `building_dagbani_lm.txt`, `building_dagbani_lm_tts.txt`, and `grandmasters_snippet.txt`.
- **Validation Script**: Run Python test scripts in `.agents/explorer_ling/` verifying regex rules for BGL-ASCII transliteration, phoneme lookup tables, and ATR vowel harmony validation logic.
