# BRIEFING — 2026-08-14T21:15:00Z

## Mission
Adversarially challenge and stress-test Dagbani linguistics, G2P, orthography normalization, syllabification, tokenization, and dataset cleaning tooling.

## 🔒 My Identity
- Archetype: challenger
- Roles: critic, specialist
- Working directory: d:/ATS Tech/Dagbani AI/.agents/challenger_1/
- Original parent: c9f60174-8731-412c-b5ff-fbaf09343052
- Milestone: M4 (Verification, Testing & Audit)
- Instance: 1 of 2

## 🔒 Key Constraints
- Review and test execution only — do NOT modify production implementation code directly in `skills/` or `knowledge/` unless instructed; identify and report bugs, failure modes, counterexamples, and reproduction scripts.
- Every claim must be empirically verified by running code/tests.
- Write handoff report with 5 components and verdict: APPROVE or REQUEST_CHANGES.

## Current Parent
- Conversation ID: c9f60174-8731-412c-b5ff-fbaf09343052
- Updated: 2026-08-14T21:15:00Z

## Review Scope
- **Files to review & test**:
  1. `skills/dagbani-linguistics/scripts/dagbani_g2p.py`
  2. `skills/dagbani-linguistics/scripts/orthography_normalizer.py`
  3. `skills/dagbani-linguistics/scripts/dagbani_syllabifier.py`
  4. `skills/dagbani-llm-tokenization-datasets/scripts/train_dagbani_tokenizer.py`
  5. `skills/dagbani-llm-tokenization-datasets/scripts/dataset_cleaner.py`
- **Target Failure Modes**:
  - Vowel harmony edge cases: Disharmonic loanwords (*asibiti*, *sooja*), compounds with mixed ATR roots, opaque consonant blocking ({l, s, r}).
  - Orthography mutations: Mixed case with special glyphs (`Ŋ, ŋ, Ɛ, ɛ, Ɔ, ɔ, Ɣ, ɣ, Ʒ, ʒ`), messy ASCII transliterations (`ng, gh, zh, sh, ch, ny, kp, gb, nm`), curly/smart quotes vs apostrophes (`'` vs `’` vs `‘`).
  - Complex syllabification: Moraic coda nasals (*ŋ* in *Dagbaŋ*), vowel hiatus, vowel lengthening.
  - Extreme text cleaning: Corrupted Unicode, non-Latin scripts, zero-width spaces, huge inputs, empty strings.
  - Tokenizer training on synthetic corpus with Dagbani digraphs.

## Attack Surface
- **Hypotheses tested**: [TBD - will populate during test execution]
- **Vulnerabilities found**: [TBD]
- **Untested angles**: [TBD]

## Loaded Skills
- **Source**: `d:\ATS Tech\Dagbani AI\.agents\skills\dagbani-linguistics\SKILL.md`
  - **Core methodology**: Deterministic G2P, vowel harmony engine, consonant lenition/palatalization, tone registers, BGL orthography normalizer, syllable parser.
- **Source**: `d:\ATS Tech\Dagbani AI\.agents\skills\dagbani-llm-tokenization-datasets\SKILL.md`
  - **Core methodology**: Byte-level BPE tokenizer training preserving digraphs & BGL glyphs, text hygiene & deduplication pipelines, LoRA instruction tuning.

## Key Decisions Made
- Will create comprehensive Python adversarial test suites and execute them against the 5 target scripts.
- Will test edge cases systematically across phonology, morphology, orthography, regex edge cases, tokenization, text cleaning, exception handling, and performance under stress.

## Artifact Index
- `.agents/challenger_1/BRIEFING.md` — persistent memory and state
- `.agents/challenger_1/progress.md` — liveness heartbeat and step tracking
- `.agents/challenger_1/DISPATCH.md` — task dispatch log
- `.agents/challenger_1/handoff.md` — comprehensive final challenge report and verdict
