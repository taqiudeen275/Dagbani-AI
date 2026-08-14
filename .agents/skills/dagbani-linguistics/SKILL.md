---
name: dagbani-linguistics
description: Comprehensive linguistic toolkit, phonology engine, G2P converter, syllable parser, and orthography normalizer for the Dagbani (Dagbanli) language.
---

# Dagbani Linguistics Skill

The `dagbani-linguistics` skill provides an authoritative, production-grade computational toolkit and knowledge base for the Dagbani language (*Dagbanli*, *Dagbane*, ISO 639-3: `dag`). It encompasses the complete 1998 Bureau of Ghana Languages (BGL) orthography, advanced tongue root ([±ATR]) vowel harmony engine, consonant lenition/palatalization rules, register tone modeling, syllable parsing, and Grapheme-to-Phoneme (G2P) conversion.

---

## 1. Capabilities & Features

1. **Deterministic Grapheme-to-Phoneme (G2P) Engine (`scripts/dagbani_g2p.py`)**:
   - Converts standard 1998 BGL orthography and ASCII transliterations into International Phonetic Alphabet (IPA) representations.
   - Accurately models context-sensitive phonological processes:
     - Digraph recognition (`kp`, `gb`, `ŋm`, `ny`, `ch`, `sh`).
     - Velar & alveolar palatalization before front vowels (`k, g, s, z, ŋ` $\rightarrow$ `[t͡ʃ, d͡ʒ, ʃ, ʒ, ɲ]`).
     - Intervocalic alveolar stop lenition (`d` $\rightarrow$ `[r]`).
     - Intervocalic fricative debuccalization (`s` $\rightarrow$ `[h]`).
     - Labial-coronal realization of labial-velars before front vowels (`kp, gb, ŋm` $\rightarrow$ `[t͡p, d͡b, n͡m]`).
     - Epenthetic weak vowel insertion (`[ɨ]`, `[ʊ]`) for illegal consonant clusters.
     - Tone assignment (High `´`, Low `` ` ``, Downstep `!H`).

2. **Orthography Normalizer & Sanitizer (`scripts/orthography_normalizer.py`)**:
   - Bidirectional converter between standard 1998 BGL Unicode (`ɛ, ɔ, ŋ, ɣ, ʒ`) and ASCII fallback transliterations (`e/E, o/O, ng/N, gh/G, zh/Z`).
   - Unicode NFC canonicalization, curly quote sanitization, and whitespace/punctuation hygiene.
   - Spell checking heuristics and dialectal lexical mapping (Western *Tomosili* vs Eastern *Nayahili*).

3. **Syllable Parser & Prosodic Weight Analyzer (`scripts/dagbani_syllabifier.py`)**:
   - Parses words into valid Dagbani syllable structures (`CV`, `CVC`, `CVV`, `CVVC`, `V`, `N`).
   - Identifies Tone-Bearing Units (TBUs) across vocalic nuclei and moraic coda nasals (`m, n, ŋ`).
   - Computes syllable weight (monomoraic vs bimoraic vs trimoraic) and phonotactic validity checks.

---

## 2. Directory Structure

```
skills/dagbani-linguistics/
├── SKILL.md                          # Main skill documentation and usage guide
├── references/
│   ├── phonology_matrix.md           # 27+ consonants, 11 vowels, ATR harmony, tone registers
│   ├── orthography_bgl.md            # 1998 BGL alphabet, Unicode table, ASCII conversion
│   └── morphophonology_rules.md      # Nasal assimilation, lenition, hiatus elision, sandhi
├── examples/
│   ├── g2p_conversion.md             # Worked G2P transformations for complex words & loans
│   └── orthography_normalization.md  # Raw noisy text cleanup & normalization examples
└── scripts/
    ├── dagbani_g2p.py                # Standalone & importable G2P conversion tool
    ├── orthography_normalizer.py     # Orthographic standardization & transliteration tool
    └── dagbani_syllabifier.py        # Syllable parser, mora counter & TBU identifier
```

---

## 3. Quick Start & CLI Usage

### 3.1 Grapheme-to-Phoneme (G2P) Conversion
```bash
# Convert a single phrase using BGL orthography
python skills/dagbani-linguistics/scripts/dagbani_g2p.py --text "Dagbamba biɛɣu viɛli"

# Convert ASCII transliterated text
python skills/dagbani-linguistics/scripts/dagbani_g2p.py --text "Dagbamba biegu vieli" --ascii-input

# Convert a text file line-by-line to IPA with tone
python skills/dagbani-linguistics/scripts/dagbani_g2p.py --input-file sentences.txt --output-file phonemes.txt --with-tones

# Run built-in self tests
python skills/dagbani-linguistics/scripts/dagbani_g2p.py --self-test
```

### 3.2 Orthography Normalization
```bash
# Convert ASCII text to canonical BGL orthography
python skills/dagbani-linguistics/scripts/orthography_normalizer.py --text "nyela paga mini dokita" --to-bgl

# Clean noisy punctuation, fix quotes, and ensure Unicode NFC
python skills/dagbani-linguistics/scripts/orthography_normalizer.py --text "O yeliya: 'Biɛɣu viɛli!'" --clean

# Batch normalize an entire corpus file
python skills/dagbani-linguistics/scripts/orthography_normalizer.py --input-file raw_corpus.txt --output-file clean_corpus.txt --clean --to-bgl
```

### 3.3 Syllabification & Mora Analysis
```bash
# Syllabify words and inspect onset-nucleus-coda breakdown
python skills/dagbani-linguistics/scripts/dagbani_syllabifier.py --text "bɛ'ʊ́ dɔ́rtí gballi"

# Analyze moraic weight and tone-bearing units (TBUs)
python skills/dagbani-linguistics/scripts/dagbani_syllabifier.py --text "sɔ́ŋ duunsi" --detailed
```

---

## 4. Python API Usage

```python
from skills.dagbani_linguistics.scripts.dagbani_g2p import DagbaniG2P
from skills.dagbani_linguistics.scripts.orthography_normalizer import DagbaniNormalizer
from skills.dagbani_linguistics.scripts.dagbani_syllabifier import DagbaniSyllabifier

# 1. Normalize Orthography
normalizer = DagbaniNormalizer()
clean_text = normalizer.normalize_to_bgl("pagaba mini bihi ban be Tamale")
# -> "paɣaba mini bihi ban be Tamale"

# 2. Convert to IPA Phonemes
g2p = DagbaniG2P()
ipa_output = g2p.convert_sentence("paɣaba mini bihi")
# -> "[pá.ɣá.bá mí.ní bí.hí]"

# 3. Syllabify and extract TBUs
syllabifier = DagbaniSyllabifier()
syllables = syllabifier.syllabify_word("gballi")
# -> [Syllable(onset='ɡ͡b', nucleus='a', coda='l', tone='H', moras=2),
#     Syllable(onset='l', nucleus='i', coda='', tone='H', moras=1)]
```

---

## 5. Verification & Quality Standards

- **Phonological Correctness**: Every rule is calibrated against peer-reviewed linguistic field documentation (Olawsky 1999, Hudu 2010/2016, Shaibu 2020).
- **Zero External Dependencies**: All core logic runs on standard Python 3.8+ (`unicodedata`, `re`, `argparse`, `dataclasses`).
- **Comprehensive Test Coverage**: Each script contains a `--self-test` suite covering digraphs, lenition, vowel harmony, epenthesis, and edge cases.
