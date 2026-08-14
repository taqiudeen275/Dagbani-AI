# Dagbani Grapheme-to-Phoneme (G2P) & Phonemizer Pipeline

This reference document outlines the comprehensive Grapheme-to-Phoneme (G2P), tone-tier extraction, and prosodic annotation pipeline for Dagbani Text-to-Speech synthesis.

---

## 1. The G2P Front-End Architecture

```
Raw Orthographic Text (BGL Unicode or ASCII)
  │
  ├──► [1. Normalization & Canonicalization] (Unicode NFC, apostrophe standardization)
  ├──► [2. ASCII-to-BGL Fallback Disambiguation] (e.g., 'gh' -> 'ɣ', 'zh' -> 'ʒ', 'ngm' -> 'ŋm')
  ├──► [3. Digraph Extraction] (<ŋm>, <kp>, <gb>, <ny>, <sh>, <ch>)
  ├──► [4. Contextual Allophonic Mutation] (Palatalization before {i, e, ɛ}, Labialization before {o, u, ɔ})
  ├──► [5. Intervocalic Lenition & Debuccalization] (/d/ -> [r], /s/ -> [h], /g/ -> [ɣ]/[ʔ])
  ├──► [6. Vowel Harmony [±ATR] Verification] (Suffix-to-root & root-to-suffix)
  ├──► [7. Syllabification & Moraic Coda Parsing] (Identifies moraic coda nasals)
  └──► [8. Tone Diacritic Restoration (TDR)] (Assigns H [´], L [`], or !H tone tiers)
  │
  ▼
Phoneme Symbol Sequence + Tone IDs + Duration Flags
```

---

## 2. Phoneme Inventory & Mapping Table

### Consonants (IPA Symbols and Token IDs)

| Token ID | Symbol | Orthography (BGL) | ASCII Alternative | Description | Context Rules |
|---|---|---|---|---|---|
| 1 | `_` | ` ` | ` ` | Word Boundary / Space | Inter-word pause |
| 2 | `p` | `<p>` | `p` | Voiceless bilabial plosive | Never in coda |
| 3 | `b` | `<b>` | `b` | Voiced bilabial plosive | Labialized before /u, o, ɔ/ |
| 4 | `t` | `<t>` | `t` | Voiceless alveolar plosive | Never in coda |
| 5 | `d` | `<d>` | `d` | Voiced alveolar plosive | Becomes `[r]` intervocalically |
| 6 | `k` | `<k>` | `k` | Voiceless velar plosive | Palatalizes to `[t͡ʃ]` before `/i, e, ɛ/` |
| 7 | `g` | `<g>` | `g` | Voiced velar plosive | Palatalizes to `[d͡ʒ]` before `/i, e, ɛ/`; lenites to `[ɣ]/[ʔ]` intervocalically |
| 8 | `kp` | `<kp>` | `kp` | Voiceless labial-velar stop `/k͡p/` | Becomes `[t͡p]` before `/i, e, ɛ/` |
| 9 | `gb` | `<gb>` | `gb` | Voiced labial-velar stop `/ɡ͡b/` | Becomes `[d͡b]` before `/i, e, ɛ/` |
| 10 | `ʔ` | `<'>` | `'` | Glottal plosive | Occurs in broken syllables (*bɛ'ʊ*) |
| 11 | `ch` | `<ch>` | `ch`, `c` | Voiceless post-alveolar affricate `/t͡ʃ/` | |
| 12 | `j` | `<j>` | `j` | Voiced post-alveolar affricate `/d͡ʒ/` | |
| 13 | `f` | `<f>` | `f` | Voiceless labiodental fricative | |
| 14 | `v` | `<v>` | `v` | Voiced labiodental fricative | |
| 15 | `s` | `<s>` | `s` | Voiceless alveolar fricative | Palatalizes to `[ʃ]` before `/i, e, ɛ/`; debuccalizes to `[h]` intervocalically |
| 16 | `z` | `<z>` | `z` | Voiced alveolar fricative | Palatalizes to `[ʒ]` before `/i, e, ɛ/` |
| 17 | `sh` | `<sh>` | `sh` | Voiceless post-alveolar fricative `/ʃ/` | |
| 18 | `ʒ` | `<ʒ>` | `zh` | Voiced post-alveolar fricative `/ʒ/` | |
| 19 | `ɣ` | `<ɣ>` | `gh` | Voiced velar fricative `/ɣ/` | |
| 20 | `h` | `<h>` | `h` | Voiceless glottal fricative | |
| 21 | `m` | `<m>` | `m` | Bilabial nasal | Tone-bearing in coda & syllabic prefix |
| 22 | `n` | `<n>` | `n` | Alveolar nasal | Tone-bearing in coda & syllabic prefix |
| 23 | `ny` | `<ny>` | `ny` | Palatal nasal `/ɲ/` | |
| 24 | `ŋ` | `<ŋ>` | `ng` | Velar nasal `/ŋ/` | Tone-bearing in coda |
| 25 | `ŋm` | `<ŋm>` | `ngm` | Labial-velar nasal `/ŋ͡m/` | |
| 26 | `l` | `<l>` | `l` | Alveolar lateral approximant | Blocks progressive ATR harmony |
| 27 | `r` | `<r>` | `r` | Alveolar tap/trill `[ɾ]` | Intervocalic allophone of `/d/` |
| 28 | `y` | `<y>` | `y`, `j` | Palatal approximant `/j/` | |
| 29 | `w` | `<w>` | `w` | Labial-velar approximant `/w/` | |

---

### Vowels & Length Contrasts

| Token ID | Symbol | Phoneme | ATR Category | Length | Description |
|---|---|---|---|---|---|
| 30 | `a` | `/a/` | -ATR | Short | Open central unrounded |
| 31 | `a:` | `/aː/` | -ATR | Long (Bimoraic) | Long open central |
| 32 | `e` | `/e/` | +ATR | Short | Close-mid front unrounded |
| 33 | `e:` | `/eː/` | +ATR | Long (Bimoraic) | Long close-mid front |
| 34 | `ɛ` | `/ɛ/` | -ATR | Short | Open-mid front unrounded |
| 35 | `ɛ:` | `/ɛː/` | -ATR | Long (Bimoraic) | Long open-mid front |
| 36 | `i` | `/i/` | +ATR | Short | Close front unrounded |
| 37 | `i:` | `/iː/` | +ATR | Long (Bimoraic) | Long close front |
| 38 | `ɨ` | `/ɨ/` | -ATR | Short (Weak) | Close central unrounded |
| 39 | `o` | `/o/` | +ATR | Short | Close-mid back rounded |
| 40 | `o:` | `/oː/` | +ATR | Long (Bimoraic) | Long close-mid back |
| 41 | `ɔ` | `/ɔ/` | -ATR | Short | Open-mid back rounded |
| 42 | `ɔ:` | `/ɔː/` | -ATR | Long (Bimoraic) | Long open-mid back |
| 43 | `u` | `/u/` | +ATR | Short | Close back rounded |
| 44 | `u:` | `/uː/` | +ATR | Long (Bimoraic) | Long close back |

---

## 3. Tone Tier Encoding

Dagbani incorporates three discrete tone tier tokens:

| Tone ID | Symbol | Name | Acoustic Target ($F_0$) | Context / Trigger |
|---|---|---|---|---|
| 101 | `́` (H) | High Tone | Top 25% of pitch range | Primary stressed TBUs, TAM perfectives |
| 102 | `̀` (L) | Low Tone | Lower 30% of pitch range | Default citation baseline, hesternal particles |
| 103 | `!` (DH) | Downstep | High tone lowered by ~25 Hz | High tone following overt or floating Low tone |

### Tone-Bearing Units (TBU)
- **Short Vowels**: 1 TBU
- **Long Vowels & Diphthongs**: 2 TBUs (can hold contour tones, e.g., $HL$ `áà` or $LH$ `àá`)
- **Coda Nasals**: Syllable-final `/m, n, ŋ/` receive an explicit tone tier token.

---

## 4. Rule-Based Mutation Algorithms

```python
def apply_intervocalic_lenition(tokens: list[str]) -> list[str]:
    """
    Applies intervocalic /d/ -> [r] and /s/ -> [h] unless blocked by a cluster.
    """
    vowels = {"a", "a:", "e", "e:", "ɛ", "ɛ:", "i", "i:", "ɨ", "o", "o:", "ɔ", "ɔ:", "u", "u:"}
    output = list(tokens)
    for i in range(1, len(tokens) - 1):
        if tokens[i] == "d" and tokens[i-1] in vowels and tokens[i+1] in vowels:
            output[i] = "r"
        elif tokens[i] == "s" and tokens[i-1] in vowels and tokens[i+1] in vowels:
            # debuccalization to glottal fricative
            output[i] = "h"
    return output
```
