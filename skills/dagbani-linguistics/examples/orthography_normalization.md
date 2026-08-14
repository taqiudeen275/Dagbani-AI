# Dagbani Orthography Normalization & Text Sanitization Examples

**Document Version**: 1.0.0  
**Domain**: Corpus Cleaning, NLP Preprocessing, ASR Transcription Sanitization  

---

## 1. ASCII to Canonical BGL Transliteration

Non-standard keyboard input commonly substitutes ASCII approximations for Dagbani letters. The normalizer deterministically recovers standard BGL characters.

### Example 1: Basic ASCII Transliteration
- **Input (Raw ASCII)**: `nyela paga mini bihi ban be Tamale`
- **Detected Issues**:
  - `nyela` $\rightarrow$ contains `e` for `ɛ` (*nyɛla* 'is/are').
  - `paga` $\rightarrow$ contains `g` for velar fricative `ɣ` (*paɣa* 'woman/wife').
  - `be` $\rightarrow$ contains `e` for `ɛ` (*bɛ* 'they/are in').
- **Normalized Output (BGL Standard)**: `nyɛla paɣa mini bihi ban bɛ Tamale`

---

### Example 2: Digraph & Fricative Transliteration
- **Input (Raw ASCII)**: `O yeliya: "N zhangmi gbana mini pagaba lahabali"`
- **Detected Issues**:
  - `zhangmi` $\rightarrow$ contains `zh` for `ʒ` and `ngm` for `ŋm` (*ʒaŋmi* / *ʒaŋmi*).
  - Curly quotes (`"`) require canonicalization.
  - `pagaba` $\rightarrow$ *paɣaba* ('women').
- **Normalized Output (BGL Standard)**: `O yeliya: 'N ʒaŋmi gbana mini paɣaba lahabali'`

---

## 2. Noisy Web & Social Media Text Sanitization

### Example 3: Mixed Punctuation, Diacritics & Spacing
- **Input (Noisy Social Media Post)**:
  ```text
  Ti'   kpalinzhoo   n-nyɛ   din   vieli   pammm...   Bɛ`   doo    ghana   la   !!
  ```
- **Cleaning Transformations**:
  1. Standardize apostrophes: Convert backticks (`` ` ``) and curly apostrophes (`’`, `‘`) to straight standard apostrophe (`'`).
  2. Collapse repeated characters: Normalize `pammm` $\rightarrow$ `pam` ('very much').
  3. Transliterate digraphs: `ghana` $\rightarrow$ `ɣana` (*gbana* / *ɣana*); `vieli` $\rightarrow$ `viɛli`.
  4. Fix whitespace and multi-punctuation: Collapse `...` and `!!` into standard terminal punctuation, collapse multiple spaces into single space.
- **Normalized Output**: `Ti kpalinʒoo n-nyɛ din viɛli pam. Bɛ doo ɣana la!`

---

## 3. Punctuation & Unicode Normalization Table

| Input String (Noisy / Mixed) | Normalization Operations Applied | Clean Output String |
| :--- | :--- | :--- |
| `“Wuntang’a” puli ni` | Convert curly quotes $\to$ `'`, `ng'` before vowel $\to$ `ŋ` | `'Wuntaŋa' puli ni` |
| `Zhaŋmi  bɛ'ʊ    din   viɛli` | Normalize `Zh` $\to$ `Ʒ`, strip multiple whitespace | `Ʒaŋmi bɛ'ʊ din viɛli` |
| `M`ba   bɔri   karimba` | Fix apostrophe, transliterate `karimba` | `M'ba bɔri karimba` |
| `kɔbga   mini    kɔwa` | Whitespace normalization, dialect recognition | `kɔbga mini kɔwa` |
| `Naa   Gbewaa   yili   zuliya` | Canonical NFC Unicode composition | `Naa Gbewaa yili zuliya` |

---

## 4. Dialectal Variant Normalization

The normalizer includes an optional dialect standardizer that maps eastern variants (*Nayahili*) to the standard Western educational standard (*Tomosili*), or vice versa:

| Feature | Western Variant (*Tomosili* / Standard) | Eastern Variant (*Nayahili*) | Semantic Gloss |
| :--- | :--- | :--- | :--- |
| **Vowel Quality** | *yɛltɔɣa* | *yɛltoɣa* | word / speech / matter |
| **Coalescence** | *kɔwa* | *kɔbga* | hundred |
| **Fricativization**| *tɔxɨ* | *tɔɣsɨ* | to speak / converse |
| **Epenthesis** | *chali* | *kpalim* | to remain / stay |
