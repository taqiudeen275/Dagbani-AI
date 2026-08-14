# Dagbani Grapheme-to-Phoneme (G2P) Conversion Examples

**Document Version**: 1.0.0  
**Scope**: Step-by-step transformations from Dagbani Orthography (BGL 1998 & ASCII) to IPA Phonemic & Phonetic Sequences.

---

## 1. Native Dagbani Lexical Items

### Example 1: `bɛ'ʊ́` ('badness / ugliness / morning')
- **Orthography**: `<bɛ'ʊ́>` (or ASCII `<be'u>`)
- **Step 1 (Tokenization)**: `['b', 'ɛ', '\'', 'ʊ́']`
- **Step 2 (Consonants)**:
  - `b` $\rightarrow$ `[b]` (voiced bilabial plosive)
  - `'` $\rightarrow$ `[ʔ]` (glottal stop)
- **Step 3 (Vowels & ATR)**:
  - `ɛ` $\rightarrow$ `[ɛ]` ([-ATR] open-mid front unrounded vowel)
  - `ʊ` $\rightarrow$ `[ʊ]` ([-ATR] high back rounded vowel, harmonizes with `ɛ`)
- **Step 4 (Tone)**: High tone `[´]` on both vocalic moras.
- **Output IPA**: **`[bɛ́.ʔʊ́]`**

---

### Example 2: `gballi` ('grave / tomb' vs 'zana grass mat')
- **Orthography**: `<gballi>` (or ASCII `<gballi>`)
- **Step 1 (Digraph Detection)**: `gb` identified as atomic labial-velar stop `/ɡ͡b/`.
- **Step 2 (Gemination)**: `-lli` represents geminate alveolar lateral `[lː]`.
- **Step 3 (Tone Context)**:
  - Case A (Tomb): High-High melody $\rightarrow$ **`[ɡ͡bálːí]`**
  - Case B (Grass Mat): High-Low melody $\rightarrow$ **`[ɡ͡bálːì]`**
- **Output IPA (Citation Tomb)**: **`[ɡ͡bálːí]`**

---

### Example 3: `biɛɣu` ('bad / day / tomorrow')
- **Orthography**: `<biɛɣu>` (or ASCII `<bieghu>`)
- **Step 1 (Digraphs & Special Glyphs)**: `biɛ` (palatalized front sequence) + `ɣ` (velar fricative).
- **Step 2 (Consonants)**:
  - `b` + `iɛ` $\rightarrow$ `[bjɛ]` (secondary palatal glide onset)
  - `ɣ` $\rightarrow$ `[ɣ]` (voiced velar fricative)
- **Step 3 (Vowels & ATR)**: `[ɛ]` ([-ATR]) and `[u]` (surfaces as `[ʊ]` in [-ATR] environment).
- **Output IPA**: **`[bjɛ́.ɣú]`**

---

### Example 4: `ŋmaŋa` ('calabash / self')
- **Orthography**: `<ŋmaŋa>` (or ASCII `<ngmanga>`)
- **Step 1 (Digraph Detection)**: `ŋm` $\rightarrow$ `/ŋ͡m/` (labial-velar nasal).
- **Step 2 (Medial Nasal)**: `ŋ` $\rightarrow$ `/ŋ/` (velar nasal onset).
- **Step 3 (Vowels)**: Low central vowel `[a]`.
- **Output IPA**: **`[ŋ͡má.ŋá]`**

---

### Example 5: `kpɛ` ('to enter')
- **Orthography**: `<kpɛ>` (or ASCII `<kpe>`)
- **Step 1 (Digraph & Front Vowel Mutation)**:
  - `kp` is a labial-velar stop `/k͡p/`.
  - Followed by front vowel `ɛ`, the closure shifts forward to labial-coronal `[t͡p]`.
- **Step 2 (Vowel)**: `[ɛ]` with High tone.
- **Output IPA**: **`[t͡pɛ́]`** (phonetic) / **`/k͡pɛ́/`** (phonemic)

---

## 2. Morphologically Complex & Inflected Words

### Example 6: `bihi` ('children' — plural of *bia*)
- **Underlying Form**: `/bii/` ('child') + `/-si/` (Class 1 plural suffix).
- **Phonological Derivation**:
  1. *bii* + *si* $\rightarrow$ `/biisi/`
  2. Intervocalic debuccalization rule: `/s/` between vowels $\rightarrow$ `[h]`.
  3. Resulting surface form: `[bíhí]`.
- **Output IPA**: **`[bí.hí]`**

---

### Example 7: `duunsi` ('rooms' — plural of *duu*)
- **Underlying Form**: `/duu/` ('room') + `/-n/` (nasal extender) + `/-si/` (plural).
- **Phonological Derivation**:
  1. `/duun-si/` contains a coda nasal before `/s/`.
  2. Debuccalization is **strictly blocked** by the consonant cluster constraint.
  3. Sibilant `[s]` is fully retained.
- **Output IPA**: **`[dúːn.sí]`**

---

### Example 8: `m-bɔra` ('I want / seeking')
- **Underlying Form**: `/N-/` (1SG prefix) + `/bɔ/` ('want') + `/-da/` (imperfective).
- **Phonological Derivation**:
  1. Nasal assimilation: `/N-/` before bilabial `/b/` $\rightarrow$ `[m̀-]`.
  2. Alveolar lenition: `/d/` between vowels in `/-da/` $\rightarrow$ `[r]`.
- **Output IPA**: **`[m̀.bɔ́.rá]`**

---

## 3. Adapted Loanwords (English & Arabic)

### Example 9: `shikuru` (from English *school*)
- **Source**: English `/skuːl/`
- **Dagbani Phonotactic Repair**:
  1. Palatalization: `/s/` before front/high vowels $\rightarrow$ `[ʃ]`.
  2. Cluster breaking epenthesis: `/i/` breaks initial `/sk-/` $\rightarrow$ `shi-ku-`.
  3. Liquid substitution & final vowel: coda `/l/` becomes `[r]` with epenthetic `[u]`.
- **Output IPA**: **`[ʃì.kúː.rù]`**

---

### Example 10: `lahabali` (from Arabic *al-khabar* via Hausa)
- **Source**: Arabic *al-khabar* ('news / story')
- **Dagbani Phonotactic Repair**:
  1. Reanalysis of article *al-* $\rightarrow$ initial onset `la-`.
  2. Fricative substitution: velar fricative `/x/` $\rightarrow$ glottal `[h]`.
  3. Suffixation to Class 2 nominal root: `-li`.
- **Output IPA**: **`[là.há.bá.lí]`**
