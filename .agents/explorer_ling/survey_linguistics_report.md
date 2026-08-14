# Comprehensive Dagbani Linguistics, Phonology, and Orthography Specification Report

**Document ID**: `SPEC-DAG-LING-001`  
**Author**: Linguistics & Phonology Specification Miner (`explorer_ling`)  
**Target Projects**: `dagbani-linguistics`, `dagbani-asr-whisper`, `dagbani-tts-synthesis`, `dagbani-llm-tokenization-datasets`, `knowledge/dagbani_phonology_orthography_guide.md`  
**Authoritative Sources**:
1. *Dagbani English: The Influence of Dagbani on the Use of English in Ghana* (PhD Dissertation, Dr. Memunatu Shaibu, University of Bayreuth, 2020).
2. *Grandmasters of the Drum: A Literary Linguistic Analysis of the Dagbamba Panegyrics* (Special Issue 6, June 2021).
3. *Computational Strategies for the Digitization of Oral-Centric Gur Languages: A Comprehensive Framework for Dagbani Language Modeling and Synthetic Speech* (2026).
4. *Architecting Speech-Centric Large Language Models for Oral-Predominant Low-Resource Languages: A Strategic and Technical Blueprint for Dagbani* (2026).
5. *Breaking the Low-Resource Barrier for Dagbani ASR: From Data Collection to ASR Modeling* (GhanaNLP, ICLR 2023).
6. *Aspects of Dagbani Grammar* (Olawsky 1999) & *A Phonetic Inquiry into Dagbani Vowel Neutralisations* (Hudu 2010, 2016).
7. *Bureau of Ghana Languages (BGL) Standard Orthography Committee Report* (1996/1998).

---

## 1. Executive Summary & Linguistic Affiliation

- **Language Name**: Dagbani (autonyms: *Dagbanli*, *Dagbane*, *Dagbaŋ*)
- **People / Indigenes**: *Dagbamba* (singular: *Dagbana*)
- **Geographic Area**: Northern Region of Ghana (capitals: Tamale, Yendi, Bimbila) and border areas of northern Togo.
- **Genetic Classification**: Niger-Congo $\rightarrow$ Atlantic-Congo $\rightarrow$ Volta-Congo $\rightarrow$ North Volta-Congo $\rightarrow$ Central Gur $\rightarrow$ Oti-Volta $\rightarrow$ Western Oti-Volta $\rightarrow$ Southeast (Mabia subgroup).
- **ISO 639-3 Code**: `dag`
- **Mutually Intelligible / Sister Languages**: Mampruli, Gurene (Frafra), Kusaal, Talensi, Dagaare, Moore (Mossi), Wali.
- **Dialectal Landscape**:
  1. **Tòmòsílí (Western Dialect)**: Centered around Tamale (major economic/administrative city). Regarded as the standard for educational curricula, media broadcast, and literary orthography. Exhibits strong Advanced Tongue Root (+ATR) assimilation and distinctive consonant lenitions.
  2. **Nàyàhílí (Eastern Dialect)**: Centered around Yendi (traditional royal seat of the *Yaa Naa* / King of Dagbon). Displays retracted tongue root (-ATR) stability in specific noun bases and distinct root vowel preservation.
  3. **Nànùndí**: The variety spoken by the Nanumba indigenes in and around Bimbila.

---

## 2. Complete Dagbani Phoneme Inventory

### 2.1 Consonants Matrix

Dagbani exhibits 27+ consonantal segments, characterized by labial-velar stops, post-alveolar affricates, voiced/voiceless fricatives, palatal nasals, and context-dependent lenition/debuccalization.

| IPA Symbol | Orthography (BGL 1998) | ASCII / Alternate | Place of Articulation | Manner of Articulation | Voicing | Example Word | English Gloss | Phonotactic Distribution / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **/p/** | `<p>` | `p` | Bilabial | Plosive | Voiceless | *pìɛ́ɣú* [pjɛ́ɣú] | 'basket' | Word-initial, intervocalic; never in coda. |
| **/b/** | `<b>` | `b` | Bilabial | Plosive | Voiced | *bíá* [bíá] | 'child' | Initial, intervocalic; labialized before round vowels. |
| **/t/** | `<t>` | `t` | Alveolar | Plosive | Voiceless | *tìá* [tìá] | 'tree' | Word-initial, intervocalic; never coda. |
| **/d/** | `<d>` | `d` | Alveolar | Plosive | Voiced | *dór-ó* [dóró] | 'illness' | Word-initial; alternates with [r] intervocalically. |
| **/k/** | `<k>` | `k` | Velar | Plosive | Voiceless | *kòbígá* [kɔ̀bɨ̀ɡá] | 'hundred' | Palatalizes to [t͡ʃ] before front vowels; labialized [kʷ] before round. |
| **/ɡ/** | `<g>` | `g` | Velar | Plosive | Voiced | *gáafárà* [ɡáfárà] | 'excuse me' | Palatalizes to [d͡ʒ] before front vowels; lenites to [ɣ]/[ʔ] intervocalically. |
| **/k͡p/** | `<kp>` | `kp` | Labial-velar | Plosive (Co-articulated) | Voiceless | *kpɛ́* [k͡pɛ́] | 'to enter' | Realized as labial-coronal [t͡p] before front vowels. |
| **/ɡ͡b/** | `<gb>` | `gb` | Labial-velar | Plosive (Co-articulated) | Voiced | *gbállí* [ɡ͡bálːí] | 'grave/tomb' | Realized as labial-coronal [d͡b] before front vowels. |
| **/ʔ/** | `<'>` | `'` | Glottal | Plosive / Stop | Voiceless | *bɛ́'ʊ́* [bɛ́ʔʊ́] | 'bad / ugly' | Debuccalized stop intervocalically and post-vocalically. |
| **/t͡ʃ/** | `<ch>` | `ch`, `c`, `ky` | Post-alveolar | Affricate | Voiceless | *chán-dí* [t͡ʃándí] | 'going' | Historical / surface palatalization of /k/ before front vowels. |
| **/d͡ʒ/** | `<j>` | `j`, `gy`, `dzh` | Post-alveolar | Affricate | Voiced | *jɛ̀rígú* [d͡ʒɛ̀ríɡú] | 'fool' | Surface palatalization of /ɡ/ before front vowels; also lexical. |
| **/f/** | `<f>` | `f` | Labiodental | Fricative | Voiceless | *fɔ́ŋ* [fɔ́ŋ] | 'neighborhood' | Word-initial, intervocalic; never in coda. |
| **/v/** | `<v>` | `v` | Labiodental | Fricative | Voiced | *vìɛ́lì* [vjɛ̀lì] | 'beauty' | Word-initial, intervocalic. |
| **/s/** | `<s>` | `s` | Alveolar | Fricative | Voiceless | *sùá* [sùá] | 'knife / machete' | Palatalizes to [ʃ] before front vowels; debuccalizes to [h] intervocalically. |
| **/z/** | `<z>` | `z` | Alveolar | Fricative | Voiced | *zúŋɔ́* [zúŋɔ́] | 'today' | Palatalizes to [ʒ] before front vowels. |
| **/ʃ/** | `<sh>` | `sh`, `ʃ` | Post-alveolar | Fricative | Voiceless | *shɛ́lí* [ʃɛ́lí] | 'something' | Surface variant of /s/ before front vowels; phonemic in loanwords. |
| **/ʒ/** | `<ʒ>` | `zh`, `ʒ`, `j` | Post-alveolar | Fricative | Voiced | *ʒɛ́'ʊ́* [ʒɛ́ʔʊ́] | 'storm' | Surface variant of /z/ before front vowels; prohibited before back vowels. |
| **/ɣ/** | `<ɣ>` | `gh`, `ɣ` | Velar | Fricative | Voiced | *bíɛ̀ɣú* [bjɛ̀ɣú] | 'badness' | Lenited allophone of /ɡ/ intervocalically; written `<ɣ>` in BGL. |
| **/x/** | `<x>` | `kh`, `x` | Velar | Fricative | Voiceless | *xálí* [xálí] | 'character / habit' | Occurs in loanwords (Arabic/Hausa) and specific dialects. |
| **/h/** | `<h>` | `h` | Glottal | Fricative | Voiceless | *bíhí* [bíhí] | 'children' | Intervocalic debuccalization of /s/; loans (*Hàrúnà*). |
| **/m/** | `<m>` | `m` | Bilabial | Nasal | Voiced | *m̀-bɔ́* [m̀bɔ́] | 'I want' | Onset and moraic tone-bearing coda; syllabic prefix. |
| **/n/** | `<n>` | `n` | Alveolar | Nasal | Voiced | *nîn-kúrúgú* [nînkúrúɡú] | 'elderly person' | Onset and moraic tone-bearing coda; syllabic prefix. |
| **/ɲ/** | `<ny>` | `ny`, `ñ`, `ɲ` | Palatal | Nasal | Voiced | *nyɛ́bígá* [ɲɛ́bíɡá] | 'crocodile' | Onset before front and low vowels; palatalization of /ŋ/. |
| **/ŋ/** | `<ŋ>` | `ng`, `ŋ` | Velar | Nasal | Voiced | *ŋmɛ́* [ŋ͡mɛ́] / *sɔ́ŋ* [sɔ́ŋ] | 'knock' / 'mat' | Written `<ŋ>` in BGL; moraic coda; palatalizes to [ɲ] before front vowels. |
| **/ŋ͡m/** | `<ŋm>` | `ngm`, `ŋm` | Labial-velar | Nasal (Co-articulated) | Voiced | *ŋmáŋá* [ŋ͡máŋá] | 'calabash / self' | Becomes labial-coronal [n͡m] before front vowels. |
| **/l/** | `<l>` | `l` | Alveolar | Lateral Approximant | Voiced | *lá* [lá] | 'focus particle' | Onset and intervocalic; blocks root-to-suffix ATR vowel harmony. |
| **/r/** | `<r>` | `r` | Alveolar | Tap / Trill | Voiced | *bíndírá* [bíndírá] | 'food (pl.)' | Intervocalic allophone of /d/; initial in loanwords (*Ràbì* $\rightarrow$ *Labi*). |
| **/j/** | `<y>` | `y`, `j` | Palatal | Approximant / Glide | Voiced | *yɛ́m* [jɛ́m] | 'sense / wisdom' | Onset glide; inserted in hiatus resolution. |
| **/w/** | `<w>` | `w` | Labial-velar | Approximant / Glide | Voiced | *wúlá* [wúlá] | 'how / why' | Labial glide; does NOT undergo front-vowel palatalization. |

---

### 2.2 Vowel Inventory & Advanced Tongue Root (ATR) Classification

Dagbani contains **11 phonemic vowels** distributed into **6 short vowels** (/i, e, ɨ, a, o, u/) and **5 long vowels** (/iː, eː, aː, oː, uː/), accompanied by allophonic and dialectal [±ATR] expansions.

```
       FRONT                      CENTRAL                     BACK
                  [+ATR]   [-ATR]            [+ATR]   [-ATR]
HIGH:      /i/ [i]           /ɨ/ [ɨ, ɘ, ə]   /u/ [u]   [ʊ]
MID-HIGH:  /e/ [e]                           /o/ [o]
MID-LOW:           /ɛ/ [ɛ]                             /ɔ/ [ɔ]
LOW:                         /a/ [a, a̘]
```

#### Vowel Feature & ATR Mapping Table

| Vowel Grapheme (BGL) | IPA Phoneme | ATR Status | Height | Backness | Length | Allophonic / Surface Realization | Example Word | English Gloss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<i>` | **/i/** | **+ATR** | High | Front | Short | `[i]` | *bíá* | 'child' |
| `<ii>` | **/iː/** | **+ATR** | High | Front | Long | `[iː]` | *díí* | 'suddenly / just' |
| `<i>` (weak) / `<b>` | **/ɨ/** | **-ATR** | High/Central | Central | Short | `[ɨ]`, `[ɘ]`, `[ə]` | *bɨ́ndírígù* | 'food' |
| `<u>` | **/u/** | **+ATR** | High | Back | Short | `[u]` | *búá* | 'goat' |
| `<uu>` | **/uː/** | **+ATR** | High | Back | Long | `[uː]` | *dúú* | 'room' |
| `<u>` / `<o>` | **[ʊ]** | **-ATR** | High | Back | Short | `[ʊ]` | *zɛ́'ʊ́* [ʒɛ́ʔʊ́] | 'storm' |
| `<e>` | **/e/** | **+ATR** | Mid | Front | Short | `[e]` | *kore* [ko-re] | 'desire / appetite' |
| `<ee>` | **/eː/** | **+ATR** | Mid | Front | Long | `[eː]` | *yéé* | 'tune / tone' |
| `<ɛ>` / `<e>` | **/ɛ/** | **-ATR** | Mid-Low | Front | Short | `[ɛ]` | *kpɛ́* [k͡pɛ́] | 'enter' |
| `<ɛɛ>` / `<ee>`| **/ɛː/** | **-ATR** | Mid-Low | Front | Long | `[ɛː]` | *pɛ̀ɛ́* [pɛ̀ː] | 'sheep' |
| `<o>` | **/o/** | **+ATR** | Mid | Back | Short | `[o]` | *dóró* [dóró] | 'illness' |
| `<oo>` | **/oː/** | **+ATR** | Mid | Back | Long | `[oː]` | *dóó* [dóː] | 'man / male' |
| `<ɔ>` / `<o>` | **/ɔ/** | **-ATR** | Mid-Low | Back | Short | `[ɔ]` | *sɔ́ŋ* [sɔ́ŋ] | 'grass mat' |
| `<ɔɔ>` / `<oo>`| **/ɔː/** | **-ATR** | Mid-Low | Back | Long | `[ɔː]` | *kɔ̀ɔ́* [kɔ̀ː] | 'to farm / weed' |
| `<a>` | **/a/** | **-ATR** | Low | Central | Short | `[a]` (or `[a̘]` in +ATR context) | *bá* [bá] | 'father' |
| `<aa>` | **/aː/** | **-ATR** | Low | Central | Long | `[aː]` | *dáá* [dáː] | 'market' |

---

### 2.3 ATR Vowel Harmony Mechanics

Dagbani exhibits a sophisticated **Advanced Tongue Root ([±ATR]) harmony system** governed by strict directional and segmental constraints:

1. **The ATR Sets**:
   - **[+ATR] Harmonic Set**: `{ /i/, /e/, /o/, /u/, [a̘] }`
   - **[-ATR] Harmonic Set**: `{ /ɨ/, /ɛ/, /ɔ/, [ʊ], /a/ }`
2. **Directionality & Triggers**:
   - **Suffix-to-Root Harmony (Regressive)**: A [+ATR] suffix vowel (`-o`, `-e`) triggers [+ATR] raising of the root vowel:
     - Root `/dɔr/` + Suffix `/-o/` $\rightarrow$ **[dóró]** ('illness', singular) vs Plural `/dɔr/` + `/-tɨ/` $\rightarrow$ **[dɔ́rtɨ́]** ('illnesses').
     - Root `/kɔr/` + Suffix `/-e/` $\rightarrow$ **[kore]** ('appetite') vs Plural `/-ɨsɨ/` $\rightarrow$ **[kɔ́rɨ́sɨ́]**.
   - **Root-to-Suffix Harmony (Progressive)**: A [+ATR] root vowel (`/i/`, `/u/`) triggers [+ATR] realization of suffix vowels:
     - Root `/dir/` + Suffix `/-gʊ/` $\rightarrow$ **[dirigu]** ('food/utensil') vs Plural `/-tɨ/` $\rightarrow$ **[dɨ́rɨ́tɨ́]**.
3. **Height-Conditioned Harmony Constraint**:
   - Trigger and target vowels **must agree in height** (`[+high]` triggers `[+high]`; `[-high]` triggers `[-high]`) for harmony propagation to occur across morphs.
4. **Opaque Consonant Harmony Blockers**:
   - The alveolar coronal consonants **`{ /l/, /s/, /r/ }`** act as opaque barriers that strictly block progressive root-to-suffix [+ATR] spread from a root `/i/`:
     - `/pil/` + `/-i/` $\rightarrow$ **[pílɨ́]** (*not* `*[pili]`, 'to start/cover').
     - `/jir/` + `/-gi/` $\rightarrow$ **[jírɨ́gɨ́]** (*not* `*[jirigi]`, 'get startled').
     - `/jin/` + `/-si/` $\rightarrow$ **[jínsɨ́]** (*not* `*[jinsi]`, 'houses').
5. **Domain Boundary Constraints**:
   - Vowel harmony is strictly confined within the **lexical word domain** (root + affixes).
   - Harmony **never crosses compound boundaries or syntactic word boundaries**:
     - `[[bi]rt.] [[bɛ-ʔʊ]suf.]` $\rightarrow$ **[bí bɛ́ʔʊ́]** ('ugly child', *never* `*[bí béʔú]`).
6. **Disharmonic Loanwords**:
   - Unassimilated borrowings from English, Arabic, and Hausa violate native ATR harmony constraints (e.g. *sukuuru* 'school', *aliba* 'wages').

---

### 2.4 Diphthongs and Co-occurring Vowels

1. **Pure Opening Diphthongs**:
   - **/ia/**: *bía* [bíá] ('child'), *shía* [ʃíá] ('bee/spirit'), *tìa* [tìá] ('tree').
   - **/ua/**: *búa* [búá] ('goat'), *tùa* [tùá] ('baobab tree'), *sùa* [sùá] ('knife/machete').
2. **Closing Diphthongs (Vowel + Palatal Glide /j/)**:
   - **/aai/**, **/eei/**, **/ooi/**, **/uui/**: *baai* ('to stretch'), *kooi* ('to drain').
3. **Palatalized Secondary Nuclei `[iɛ]`**:
   - Arises phonologically via palatalization of front vowel `/ɛ/` following consonants:
     - *biɛɣu* [bjɛ́ɣú] ('ugly / bad / morning'), *piɛɣu* [pjɛ́ɣú] ('basket'), *viɛli* [vjɛ́lí] ('beauty'), *shiɛɣu* [ʃjɛ́ɣú] ('rainy season').
4. **Triphthongs**: Strictly **prohibited** in Dagbani phonotactics.

---

## 3. Suprasegmentals & Tone System

### 3.1 Tone Levels and Prosodic Units

Dagbani is a **register tone language** operating on **two level tones** augmented by **phonemic and automatic downstep**:

1. **High Tone (H / ´)**: High pitch register ($F_0$ peak).
2. **Low Tone (L / `)**: Low pitch register ($F_0$ baseline).
3. **Downstep (!H)**: Lowering of a High tone following an overt or floating Low tone ($H \rightarrow !H$).
4. **Tone-Bearing Unit (TBU)**:
   - Every vowel mora ($V = 1\text{ TBU}$, $VV = 2\text{ TBUs}$).
   - Syllable-final nasals (`/m, n, ŋ/`) in coda position function as **active, moraic TBUs** bearing contrastive pitch:
     - *sɔ́ŋ* [sɔ́ŋ́] ('mat', H coda nasal).
     - *ǹ-dá* [ǹ-dá] ('I bought', L syllabic nasal prefix).

### 3.2 Tone Terracing & Pitch Downdrift

In connected speech, Dagbani exhibits **automatic downstep (tone terracing)**:
$$\text{In a sequence } H_1 - L_1 - H_2 - L_2 - H_3\dots, \quad \text{Pitch}(H_2) < \text{Pitch}(H_1), \quad \text{Pitch}(H_3) < \text{Pitch}(H_2)$$
This creates a stepwise downward staircase profile across the intonational phrase.

### 3.3 Lexical Minimal Tone Pairs

| Lexical Form | Tonal Melody | Phonetic Realization | English Meaning |
| :--- | :--- | :--- | :--- |
| *gballi* | **H-H** | `[ɡ͡bálːí]` | 'grave / tomb' |
| *gballi* | **H-L** | `[ɡ͡bálːì]` | 'zana mat (woven grass mat)' |
| *duu* | **H-H** | `[dúú]` | 'room' |
| *duu* | **L-L** | `[dùù]` | 'pig' |
| *tia* | **H-H** | `[tíá]` | 'tree' |
| *tia* | **L-L** | `[tìà]` | 'companion / peer' |
| *kari* | **H-H** | `[kárí]` | 'to chase away / drive out' |
| *kari* | **L-L** | `[kàrì]` | 'to read / count' |

### 3.4 Grammatical Tone Functions

1. **Tense/Aspect/Mood (TAM) Marking**:
   - Verb roots adopt overriding tonal melodies based on TAM prefixes and suffixes:
     - Perfective: Low tone root melody + high suffix (e.g. *chàŋ-yá*).
     - Imperfective: High tone root melody (e.g. *chán-dí*).
2. **Noun Class Pluralization**:
   - Plural suffixes induce tone shifts on nominal roots (e.g. *dá-á* [H-H] $\rightarrow$ *dá-hí* [H-H]; *dó-ó* [H-H] $\rightarrow$ *dɔ́r-tí* [H-H]).
3. **Noun Phrase Leftward Tone Spreading**:
   - 92% of nominal melodies conform to `{H-H, H-H-H, H-L, L-H, L-L-H}`. High tone spreads leftward from right phrase boundaries.
4. **Negation Polarity**:
   - Negative particles (*kù* 'future negative', *bì* 'past negative', *dī* 'prohibitive') enforce tone polarity shifts on the succeeding verbal nucleus.

### 3.5 Syllable Structure Constraints

| Syllable Template | Frequency | Examples | Structural Rules & Constraints |
| :--- | :--- | :--- | :--- |
| **CV** | Most Common | *bá* ('father'), *kó* ('only') | Single onset C required; /ɨ/ and /ʊ/ cannot occur in open CV words (*kɨ). |
| **CVC** | Frequent | *yɛ́m* ('wisdom'), *sɔ́ŋ* ('mat') | Coda C **strictly restricted** to nasals {m, n, ŋ} and glottal stop [ʔ]. |
| **CVV** | Frequent | *dáá* ('market'), *búa* ('goat') | Long vowels or opening diphthongs; bimoraic (2 TBUs). |
| **CVVC** | Common | *dúún-sí* ('rooms') | Long vowel + nasal coda; trimoraic prosodic weight. |
| **V** | Restricted | *ó* ('he/she'), *á* ('you') | Restricted to pronouns and word-initial position; never medial/final. |
| **N** (Syllabic) | Frequent | *ǹ-* ('1SG'), *ḿ-* ('1SG before labial') | Homorganic nasal prefix bearing tone. |
| **CCV** | Non-native | *kɨ̀láasì* ('class') | **Forbidden in native vocabulary**; broken by epenthetic [ɨ]/[ə]. |

---

## 4. Orthographic Systems

### 4.1 Historical vs Modern Orthographies

1. **Ajami Script (Pre-1960s)**:
   - Arabic script adapted with diacritics for Dagbani labial-velars and vowels (documented by Afa Yusif Ajura, 1959).
2. **Early Latin Standard (1968–1995)**:
   - Introduced by Blair & Tamakloe (1941) and GILLBT (1968); lacked uniform handling of open mid vowels and velar fricatives.
3. **1998 Bureau of Ghana Languages (BGL) Standard Orthography**:
   - The authoritative modern orthography established by the Dagbani Orthography Committee (1996) and published by BGL (1998).
   - Features **28 Latin characters + Digraphs**, incorporating 5 special IPA-derived extended glyphs: `ɛ`, `ɔ`, `ŋ`, `ɣ`, `ʒ`.

### 4.2 Orthographic Alphabet Mapping Table

| Letter / Digraph | Uppercase | Lowercase | Unicode Code Point | IPA Phoneme | ASCII Fallback | Pronunciation / Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | `A` | `a` | U+0041 / U+0061 | `/a/` | `a` | [a] |
| **B** | `B` | `b` | U+0042 / U+0062 | `/b/` | `b` | [b] |
| **CH** | `Ch` | `ch` | - | `/t͡ʃ/` | `ch`, `c` | [t͡ʃ] |
| **D** | `D` | `d` | U+0044 / U+0064 | `/d/` | `d` | [d] |
| **E** | `E` | `e` | U+0045 / U+0065 | `/e/` | `e` | [e] |
| **Ɛ** (Open E) | `Ɛ` | `ɛ` | **U+0190 / U+025B** | `/ɛ/` | `e`, `E` | [ɛ] |
| **F** | `F` | `f` | U+0046 / U+0066 | `/f/` | `f` | [f] |
| **G** | `G` | `g` | U+0047 / U+0067 | `/ɡ/` | `g` | [ɡ] |
| **GB** | `Gb` | `gb` | - | `/ɡ͡b/` | `gb` | [ɡ͡b] |
| **Ɣ** (Gamma) | `Ɣ` | `ɣ` | **U+0194 / U+0263** | `/ɣ/` | `gh`, `G` | [ɣ] |
| **H** | `H` | `h` | U+0048 / U+0068 | `/h/` | `h` | [h] |
| **I** | `I` | `i` | U+0049 / U+0069 | `/i/`, `/ɨ/` | `i` | [i], [ɨ] |
| **J** | `J` | `j` | U+004A / U+006A | `/d͡ʒ/` | `j` | [d͡ʒ] |
| **K** | `K` | `k` | U+004B / U+006B | `/k/` | `k` | [k] |
| **KP** | `Kp` | `kp` | - | `/k͡p/` | `kp` | [k͡p] |
| **L** | `L` | `l` | U+004C / U+006C | `/l/` | `l` | [l] |
| **M** | `M` | `m` | U+004D / U+006D | `/m/` | `m` | [m] |
| **N** | `N` | `n` | U+004E / U+006E | `/n/` | `n` | [n] |
| **NY** | `Ny` | `ny` | - | `/ɲ/` | `ny`, `ñ` | [ɲ] |
| **Ŋ** (Eng) | `Ŋ` | `ŋ` | **U+014A / U+014B** | `/ŋ/` | `ng`, `N` | [ŋ] |
| **ŊM** | `Ŋm` | `ŋm` | - | `/ŋ͡m/` | `ngm`, `ŋm` | [ŋ͡m] |
| **O** | `O` | `o` | U+004F / U+006F | `/o/` | `o` | [o] |
| **Ɔ** (Open O) | `Ɔ` | `ɔ` | **U+0186 / U+0254** | `/ɔ/` | `o`, `O` | [ɔ] |
| **P** | `P` | `p` | U+0050 / U+0070 | `/p/` | `p` | [p] |
| **R** | `R` | `r` | U+0052 / U+0072 | `[r]` | `r` | [r] |
| **S** | `S` | `s` | U+0053 / U+0073 | `/s/` | `s` | [s] |
| **SH** | `Sh` | `sh` | - | `/ʃ/` | `sh` | [ʃ] |
| **T** | `T` | `t` | U+0054 / U+0074 | `/t/` | `t` | [t] |
| **U** | `U` | `u` | U+0055 / U+0075 | `/u/` | `u` | [u] |
| **V** | `V` | `v` | U+0056 / U+0076 | `/v/` | `v` | [v] |
| **W** | `W` | `w` | U+0057 / U+0077 | `/w/` | `w` | [w] |
| **Y** | `Y` | `y` | U+0059 / U+0079 | `/j/` | `y` | [j] |
| **Z** | `Z` | `z` | U+005A / U+007A | `/z/` | `z` | [z] |
| **Ʒ** (Ezh) | `Ʒ` | `ʒ` | **U+01B7 / U+0292** | `/ʒ/` | `zh`, `Z` | [ʒ] |
| **'** (Apostrophe)| `'` | `'` | U+0027 | `/ʔ/` | `'` | Glottal / Elision |

### 4.3 Tone Marking Conventions in Written Text

1. **Standard Published Text (BGL, Wikipedia, Newspapers, Signboards)**:
   - **Tone is completely unmarked**. Readers resolve tonal ambiguities strictly via contextual semantic and syntactic cues.
2. **Linguistic & Educational Literature**:
   - Uses acute accent (`á, é, ɛ́, í, ó, ɔ́, ú, ń, ḿ`) for High tone.
   - Uses grave accent (`à, è, ɛ̀, ì, ò, ɔ̀, ù, ǹ, m̀`) for Low tone.
   - Uses circumflex (`â`) for High-Low falling contour.
   - Uses caron (`ǎ`) for Low-High rising contour.
   - Uses exclamation mark (`!H`) or downstep mark (`ꜜH`) for Downstep.

---

## 5. Grapheme-to-Phoneme (G2P) & Morphophonology

### 5.1 Exact G2P Transformation Pipeline

A deterministic G2P conversion engine for Dagbani must execute the following prioritized ordered phases:

```
Raw Text (BGL or ASCII) 
  │
  ▼ [Step 1: Text Normalization & Unicode Canonicalization (NFC)]
  ▼ [Step 2: ASCII-to-BGL Canonical Disambiguation (e.g. 'gh' -> 'ɣ', 'zh' -> 'ʒ', 'ngm' -> 'ŋm')]
  ▼ [Step 3: Multi-character Digraph Tokenization (<ŋm>, <kp>, <gb>, <ny>, <sh>, <ch>)]
  ▼ [Step 4: Contextual Allophonic Mutation (Palatalization & Labialization)]
  ▼ [Step 5: Intervocalic Lenition & Debuccalization (/d/ -> [r], /s/ -> [h], /g/ -> [ɣ]/[ʔ])]
  ▼ [Step 6: Epenthetic Weak Vowel Insertion between Illegal Clusters]
  ▼ [Step 7: Hiatus Resolution & Vowel Elision at Word Boundaries]
  ▼ [Step 8: Tonal Contour & Prosodic Assignment (Lexicon + Syntax)]
  │
  ▼
IPA Phonemic / Phonetic Output
```

#### Rule Execution Table

| Rule ID | Context / Pattern | Input Grapheme | Output IPA | Morphophonological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **G2P-01** | General | `<ŋm>` / `ngm` | `/ŋ͡m/` | Voiced labial-velar nasal. |
| **G2P-02** | General | `<kp>` | `/k͡p/` | Voiceless labial-velar plosive. |
| **G2P-03** | General | `<gb>` | `/ɡ͡b/` | Voiced labial-velar plosive. |
| **G2P-04** | General | `<ny>` | `/ɲ/` | Voiced palatal nasal. |
| **G2P-05** | General | `<ch>` | `/t͡ʃ/` | Voiceless post-alveolar affricate. |
| **G2P-06** | General | `<j>` | `/d͡ʒ/` | Voiced post-alveolar affricate. |
| **G2P-07** | General | `<sh>` | `/ʃ/` | Voiceless post-alveolar fricative. |
| **G2P-08** | General | `<ʒ>` / `zh` | `/ʒ/` | Voiced post-alveolar fricative. |
| **G2P-09** | General | `<ɣ>` / `gh` | `/ɣ/` | Voiced velar fricative. |
| **G2P-10** | General | `<ŋ>` / `ng` | `/ŋ/` | Voiced velar nasal. |
| **G2P-11** | General | `<ɛ>` | `/ɛ/` | Open-mid front unrounded vowel. |
| **G2P-12** | General | `<ɔ>` | `/ɔ/` | Open-mid back rounded vowel. |
| **G2P-13** | Before front `{i, e, ɛ}` | `<k>` | `[t͡ʃ]` | Contextual velar palatalization. |
| **G2P-14** | Before front `{i, e, ɛ}` | `<g>` | `[d͡ʒ]` | Contextual voiced velar palatalization. |
| **G2P-15** | Before front `{i, e, ɛ}` | `<s>` | `[ʃ]` | Contextual alveolar palatalization. |
| **G2P-16** | Before front `{i, e, ɛ}` | `<z>` | `[ʒ]` | Contextual voiced alveolar palatalization. |
| **G2P-17** | Before front `{i, e, ɛ}` | `<ŋ>` | `[ɲ]` | Contextual velar nasal palatalization. |
| **G2P-18** | Before front `{i, e, ɛ}` | `<kp, gb, ŋm>` | `[t͡p, d͡b, n͡m]` | Labial-coronal realization before front vowels. |
| **G2P-19** | Intervocalic `V_V` | `<d>` | `[r]` | Alveolar stop lenition to tap/trill. |
| **G2P-20** | Intervocalic `V_V` | `<s>` (non-geminate) | `[h]` | Intervocalic debuccalization (e.g. *biisi* $\rightarrow$ [bíhí]). |
| **G2P-21** | Intervocalic `V_V` | `<g>` | `[ɣ]` / `[ʔ]` | Velar spirantization and glottal reduction. |
| **G2P-22** | CC Cluster Environment | `C_C` | `[ɨ]` / `[ə]` | Epenthetic weak vowel insertion (e.g. *kɔbga* $\rightarrow$ [kɔ̀bɨ̀ɡá]). |

---

### 5.2 Morphophonological Rules

#### 1. Nasal Prefix Homorganic Assimilation
Prefix nasal `/N-/` (1SG subject pronoun, nominalizer, or class prefix) assimilates in place of articulation to the following consonant:
$$/N-/ + C \rightarrow \begin{cases} [m] & \text{before } /p, b, f, v/ \quad (\text{e.g. } \text{/N-bɔ/ } \rightarrow \text{ [m̀-bɔ́]}) \\ [n] & \text{before } /t, d, s, z, l/ \quad (\text{e.g. } \text{/N-da/ } \rightarrow \text{ [ǹ-dá]}) \\ [\ɲ] & \text{before } /t͡ʃ, d͡ʒ, ʃ, ʒ, j/ \quad (\text{e.g. } \text{/N-chaŋ/ } \rightarrow \text{ [ɲ̀-t͡ʃáŋ]}) \\ [\ŋ] & \text{before } /k, ɡ, ɣ/ \quad (\text{e.g. } \text{/N-ka/ } \rightarrow \text{ [ŋ̀-ká]}) \\ [\ŋ͡m] & \text{before } /k͡p, ɡ͡b/ \quad (\text{e.g. } \text{/N-kpa/ } \rightarrow \text{ [ŋ͡m̀-k͡pá]})\end{cases}$$

#### 2. Intervocalic Lenition & Debuccalization
- **/d/ $\rightarrow$ [r]**: *kad-a* $\rightarrow$ [kárá] ('chase away', plural).
- **/g/ $\rightarrow$ [ɣ] / [ʔ]**: *bɔg-u* $\rightarrow$ [bɔ̀ɣú] ('hole / pit'), *bɛ-g-u* $\rightarrow$ [bɛ́ʔʊ́] ('bad / morning').
- **/s/ $\rightarrow$ [h]**: *bii-si* $\rightarrow$ [bíhí] ('children'), *daa-si* $\rightarrow$ [dáhí] ('markets'). *Blocked in consonant clusters*: *duun-si* $\rightarrow$ [dúúnsí] (*never* `*[dúúnhí]`).

#### 3. Vowel Elision & Hiatus Resolution
When a vowel-final word precedes a vowel-initial word in rapid speech, the final vowel of $W_1$ elides:
$$V_1 \# V_2 \rightarrow \emptyset \# V_2 \quad (\text{written with apostrophe: } \text{'tí ó'} \rightarrow \text{'t\'ó'} \text{ [t-ó]})$$

#### 4. Consonant Gemination in Class Nominal Suffixation
Suffixation of noun class markers (e.g. Class 2 `-li/-la`) induces gemination of root-final sonorant consonants:
- *gbal* + *-li* $\rightarrow$ **[ɡ͡bálːí]** ('grave')
- *bil* + *-li* $\rightarrow$ **[bílːí]** ('seedling')

---

### 5.3 Loanword Adaptation Mechanics (English, Arabic, Hausa)

Dagbani has historically adapted loanwords through regularized phonotactic repair strategies:

#### A. English Loanword Phonological Adaptation
1. **Cluster Repair via Epenthesis**:
   - Initial clusters $sC-$ and $CL-$ are broken by inserting epenthetic weak central vowel `[ɨ]` or `[ʊ]`:
     - *class* $\rightarrow$ **kɨ̀láasì** [kɨ̀láːsì]
     - *block* $\rightarrow$ **bʊ̀lókú** [bʊ̀lókú]
     - *clock* $\rightarrow$ **kʊ̀lɔ́kù** [kʊ̀lɔ́kʊ̀]
     - *school* $\rightarrow$ **sùkúùrù** [sùkúùrù]
2. **Coda Consonant Resolution**:
   - English non-nasal codas receive final epenthetic vowels:
     - *cup* $\rightarrow$ **kɔ́pùl / kɔ́pʊ̀**
     - *doctor* $\rightarrow$ **dɔ́kɨ̀tà**
     - *book* $\rightarrow$ **búkù**
3. **Consonant Substitution**:
   - English /p/ in non-initial codas often labializes or spirantizes.
   - English alveolar /r/ in initial positions becomes /l/ (e.g. *radio* $\rightarrow$ *ledio* / *redio*).

#### B. Arabic & Hausa Loanword Adaptation
Islam has had a profound linguistic impact on Dagbon since the 17th century (via Hausa and Wangara traders).
1. **Phonetic Liquid Alternation**:
   - Arabic names with initial /r/ were traditionally nativized with alveolar lateral /l/ or stop /d/:
     - *Rabi* $\rightarrow$ **Labi**
     - *Rukaya* $\rightarrow$ **Lukaya**
     - *Rubaba* $\rightarrow$ **Lubaba**
     - *Rahaman* $\rightarrow$ **Dahimani**
     - *Rafia* $\rightarrow$ **Lafia**
     - *Haruna* $\rightarrow$ **Aduna**
2. **Cultural & Religious Vocabulary Adaptation**:
   - *al-baraka* (Arabic) $\rightarrow$ **alibaari / albarika** ('blessings')
   - *as-subh* (Arabic) $\rightarrow$ **asiba** ('morning')
   - *al-khabar* (Arabic) $\rightarrow$ **lahabali** ('news / information / story')
   - *ad-du'a* (Arabic) $\rightarrow$ **aduwa** ('prayer')
   - *al-ahd* (Arabic) $\rightarrow$ **alkawle** ('covenant / promise')
   - *shinkafa* (Hausa) $\rightarrow$ **shinkaafa** ('rice')
   - *kudi* (Hausa) $\rightarrow$ **laɣifu / liɣiri** ('money')

---

## 6. Syntactic and Grammatical Blueprint

### 6.1 Canonical Sentence Template

Dagbani is strictly **Subject-Verb-Object (SVO)**. The canonical declarative clause adheres to a 6-slot linear template:

$$\text{[Slot 1: Subject]} \rightarrow \text{[Slot 2: Time-Depth Marker]} \rightarrow \text{[Slot 3: Tense/Aspect]} \rightarrow \text{[Slot 4: Verb Base]} \rightarrow \text{[Slot 5: Discourse/Focus Particle]} \rightarrow \text{[Slot 6: Object/Adverbial]}$$

| Clause Slot | Syntactic Function | Morpheme Examples | Description |
| :--- | :--- | :--- | :--- |
| **Slot 1** | Subject / Agent | *ó* (3SG), *tí* (1PL), *bɛ̀* (3PL), *nádòo* ('chief') | Pronoun or nominal phrase. |
| **Slot 2** | Time-Depth Marker (TDM) | *dí* (hodiernal), *sá* (hesternal), *dáá* (pre-hesternal) | Metrical tense / remoteness marker. |
| **Slot 3** | Tense / Aspect Modifier | *yɛ̀n* (future), *yáá* (habitual), *shírí* (affirmative), *ná* (progressive) | Pre-verbal aspectual particles. |
| **Slot 4** | Verb Base | *chàŋ* ('go'), *dī* ('eat'), *nyɛ̄* ('see'), *bɔ́hì* ('ask') | Perfective or Imperfective root. |
| **Slot 5** | Discourse / Focus Particle | *mì* (declarative/affirmative), *lá* (focus), *mí* (emphatic) | Post-verbal pragmatic marker. |
| **Slot 6** | Object / Adverbial | *shìkúrú* ('school'), *zúŋɔ́* ('today'), *sɔ̀hálá* ('yesterday') | Patient, theme, or temporal adjunct. |

### 6.2 Metrical Tense & Degree of Remoteness System

Dagbani possesses one of the most sophisticated **metrical tense (time-depth)** systems in the Gur family (Bodomo 2001, Shaibu 2020):

| Time Depth Category | Marker | Temporal Scope (Past) | Temporal Scope (Future) | Example Sentence | Gloss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hodiernal** | **dí** / **díín** | Today (hours/minutes ago) | Near future (today / soon) | *Ń dí chàŋ shìkúrú zúŋɔ́.* | 'I went to school earlier today.' |
| **Hesternal** | **sá** / **sán** | Yesterday (1 day ago) | Tomorrow (1 day future) | *Ń sá chàŋ shìkúrú.* | 'I went to school yesterday.' |
| **Pre-Hesternal** | **dáá** / **dáán** | 2+ days ago (remote past) | 2+ days ahead (remote future)| *Ń dáá chàŋ shìkúrú.* | 'I went to school two or more days ago.' |

---

## 7. Features Discovered Table

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Consonant Inventory | Labial-Velar Plosives & Nasals | Phonemes /k͡p, ɡ͡b, ŋ͡m/ acting as atomic consonants. | Text `<kp, gb, ŋm>` | IPA `[k͡p, ɡ͡b, ŋ͡m]` | Splitting into discrete k+p, g+b creates incorrect phonemes. | Thesis Ch 2, Grandmasters Table 2.1 |
| 2 | Consonant Inventory | Labial-Coronal Mutation | /k͡p, ɡ͡b, ŋ͡m/ realize as [t͡p, d͡b, n͡m] before front vowels. | /k͡p/ + /ɛ/ | `[t͡pɛ́]` | Treating as velar before front vowel fails acoustic match. | Hudu (2010), Thesis Ch 2 |
| 3 | Consonant Inventory | Velar Palatalization | /k, ɡ, ŋ/ shift to [t͡ʃ, d͡ʒ, ɲ] before front vowels. | `<k>` + `<ɛ>` (*kɛhi*) | `[t͡ʃɛ́hɨ́]` | Retaining [k] creates non-native pronunciation. | Olawsky (1999), Thesis Ch 2 |
| 4 | Consonant Inventory | Alveolar Palatalization | /s, z/ shift to [ʃ, ʒ] before front vowels. | `<s>` + `<i>` (*si-a*) | `[ʃí-â]` | Retaining [s] before front vowel violates phonotactics. | Hudu (2010), Thesis Ch 2 |
| 5 | Consonant Inventory | Intervocalic Lenition | /d/ lenites to [r] intervocalically in native stems. | /kad-a/ | `[kárá]` | Failing to lenite leads to unnatural stilted speech. | Olawsky (1999), Grandmasters p. 52 |
| 6 | Consonant Inventory | Intervocalic Debuccalization | /s/ debuccalizes to [h] intervocalically when non-clustered. | /bii-si/ | `[bíhí]` | Overgeneralizing to clusters (*duunsi -> *duunhi) errors. | Hudu (2010), Thesis Ch 2 |
| 7 | Vowel Inventory | 11 Phonemic Vowels | 6 short (/i, e, ɨ, a, o, u/) and 5 long (/iː, eː, aː, oː, uː/). | Vowel graphemes | IPA Phonemes | Treating [ɨ] as /i/ or /e/ distorts vowel harmony. | Hudu (2016), Thesis Ch 2 |
| 8 | Vowel Harmony | Bidirectional ATR Harmony | Suffix-to-root and root-to-suffix [+ATR] vowel raising. | Root + Suffix morphs | Harmonized word | Generating disharmonic native roots fails fluency. | Hudu (2010, 2016), Thesis Ch 2 |
| 9 | Vowel Harmony | Opaque Consonant Blocking | Alveolars {l, s, r} block root-to-suffix [+ATR] spreading. | /pil-i/ | `[pílɨ́]` (*not* `[pili]`) | Spreading across {l, s, r} produces illicit forms. | Hudu (2010), Thesis Ch 2 |
| 10 | Vowel Harmony | Domain Boundary Barrier | Harmony strictly blocked across compound/word boundaries. | `bi` + `bɛ'ʊ` | `[bí bɛ́ʔʊ́]` | Cross-word harmony propagation corrupts adjacent words. | Hudu (2010), Thesis Ch 2 |
| 11 | Suprasegmentals | 2-Level Register Tone + Downstep | Contrastive High (H), Low (L), and Downstep (!H). | Syllable TBUs | Tone pitch melody | Unmarked text causes ASR/TTS semantic ambiguity. | GhanaNLP (2023), LM Doc p. 1-3 |
| 12 | Suprasegmentals | Moraic Nasal Tone-Bearing Units | Syllable-final nasals (/m, n, ŋ/) bear independent tone. | *sɔ́ŋ* [sɔ́ŋ́] | Moraic TBU | Treating coda nasals as non-moraic distorts rhythm. | LM Doc p. 2, Thesis Ch 2 |
| 13 | Suprasegmentals | Grammatical Tone Assignment | TAM and noun plurals override lexical root tone melodies. | Verb + TAM marker | Overriding tonal melody | Retaining citation lexical tone in inflected verbs errors. | Shaibu (2020), LM Doc p. 3 |
| 14 | Orthography | BGL 1998 Standard Graphemes | Letters `ɛ`, `ɔ`, `ŋ`, `ɣ`, `ʒ` and digraphs `kp, gb, ŋm, ny, ch, sh`. | BGL Unicode text | Standard Dagbani | Stripping special characters collapses phonemic contrasts. | BGL (1998), GhanaNLP (2023) |
| 15 | Orthography | ASCII Transliteration Equivalence | Deterministic bi-directional mapping between BGL and ASCII. | `gh, zh, ng, e, o` | `ɣ, ʒ, ŋ, ɛ, ɔ` | Ambiguity in 'e' (could be /e/ or /ɛ/) requires lexicon. | GhanaNLP (2023), Grandmasters |
| 16 | Morphophonology | Homorganic Nasal Assimilation | Prefix /N-/ assimilates to point of articulation of onset C. | /N-/ + /p, b, k, t/ | `[m-, n-, ŋ-, ŋ͡m-]` | Fixed /n/ prefix causes unnatural phonetics. | Thesis Ch 2, Olawsky (1999) |
| 17 | Morphophonology | Hiatus Vowel Elision | Word-final vowel elides before vowel-initial word ($V_1\#V_2 \rightarrow V_2$). | *ti o* | *t'o* `[t-ó]` | Maintaining hiatus produces unnatural glottal stops. | Grandmasters p. 53, Thesis Ch 2 |
| 18 | Morphophonology | Loanword Epenthesis Repair | Illegal English clusters/codas repaired with epenthetic [ɨ]/[ʊ]. | *class*, *block* | *kɨlaasi*, *buloku* | Preserving complex onsets produces non-native speech. | Olawsky (1999), Thesis Ch 7 |
| 19 | Morphophonology | Arabic/Hausa Liquid Nativity | Arabic initial /r/ historically nativized to /l/ or /d/. | *Rabi*, *Rahaman* | *Labi*, *Dahimani* | Rigid modern spelling may clash with oral pronunciation. | Inusah (2019), Thesis Ch 2 |
| 20 | Syntax / Semantics | Metrical Time-Depth Particles | Preverbal particles *dí* (today), *sá* (yesterday), *dáá* (remote). | TDM + Verb Base | Remoteness distinction | Confusing TDM markers distorts temporal truth conditions. | Bodomo (2001), Thesis Ch 2 |

---

## 8. Edge Cases and Observed Linguistic Behaviors

| # | Feature | Input / Context | Observed Behavior & Analysis |
| :--- | :--- | :--- | :--- |
| 1 | Debuccalization Block in Clusters | `/duun-si/` (rooms) vs `/bii-si/` (children) | Intervocalic /s/ debuccalizes to [h] in *bíhí*, but is **strictly blocked** in *dúúnsí* because /s/ follows a nasal coda cluster. Generating `*[duunhi]` is ungrammatical. |
| 2 | Opaque Harmony Blocking with /l, s, r/ | Root `/pil/` (+ATR trigger /i/) + Suffix `/-i/` | Despite root /i/ being [+ATR], the intervening lateral /l/ blocks harmony spread, causing suffix to surface as [-ATR] `[ɨ]` (*pílɨ́*, *never* `*[pili]`). |
| 3 | Labial-Velar Fronting Mutation | `/k͡p/` + `/ɛ/` (*kpɛ́* 'enter') | The labial-velar closure shifts forward to a labial-coronal co-articulation `[t͡pɛ́]`. Similarly, `/ɡ͡b/` + `/ɛ/` $\rightarrow$ `[d͡bɛ́]`. |
| 4 | Velar Neutralization with /a/ | Velar plosive /k/ before low vowel /a/ vs front /ɛ/ | While /k/ palatalizes to [t͡ʃ] before /i, e, ɛ/, before /a/ it remains velar [ka]. Mid vowels without secondary onset articulation completely neutralize with [a]. |
| 5 | Minimal Tone Pair Disambiguation | Text token `<gballi>` without tone diacritics | Orthographically identical token represents two distinct lexemes: High-High `[ɡ͡bálːí]` ('grave') vs High-Low `[ɡ͡bálːì]` ('zana grass mat'). ASR/TTS must use syntactic context. |
| 6 | Moraic Coda Nasal Tone Pitch | Syllable-final nasal in *sɔ́ŋ* ('mat') vs *chaŋ* ('go') | Coda nasals hold a distinct tone mora; in *sɔ́ŋ*, the nasal carries High tone, lengthening the acoustic syllable duration by ~40-60ms relative to non-moraic codas. |
| 7 | English s-Cluster Epenthesis | English loan *school* $\rightarrow$ Dagbani *sùkúùrù* | Dagbani inserts epenthetic /u/ to break initial /sk-/ and epenthetic /u/ to resolve final /-l/, adding tone melodies to match native prosody. |
| 8 | Intervocalic /d/ vs Initial /d/ | Root *dóó* ('man') vs Inflected *kád-á* ('chase away') | Initial /d/ remains an alveolar plosive [dóó], whereas intervocalic /d/ obligatorily lenites to alveolar tap/trill [kárá]. |
| 9 | Compound Boundary ATR Resistance | Compound `[bí] + [bɛ́'ʊ́]` ('ugly child') | [+ATR] root `bi` does NOT spread harmony to adjacent root `bɛ́'ʊ́`. The word boundary acts as a hard harmonic wall (*bí bɛ́'ʊ́*, not `*bí bé'ú`). |
| 10 | Glottal Stop Hiatus vs Lenited Stop | Word boundary *ti' o* ('give him') vs Root *bɛ'ʊ* ('bad') | In *ti' o*, the apostrophe represents elision of final /i/; in *bɛ'ʊ*, the apostrophe represents a phonetic glottal stop [ʔ] derived from underlying /ɡ/. |

---

## 9. Downstream Implementation Specifications

### 9.1 For `dagbani-linguistics` Skill
- Embed full phoneme and vowel matrices as Python lookup tables (`PHONEMES`, `VOWELS_ATR`, `ALLOPHONES`).
- Implement deterministic G2P rules incorporating palatalization, debuccalization, lenition, and nasal assimilation.
- Implement bidirectional ATR vowel harmony validation engine with opaque consonant blocker logic (`l, s, r`).

### 9.2 For `dagbani-asr-whisper` Skill
- Construct normalizers that unify BGL and ASCII variations (`ɛ/e`, `ɔ/o`, `ŋ/ng`, `ɣ/gh`, `ʒ/zh`).
- Handle tone-agnostic transcription evaluation while retaining phonetic precision.

### 9.3 For `dagbani-tts-synthesis` Skill
- Implement phonetic lexicon dictionary with tone pitch annotations (`H`, `L`, `!H`).
- Implement duration models respecting bimoraic vowels and moraic coda nasals.

### 9.4 For `dagbani-llm-tokenization-datasets` Skill
- Optimize BPE / WordPiece tokenizers to treat Dagbani digraphs (`kp, gb, ŋm, ny, ch, sh`) and special characters (`ɛ, ɔ, ŋ, ɣ, ʒ`) as atomic or coherent units without subword fragmentation.

---
*Report compiled and certified by explorer_ling on 2026-08-14.*
