# Dagbani Morphophonological Rules & Sandhi Processes

**Document Version**: 1.0.0  
**Domain**: Computational Morphophonology & Phonotactics  
**Target Applications**: G2P Engine, Syllable Parser, Text-to-Speech (TTS), Language Model Tokenization  

---

## 1. Overview of Morphophonological Processes

Dagbani morphophonology is characterized by dynamic phonological mutations that occur at morpheme junctions (affixation, compounding, and cliticization) and across fluent phrase boundaries.

```
Underlying Representation (UR)
   │
   ├──► 1. Homorganic Nasal Prefix Assimilation
   ├──► 2. Contextual Palatalization & Labial-Coronal Mutation
   ├──► 3. Intervocalic Consonant Lenition & Debuccalization
   ├──► 4. Suffixal Consonant Gemination
   ├──► 5. Epenthetic Vowel Insertion for Illicit Clusters
   ├──► 6. Hiatus Resolution & Vowel Elision
   └──► 7. Register Tone Sandhi & Tense/Aspect Overrides
   │
Surface Phonetic Realization (SR)
```

---

## 2. Rule Specifications

### 2.1 Homorganic Nasal Prefix Assimilation (Rule MPH-01)
The 1st person singular pronoun prefix `/N-/` (also used in nominalizations and class markers) obligatorily assimilates in place of articulation to the following onset consonant:

$$/N-/ + C_{\text{onset}} \longrightarrow \begin{cases} 
[m-] & \text{before bilabials / labiodentals } \{p, b, f, v, m\} \\
[n-] & \text{before alveolars } \{t, d, s, z, l, r, n\} \\
[\ɲ-] & \text{before palatals / post-alveolars } \{t͡ʃ, d͡ʒ, ʃ, ʒ, j, \ɲ\} \\
[\ŋ-] & \text{before velars } \{k, ɡ, ɣ, x, \ŋ\} \\
[\ŋ͡m-] & \text{before labial-velars } \{k͡p, ɡ͡b, \ŋ͡m\}
\end{cases}$$

#### Examples:
- `/N/` + `/bɔ/` ('want') $\rightarrow$ **[m̀-bɔ́]** ('I want')
- `/N/` + `/da/` ('buy') $\rightarrow$ **[ǹ-dá]** ('I bought')
- `/N/` + `/t͡ʃaŋ/` ('go') $\rightarrow$ **[ɲ̀-t͡ʃáŋ]** ('I went')
- `/N/` + `/kpa/` ('nail/fix') $\rightarrow$ **[ŋ͡m̀-k͡pá]** ('I fixed')
- `/N/` + `/ka/` ('have not') $\rightarrow$ **[ŋ̀-ká]** ('I do not have')

---

### 2.2 Contextual Consonant Mutations (Rule MPH-02 & MPH-03)

#### A. Palatalization before Front Vowels `{i, e, ɛ}`
Underlying velar and alveolar consonants shift toward post-alveolar and palatal places of articulation when immediately preceding front vowels:
- **/k/ $\rightarrow$ [t͡ʃ]**: `/kɛhi/` $\rightarrow$ **[t͡ʃɛ́hɨ́]** ('to laugh / split')
- **/ɡ/ $\rightarrow$ [d͡ʒ]**: `/ɡɛ/` $\rightarrow$ **[d͡ʒɛ́]** ('to be tired / lean')
- **/s/ $\rightarrow$ [ʃ]**: `/sia/` $\rightarrow$ **[ʃíá]** ('bee / spirit / waist')
- **/z/ $\rightarrow$ [ʒ]**: `/zɛ'ʊ/` $\rightarrow$ **[ʒɛ́ʔʊ́]** ('storm / tempest')
- **/ŋ/ $\rightarrow$ [ɲ]**: `/ŋɛb/` $\rightarrow$ **[ɲɛ́bígá]** ('crocodile')

#### B. Labial-Coronal Realization of Labial-Velars
Labial-velar stops `/k͡p, ɡ͡b/` and nasal `/ŋ͡m/` shift their secondary constriction forward to labial-coronal `[t͡p, d͡b, n͡m]` when followed by front vowels `{i, e, ɛ}`:
- **/k͡p/ + /ɛ/** $\rightarrow$ **[t͡pɛ́]** (*kpɛ* 'enter')
- **/ɡ͡b/ + /ɛ/** $\rightarrow$ **[d͡bɛ́]** (*gbe* 'to lodge / spend night')
- **/ŋ͡m/ + /ɛ/** $\rightarrow$ **[n͡mɛ́]** (*ŋmɛ* 'to hit / beat')

---

### 2.3 Intervocalic Lenition & Debuccalization (Rule MPH-04 & MPH-05)

#### A. Alveolar Stop Lenition (/d/ $\rightarrow$ [r])
In native morpheme stems and inflectional suffixes, underlying alveolar plosive `/d/` weakens to tap/flap `[r]` in intervocalic ($V\_V$) position:
- `/kad-/` (chase) + `/-a/` (imperf.) $\rightarrow$ **[kárá]** ('chasing away')
- `/dɔr-/` (illness) + `/-o/` (sing.) $\rightarrow$ **[dóró]** ('illness')

#### B. Sibilant Debuccalization (/s/ $\rightarrow$ [h])
In native stems, underlying `/s/` undergoes debuccalization (loss of supralaryngeal articulation) to glottal fricative `[h]` between vowels:
- `/bii-/` (child) + `/-si/` (plural) $\rightarrow$ **[bíhí]** ('children')
- `/daa-/` (market) + `/-si/` (plural) $\rightarrow$ **[dáhí]** ('markets')
- **Consonant Cluster Blocking**: When `/s/` follows a nasal coda, debuccalization is **strictly blocked**:
  - `/duun-/` (room) + `/-si/` (plural) $\rightarrow$ **[dúúnsí]** (*never* `*[dúúnhí]`).

#### C. Velar Fricativization (/ɡ/ $\rightarrow$ [ɣ] / [ʔ])
Underlying voiced velar plosive `/ɡ/` spirantizes intervocalically to `[ɣ]` or reduces to glottal stop `[ʔ]`:
- `/bɔɡ-u/` $\rightarrow$ **[bɔ̀ɣú]** ('hole / pit')
- `/bɛɡ-u/` $\rightarrow$ **[bɛ́ʔʊ́]** ('badness / morning')

---

### 2.4 Suffixal Consonant Gemination (Rule MPH-06)
Attachment of noun class suffixes (e.g. Class 2 singular marker `-li/-la`) to liquid-final stems induces complete assimilation and consonant gemination:
- `/ɡbal/` + `/-li/` $\rightarrow$ **[ɡ͡bálːí]** ('grave / tomb')
- `/bil/` + `/-li/` $\rightarrow$ **[bílːí]** ('seedling')
- `/pul/` + `/-li/` $\rightarrow$ **[púlːí]** ('stomach / abdomen')

---

### 2.5 Hiatus Resolution & Vowel Elision (Rule MPH-07)
In fluent connected speech, when a word ending in a short vowel precedes a word beginning with a vowel, the final vowel of the first word elides ($V_1 \# V_2 \rightarrow \emptyset \# V_2$):
- *tí* ('give') + *ó* ('him/her') $\rightarrow$ **t'ó** `[t-ó]` ('give him/her')
- *nì* ('and') + *ó* ('him/her') $\rightarrow$ **n'ó** `[n-ó]` ('and him/her')
- *bá* ('father') + *ó* ('his') $\rightarrow$ **b'ó** `[b-ó]` ('his father')

---

### 2.6 Epenthetic Vowel Insertion for Illegal Clusters (Rule MPH-08)
Dagbani phonotactics prohibits non-homorganic consonant clusters across syllable onsets. Loanwords with initial complex onsets ($sC-$, $CL-$) or illicit codas undergo epenthetic repair via weak vowels `[ɨ]` or `[ʊ]`:
- English *class* $\rightarrow$ Dagbani **kɨ̀láasì** `[kɨ̀láːsì]`
- English *block* $\rightarrow$ Dagbani **bʊ̀lókú** `[bʊ̀lókú]`
- English *school* $\rightarrow$ Dagbani **sùkúùrù** `[sùkúùrù]`
- English *doctor* $\rightarrow$ Dagbani **dɔ́kɨ̀tà** `[dɔ́kɨ̀tà]`

---

### 2.7 Tone Sandhi & Tense-Aspect Melodic Overrides (Rule MPH-09)
Verbal roots inherit dominant grammatical tonal overlays dictated by pre-verbal Tense/Aspect/Mood (TAM) particles and negation operators:

1. **Perfective Aspect Overlay**: Verb root takes Low tone melody followed by High suffix:
   - Root *chàŋ* ('go') + TAM *-ya* $\rightarrow$ **chàŋ-yá** `[t͡ʃàŋjá]`
2. **Imperfective Aspect Overlay**: Verb root takes High tone melody:
   - Root *chán* ('go') + TAM *-di* $\rightarrow$ **chán-dí** `[t͡ʃándí]`
3. **Negation Polarity Inversion**: Negative particles (*kù* 'future neg', *bì* 'past neg', *dī* 'prohibitive') invert the default lexical pitch of the immediate verb nucleus.
