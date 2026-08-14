# Comprehensive Catalog of Dagbani Speech and Text Datasets & Benchmarks

**Document ID**: `DAG-KB-DATA-001`  
**Version**: `1.0.0` (Production Reference Catalog)  
**Corpora Count**: 13 Authoritative Speech & Text Datasets  
**Coverage**: ASR, TTS, Machine Translation, Continual Pretraining, Instruction Tuning, Lexicography  

---

## 1. Master Inventory of Dagbani Corpora

Below is the definitive catalog of all known speech, parallel text, monolingual text, and lexicographical datasets for the Dagbani language, compiled across academic publications, open-source repositories, and cultural archives.

| # | Dataset Name | Primary Modality | Total Volume / Duration | License | Format / Audio Spec | Dialects / Demographics | Source / Repository Link |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **WAXAL Speech Corpus (Ghanaian Stream)** | Spontaneous Audio + Transcript | **1,000 hours** Dagbani (part of 5,000h Ghana corpus) | Open Research / Permissive | WAV (16 kHz / 24 kHz mono) | Western (Tomosili) & Eastern (Nayahali) | Google Research / Univ. of Ghana / Makerere (arXiv:2602.02734) |
| **2** | **WAXAL-TTS (Studio Synthesis Stream)** | High-Fidelity Audio + Text | **235 hours** project total (~20–40h Dagbani) | Open Research | Studio 48 kHz / 24-bit WAV | Standard Dagbani (Studio Voice Actors) | Google Research WAXAL Project |
| **3** | **Mozilla Common Voice (Dagbani v24)** | Audio + Crowdsourced Transcripts | **40,000+ sentences**, **20,000+ clips** (~20–30h) | CC0 1.0 Universal | MP3 / WAV (32–48 kHz), TSV metadata | Tomosili & Nayahali (Age / Gender tagged) | Mozilla Foundation / Dagbani Wikimedians |
| **4** | **Spell4Wiki Dagbani Speech Corpus** | Audio + Aligned Text | **11,207 utterances** (9h 34m, 48,636 tokens) | CC0 1.0 Universal | OGG / WAV (44.1 kHz mono) | 22 Speakers (15M / 7F), Tamale residential | Wikimedia Commons (`Files_uploaded_by_spell4wiki_in_dag`) |
| **5** | **UGSpeechData / SciDB Ghanaian Audio Corpus** | Audio + Transcribed Excel Metadata | **1,000+ recordings** (part of 1,000h project) | Open Academic | MP3 (16 kHz) + `Dagbani.xlsx` metadata | Northern Ghanaian multi-speaker | Science Data Bank (SciDB China / Univ. of Ghana) |
| **6** | **GhanaNLP Parallel Translation Corpus** | Parallel Text (Dagbani-English) | **41,513 sentence pairs** | Permissive Commercial / Academic | JSON / TSV | Standard Literary Dagbani | GhanaNLP (`translation.ghananlp.org` / Khaya AI) |
| **7** | **Dagbani Bible Audio & Text (Open.Bible / BibleTTS)** | Parallel Audio + Verse-Aligned Text | **~75–86 hours** (Old & New Testaments) | CC-BY-SA / Biblica Open.Bible | 48 kHz / 24 kHz WAV + Verse TXT | Standard Literary / Ecclesiastical | Open.Bible / Biblica / BibleTTS (Meyer et al.) |
| **8** | **JW300 & JW.org Dagbani Corpus** | Audio + Multi-chapter Parallel Text | **~70.04 hours** (27,068 utterances) + Text | Permissive / Religious | WAV (24 kHz) + Sentence pairs | Standard Dagbani | JW.org (scraped via `jwsoup`) / OPUS JW300 |
| **9** | **Dagbani Wikipedia Full Text Dump** | Monolingual Text | **~15,000–30,000 articles** (~5M–10M tokens) | CC-BY-SA 3.0 / CC0 | XML / JSON / Plain Text | Mixed Tomosili & Nayahali | Wikimedia Foundation (`dag.wikipedia.org`) |
| **10** | **Wikidata Dagbanli Lexicographical Database** | Structured Lexemes & Senses | **Thousands of lexemes**, forms, senses | CC0 1.0 Universal | JSON-LD / SPARQL API | Standard Dagbani | Wikidata Lexemes (`diff.wikimedia.org/...`) |
| **11** | **Oral Literature & Drum History (*Samban’ luŋa*)** | Transcribed Audio & Panegyrics | **500k+ characters text** + Field tapes | Academic / Cultural Archive | PDF / Text / Audio Field Recordings | Royal Court Dagbani (Yendi / Tamale) | *Grandmasters of the Drum* (Taluah / Univ. of Bayreuth) |
| **12** | **Tamale & Yendi Radio Broadcast Archives** | Spontaneous Broadcast Audio | **100+ hours** raw broadcast streams | Educational / Community Archive | MP3 / WAV (44.1 kHz) | Spontaneous conversational Dagbani | Local Stations (Simli Radio, Radio Justice, Zaa Radio) |
| **13** | **GhanaNLP Navigation Speech Corpus** | Audio + Domain Transcriptions | Specialized voice navigation commands | Open Access | WAV + JSON | Standard Dagbani | Hugging Face (`ghananlpcommunity/navigation-corpus-dagbani-speech`) |

---

## 2. Detailed Dataset Profiles & Technical Specifications

### 2.1 WAXAL Speech Corpus (Ghanaian Component)
- **Institutional Creators**: Google Research, University of Ghana, Makerere University.
- **Methodology**: Addresses the artificial reading prosody bottleneck by utilizing an **image-prompted spontaneous elicitation paradigm**. Native speakers describe 1,000 culturally grounded photographic scenes (market trading, Shea butter extraction, farming, clinic triage).
- **Quality & Verification**: 10% of spontaneous recordings are gold-standard transcribed by phonetic experts; remaining 90% is paired with pseudo-labels and acoustic features.
- **Use Case**: Primary pretraining corpus for robust conversational ASR and multimodal models.

### 2.2 WAXAL-TTS Studio Stream
- **Methodology**: Custom acoustic studio recording booths deployed in Tamale and Accra. Professional native voice actors record phonetically balanced scripts covering all 11 Dagbani vowels, labial-velar stops (`kp`, `gb`), and tonal nominal patterns.
- **Audio Fidelity**: 48.0 kHz, 24-bit uncompressed PCM WAV ($\text{SNR} > 38\text{ dB}$, $\text{RT}_{60} < 0.18\text{s}$).
- **Use Case**: Golden reference training set for single-speaker VITS and BigVGAN vocoder adaptation.

### 2.3 Mozilla Common Voice (Dagbani v24)
- **Community Stewards**: Dagbani Wikimedians User Group (Tamale, Ghana).
- **Corpus Dynamics**: Crowdsourced web platform where native speakers record sentence prompts extracted from Dagbani Wikipedia.
- **Validation Engine**: Each clip requires at least 2 independent peer upvotes to enter the "validated" split.
- **Demographics**: Metadata includes self-reported gender, age brackets (19–29, 30–39, 40–49), and regional accents (Western/Eastern).

### 2.4 Spell4Wiki Dagbani Speech Corpus
- **Institutional Origin**: GhanaNLP & Wikimedia Commons initiative.
- **Statistics**: 11,207 audio files; 9 hours 34 minutes total duration; 48,636 total word tokens; 7,222 unique vocabulary items.
- **Speakers**: 22 distinct native speakers (15 male, 7 female), recorded in home/office settings in Tamale using smartphone headsets.
- **Use Case**: Standardized benchmark evaluation set for zero-shot and fine-tuned ASR.

### 2.5 Dagbani Bible Audio & Verse Alignment (Open.Bible / BibleTTS)
- **Creators**: Biblica Open.Bible initiative and BibleTTS research project.
- **Acoustic Characteristics**: High-fidelity narration of the complete Old and New Testaments in Dagbani.
- **Alignment**: Verse-level timestamps generated via forced alignment tools (`fairseq` / `mfa`).
- **Volume**: ~80 hours across 66 books, providing rich ecclesiastical and narrative vocabulary.

---

## 3. Data Hygiene, Normalization & Cleaning Protocols

### 3.1 Dagbani Text Normalization Engine
All text inputs across ASR, TTS, and LLM training pipelines must pass through the canonical Dagbani normalizer:

```python
import re
import unicodedata

DAGBANI_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzɛɔɣŋʒ-' ")

def clean_dagbani_corpus_text(text: str) -> str:
    # 1. Unicode Canonical Decomposition and NFC Recomposition
    text = unicodedata.normalize("NFC", str(text))

    # 2. Transliterate typography variants to standard BGL / ASCII
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("gh", "ɣ").replace("Gh", "Ɣ")
    text = text.replace("zh", "ʒ").replace("Zh", "Ʒ")
    text = text.replace("ngm", "ŋm").replace("Ngm", "Ŋm")
    
    # 3. Lowercase
    text = text.lower()

    # 4. Filter non-alphabet characters
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in {"-", "'", " "}:
            cleaned.append(ch)
        elif ch in DAGBANI_ALLOWED_CHARS:
            cleaned.append(ch)
        elif cat.startswith("L"):
            cleaned.append(ch)
        elif ch.isdigit():
            cleaned.append(f" {ch} ")
        else:
            cleaned.append(" ")

    text = "".join(cleaned)
    return re.sub(r"\s+", " ", text).strip()
```

---

## 4. Speaker-Disjoint Splitting Protocol

### 4.1 Danger of Random Splitting
Random sentence-level train/test partitioning in speech datasets causes **acoustic data leakage**: acoustic models memorize speaker vocal tract timbre, resulting in artificially low WERs ($<5\%$) that degrade to $>35\%$ on unseen speakers.

### 4.2 Partitioning Standard

$$\mathcal{D}_{\text{total}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad \text{where } \text{Speakers}(\mathcal{D}_{\text{train}}) \cap \text{Speakers}(\mathcal{D}_{\text{test}}) = \emptyset$$

- **Train Split (80%–90% of unique speakers)**: Used exclusively for parameter updates.
- **Validation Split (5%–10% of unique speakers)**: Used for hyperparameter tuning and early stopping.
- **Test Split (5%–10% of unique speakers)**: Held-out strictly for final metric reporting.

---

## 5. Standardized Benchmarking Suite

| Benchmark Suite | Modality / Task | Evaluation Corpora | Primary Metric | Secondary Metric | SOTA Baseline Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dagbani-ASR-Bench** | Speech-to-Text | Spell4Wiki (Test) + Common Voice (Test) | **WER** (Strict & Norm) | **CER**, Special Glyph Recall | $\text{WER} < 18.0\%$, $\text{CER} < 4.0\%$ |
| **Dagbani-TTS-Bench** | Text-to-Speech | Held-out 100 sentences (WAXAL script) | **MOS** (1–5 human) | **UTMOS**, MCD ($\text{dB}$) | $\text{MOS} > 4.20$, $\text{UTMOS} > 3.40$ |
| **Dagbani-MT-Bench** | Machine Translation | GhanaNLP Parallel Test Set (2,000 pairs) | **CHRF++** | **BLEU**, **TER** | $\text{CHRF} > 52.0$, $\text{BLEU} > 28.0$ |
| **Dagbani-MMLU** | Cultural & General QA | 500 Curated Dagbamba Cultural MCQs | **Accuracy** | Log-Likelihood PPL | $\text{Accuracy} > 75.0\%$ |

---

## 6. References & Data Sources
1. Google Research, University of Ghana, Makerere University. (2026). *WAXAL: A Large-Scale Multilingual African Language Speech Corpus*. arXiv:2602.02734.
2. Mozilla Foundation. (2024). *Common Voice Corpus 17.0/24.0: Dagbani Language Component*.
3. Ibrahim, A., et al. (2023). *Breaking the Low-Resource Barrier for Dagbani ASR*. AfricaNLP @ ICLR 2023.
4. Wikimedia Foundation. (2024). *Spell4Wiki and Dagbani Wikipedia Dumps*.
5. Biblica & Meyer, J., et al. (2022). *BibleTTS: a large, high-fidelity multilingual speech corpus for speech synthesis*. Interspeech 2022.
6. Taluah, N. C. (2021). *Grandmasters of the Drum: A Literary Linguistic Analysis of Dagbamba Panegyrics*.
