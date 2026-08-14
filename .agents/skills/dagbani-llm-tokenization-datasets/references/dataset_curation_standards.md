# Dagbani Dataset Curation, Cleaning & Governance Standards

This reference guide establishes data hygiene, validation rules, deduplication protocols, and ethical governance standards for Dagbani speech and text corpora.

---

## 1. Corpus Validation & Text Cleaning Pipeline

```
Raw Scraped Dumps / Transcripts
  │
  ├──► [Step 1: Unicode Normalization] (Enforce NFC canonical decomposition & recomposition)
  ├──► [Step 2: Punctuation & Quotes Standardization] (Replace smart quotes ‘ ’ “ ” with standard ' ")
  ├──► [Step 3: Character Set Filtering] (Retain valid Dagbani letters, digits, and standard punctuation)
  ├──► [Step 4: Non-Dagbani & Mixed-Script Filtering] (Reject Cyrillic, Chinese, Arabic without loan context)
  ├──► [Step 5: Length & Word Count Constraints] (Min 3 words, Max 256 words per training line)
  └──► [Step 6: MinHash & Exact Deduplication] (Eliminate duplicate web dumps and repeated boilerplate)
  │
  ▼
Curated Training Split (train / val / test)
```

---

## 2. Dagbani Orthographic Character Whitelist

Text processors must recognize and preserve all 28 BGL 1998 standard letters and digraphs:

```python
DAGBANI_ALLOWED_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ɛɔɣŋʒ"
    "ƐƆƔŊƷ"
    "0123456789"
    ".,!?;:'\"-–—()[] "
)
```

Any utterance containing $> 5\%$ characters outside this whitelist is flagged for human review or rejected.

---

## 3. Deduplication Strategy

Scraped web text (e.g. Wikipedia navigation templates, Bible chapter headings) often contains severe duplication.
1. **Exact Line Deduplication**: Hash each normalized line with SHA-256 and drop duplicates.
2. **Fuzzy Document Deduplication**: Use 64-bit **MinHash** with Jaccard similarity threshold $> 0.85$ to remove boilerplate articles and scraped forum headers.

---

## 4. Master Datasets & Licensing Summary

| Dataset | Volume | License | Recommended Split |
|---|---|---|---|
| **WAXAL Speech Corpus** | 1,000 hours | Permissive Research | 80% Train / 10% Val / 10% Test (Speaker-Disjoint) |
| **Mozilla Common Voice (v24)** | 20,000+ clips | CC0 1.0 Universal | Speaker-Disjoint Partition |
| **Spell4Wiki Dagbani** | 11,207 clips (~9.5h) | CC0 1.0 Universal | Held-out 2 speakers for Test |
| **Dagbani Bible (Open.Bible)** | 75h Audio + Text | CC-BY-SA 4.0 | Book/Chapter-level split |
| **Dagbani Wikipedia Dumps** | ~10M tokens | CC-BY-SA 3.0 | Article-level deduplicated split |
| **GhanaNLP Translation Pairs** | 41,513 pairs | Commercial / Academic | 38k Train / 1.5k Val / 2k Test |
| **Samban’ luŋa Drum Histories** | 500k+ characters | Academic Archive | Domain-adaptation CPT only |

---

## 5. Ethical Governance & Cultural Sensitivity

1. **Sacred Panegyrics (*Samban’ luŋa*)**: Panegyric drumming texts detailing royal Yaa-Naa ancestries must preserve traditional poetic forms without disrespectful or harmful model hallucinations.
2. **Open-Access Licensing**: All derived tokenizer vocabularies, cleaned corpora, and LoRA adapters should be distributed under permissive open licenses (Apache 2.0 / CC-BY-4.0) to empower local Ghanaian developers.
