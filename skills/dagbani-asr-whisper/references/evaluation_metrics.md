# Dagbani ASR Evaluation Metrics & Forensic Audit Protocol

**Document Version**: 1.0.0  
**Scope**: Word Error Rate (WER), Normalized WER, Character Error Rate (CER), Phoneme Error Rate (PER), and Special Character Precision/Recall.

---

## 1. Metric Mathematical Formulations

### 1.1 Word Error Rate (WER)
Word Error Rate computes the minimum number of word insertions ($I$), deletions ($D$), and substitutions ($S$) required to transform the hypothesis sequence into the reference sequence, divided by the total number of reference words ($N$):

$$\text{WER} = \frac{S + D + I}{N} = \frac{\sum_{i=1}^K \text{Levenshtein}(\text{ref}_i, \text{hyp}_i)}{\sum_{i=1}^K |\text{ref}_i|}$$

### 1.2 Character Error Rate (CER)
Character Error Rate evaluates Levenshtein distance at the individual character/grapheme level:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

---

## 2. Normalized WER vs Strict WER

In low-resource oral-centric languages like Dagbani, strict string equality introduces substantial artificial penalties due to inconsistent punctuation, casing, or apostrophe variants.

### 2.1 Dagbani Text Normalization Protocol
Before computing **Normalized WER**, both reference and hypothesis strings are sanitized through the standard pipeline:
1. **Unicode NFC Canonicalization**: Ensures precomposed glyph consistency.
2. **Apostrophe & Quote Standardization**: Converts `’`, `‘`, `` ` `` $\rightarrow$ `'`, and removes outer quotes.
3. **Punctuation Stripping**: Retains only alphanumeric characters, spaces, hyphens, and apostrophes.
4. **Lowercasing**: Case-insensitive comparison.

```python
import unicodedata, re

def normalize_for_wer(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.lower()
    cleaned = [ch for ch in text if unicodedata.category(ch).startswith("L") or ch in {" ", "'", "-"}]
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()
```

---

## 3. Special Character Precision, Recall & F1 Tracking

The 5 extended Dagbani characters (`ɛ`, `ɔ`, `ŋ`, `ɣ`, `ʒ`) are essential for semantic and grammatical contrast. Evaluating model performance specifically on these glyphs detects orthographic collapse.

### 3.1 Mathematical Definitions

For each target character $c \in \{\text{'ɛ'}, \text{'ɔ'}, \text{'ŋ'}, \text{'ɣ'}, \text{'ʒ'}\}$:

$$\text{Recall}(c) = \frac{\sum_{j=1}^M \min\left(N_{\text{ref}}(c, j), N_{\text{hyp}}(c, j)\right)}{\sum_{j=1}^M N_{\text{ref}}(c, j)}$$

$$\text{Precision}(c) = \frac{\sum_{j=1}^M \min\left(N_{\text{ref}}(c, j), N_{\text{hyp}}(c, j)\right)}{\sum_{j=1}^M N_{\text{hyp}}(c, j)}$$

$$F_1(c) = \frac{2 \times \text{Precision}(c) \times \text{Recall}(c)}{\text{Precision}(c) + \text{Recall}(c) + \epsilon}$$

### 3.2 Target Quality Benchmarks
- **Overall Normalized WER Target**: $\le 25.0\%$ on clean read speech; $\le 38.0\%$ on noisy field recordings.
- **Special Glyph Recall**: $\ge 90.0\%$ for all 5 characters (`ɛ, ɔ, ŋ, ɣ, ʒ`).
- **Character Error Rate (CER)**: $\le 8.5\%$.
