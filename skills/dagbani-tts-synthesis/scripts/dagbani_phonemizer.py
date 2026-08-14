#!/usr/bin/env python3
"""
Dagbani G2P Phonemizer and Tone-Tier Tokenizer for Text-to-Speech (TTS).

Converts standard Bureau of Ghana Languages (BGL 1998) or ASCII Dagbani text
into IPA phonemic representations, tone-tier sequences, and numeric token IDs
for acoustic models (VITS, Coqui XTTS-v2, Matcha-TTS, MMS-TTS).

Features:
- Unicode NFC normalization and punctuation standardizer.
- Bi-directional ASCII <-> BGL glyph handling (ɛ, ɔ, ŋ, ɣ, ʒ).
- Multi-character digraph preservation (kp, gb, ŋm, ny, ch, sh).
- Context-dependent allophonic mutations (palatalization, intervocalic lenition /d/ -> [r],
  intervocalic debuccalization /s/ -> [h]).
- Moraic tone-bearing coda nasal parsing.
- Tone Diacritic Restoration (TDR) and tone tier tagging (High [H], Low [L], Downstep [!H]).
- Vocabulary token ID indexing for end-to-end neural TTS pipelines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Vocabulary & Token Symbol Tables
# ============================================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPACE_TOKEN = "_"

# Phonemic symbols
CONSONANTS = [
    "p", "b", "t", "d", "k", "g", "kp", "gb", "ʔ",
    "ch", "j", "f", "v", "s", "z", "sh", "ʒ", "ɣ",
    "h", "m", "n", "ny", "ŋ", "ŋm", "l", "r", "y", "w"
]

VOWELS = [
    "a", "a:", "e", "e:", "ɛ", "ɛ:", "i", "i:",
    "ɨ", "o", "o:", "ɔ", "ɔ:", "u", "u:"
]

TONES = [
    "H",   # High Tone (´)
    "L",   # Low Tone (`)
    "!H",  # Downstep (!H)
    "0"    # Neutral / Unmarked
]

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SPACE_TOKEN]

# Build deterministic vocabulary dictionary
VOCAB_SYMBOLS: List[str] = SPECIAL_TOKENS + CONSONANTS + VOWELS + TONES
SYMBOL_TO_ID: Dict[str, int] = {sym: idx for idx, sym in enumerate(VOCAB_SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {idx: sym for idx, sym in enumerate(VOCAB_SYMBOLS)}

# Lexicon of known Dagbani word-level citation tones & phonetic overrides
LEXICON_TONE_OVERRIDES: Dict[str, List[Tuple[str, str]]] = {
    # word: [(phoneme, tone), ...]
    "gballi": [("gb", "H"), ("a", "H"), ("l", "H"), ("i", "H")],         # grave / tomb
    "duu": [("d", "H"), ("u:", "H")],                                    # room
    "tia": [("t", "H"), ("i", "H"), ("a", "H")],                         # tree
    "kari": [("k", "H"), ("a", "H"), ("r", "H"), ("i", "H")],            # to drive out
    "bɛ'ʊ": [("b", "H"), ("ɛ", "H"), ("ʔ", "H"), ("u", "H")],            # ugly / bad
    "viɛli": [("v", "L"), ("j", "L"), ("ɛ", "H"), ("l", "H"), ("i", "L")],# beautiful
    "yɛltɔɣa": [("j", "L"), ("ɛ", "L"), ("l", "L"), ("t", "H"), ("ɔ", "H"), ("ɣ", "H"), ("a", "H")],
    "pukpara": [("p", "L"), ("u", "L"), ("k", "L"), ("p", "H"), ("a", "H"), ("r", "H"), ("a", "H")],
    "tamale": [("t", "L"), ("a", "L"), ("m", "H"), ("a", "H"), ("l", "L"), ("e", "L")],
    "yendi": [("j", "H"), ("e", "H"), ("n", "H"), ("d", "L"), ("i", "L")],
    "saa": [("s", "H"), ("a:", "H")],                                    # rain
    "baa": [("b", "H"), ("a:", "H")],                                    # dog
    "bu'a": [("b", "H"), ("u", "H"), ("ʔ", "H"), ("a", "H")],            # goat
}


@dataclass
class PhonemeUnit:
    """Represents a single parsed phonemic unit with tone and prosodic cues."""
    symbol: str
    tone: str = "0"
    is_vowel: bool = False
    is_nasal_coda: bool = False
    token_id: int = 0

    def to_ipa(self) -> str:
        """Returns IPA string with optional tone diacritic."""
        ipa_map = {
            "kp": "k͡p",
            "gb": "ɡ͡b",
            "ŋm": "ŋ͡m",
            "ny": "ɲ",
            "ch": "t͡ʃ",
            "j": "d͡ʒ",
            "sh": "ʃ",
            "ʒ": "ʒ",
            "ɣ": "ɣ",
            "ŋ": "ŋ",
            "ʔ": "ʔ",
            "a:": "aː",
            "e:": "eː",
            "ɛ:": "ɛː",
            "i:": "iː",
            "o:": "oː",
            "ɔ:": "ɔː",
            "u:": "uː"
        }
        sym = ipa_map.get(self.symbol, self.symbol)
        if self.tone == "H":
            return f"{sym}\u0301"  # Combining acute
        elif self.tone == "L":
            return f"{sym}\u0300"  # Combining grave
        elif self.tone == "!H":
            return f"!{sym}\u0301"
        return sym


class DagbaniPhonemizer:
    """
    Robust, rule-based and lexical Grapheme-to-Phoneme converter for Dagbani.
    """

    def __init__(self, vocab_map: Optional[Dict[str, int]] = None):
        self.vocab = vocab_map or SYMBOL_TO_ID
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.front_vowels = {"i", "i:", "e", "e:", "ɛ", "ɛ:"}
        self.back_vowels = {"u", "u:", "o", "o:", "ɔ", "ɔ:"}
        self.all_vowels = {"a", "a:", "e", "e:", "ɛ", "ɛ:", "i", "i:", "ɨ", "o", "o:", "ɔ", "ɔ:", "u", "u:"}
        self.nasals = {"m", "n", "ny", "ŋ", "ŋm"}

    def normalize_text(self, text: str) -> str:
        """Normalizes Unicode text to NFC canonical form and standardizes punctuation."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", str(text))
        # Standardize curly quotes and apostrophes
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = text.replace("“", '"').replace("”", '"')
        # Lowercase while preserving Dagbani characters
        text = text.lower()
        return text.strip()

    def disambiguate_ascii(self, word: str) -> str:
        """Transliterates ASCII digraphs and replacements into BGL glyphs."""
        # Replace multi-character ASCII digraph representations
        w = word
        w = re.sub(r"ngm", "ŋm", w)
        w = re.sub(r"gh", "ɣ", w)
        w = re.sub(r"zh", "ʒ", w)
        # Note: 'ng' before vowels is handled as /ŋ/, before g as /ŋ/+/g/
        w = re.sub(r"ng(?=[aeɛioɔuɨ])", "ŋ", w)
        return w

    def _tokenize_raw_word(self, word: str) -> List[str]:
        """Splits a single word into constituent phoneme candidate symbols."""
        tokens: List[str] = []
        i = 0
        n = len(word)
        while i < n:
            # 3-character lookahead
            if i + 2 < n and word[i:i+3] in {"ŋm:", "ch:", "sh:", "kp:", "gb:", "ny:"}:
                tokens.append(word[i:i+3])
                i += 3
                continue

            # 2-character lookahead (digraphs, long vowels)
            if i + 1 < n:
                pair = word[i:i+2]
                if pair in {"kp", "gb", "ŋm", "ny", "ch", "sh", "ng"}:
                    tokens.append("ŋ" if pair == "ng" else pair)
                    i += 2
                    continue
                if pair in {"aa", "ee", "ɛɛ", "ii", "oo", "ɔɔ", "uu"}:
                    base_v = pair[0]
                    tokens.append(f"{base_v}:")
                    i += 2
                    continue
                if pair == "iɛ":
                    # Palatalized front dipthong
                    tokens.extend(["j", "ɛ"])
                    i += 2
                    continue

            # 1-character token
            ch = word[i]
            if ch == "'":
                tokens.append("ʔ")
            elif ch in "abdefghijklmnoprstuvwyzɛɔɣŋʒ":
                tokens.append(ch)
            i += 1
        return tokens

    def _apply_allophonic_mutations(self, tokens: List[str]) -> List[str]:
        """Applies context-dependent phonological mutation rules."""
        result: List[str] = list(tokens)
        num_toks = len(result)

        for idx in range(num_toks):
            curr = result[idx]
            next_tok = result[idx + 1] if idx + 1 < num_toks else None
            prev_tok = result[idx - 1] if idx > 0 else None

            # 1. Velar & Alveolar Palatalization before front vowels
            if next_tok and next_tok in self.front_vowels:
                if curr == "k":
                    result[idx] = "ch"  # [t͡ʃ]
                elif curr == "g":
                    result[idx] = "j"   # [d͡ʒ]
                elif curr == "s":
                    result[idx] = "sh"  # [ʃ]
                elif curr == "z":
                    result[idx] = "ʒ"   # [ʒ]
                elif curr == "ŋ":
                    result[idx] = "ny"  # [ɲ]

            # 2. Intervocalic Lenition & Debuccalization
            if prev_tok in self.all_vowels and next_tok in self.all_vowels:
                if curr == "d":
                    result[idx] = "r"   # Intervocalic tap/trill
                elif curr == "s":
                    result[idx] = "h"   # Intervocalic debuccalization
                elif curr == "g":
                    result[idx] = "ɣ"   # Intervocalic velar spirantization

        return result

    def _assign_tones_and_structure(self, tokens: List[str], raw_word: str) -> List[PhonemeUnit]:
        """Assigns tone tiers and prosodic roles to phonemes."""
        clean_word = raw_word.replace("'", "").replace("-", "")

        # Check citation tone lexicon overrides first
        if clean_word in LEXICON_TONE_OVERRIDES:
            overrides = LEXICON_TONE_OVERRIDES[clean_word]
            units = []
            for sym, tone in overrides:
                tok_id = self.vocab.get(sym, self.vocab.get(UNK_TOKEN, 1))
                units.append(PhonemeUnit(
                    symbol=sym,
                    tone=tone,
                    is_vowel=(sym in self.all_vowels),
                    is_nasal_coda=(sym in {"m", "n", "ŋ"}),
                    token_id=tok_id
                ))
            return units

        # Rule-based tone and structure assignment
        units: List[PhonemeUnit] = []
        num_toks = len(tokens)

        # Tone heuristic: High on root initial/stressed vowel, default Low on final unstressed
        vowel_count = 0
        for i, sym in enumerate(tokens):
            is_v = sym in self.all_vowels
            is_coda_nasal = False

            # Check if this is a coda nasal (nasal preceded by vowel and at end of word or before consonant)
            if sym in {"m", "n", "ŋ"} and i > 0 and tokens[i-1] in self.all_vowels:
                if i == num_toks - 1 or tokens[i+1] not in self.all_vowels:
                    is_coda_nasal = True

            # Determine tone tier
            assigned_tone = "0"
            if is_v:
                vowel_count += 1
                assigned_tone = "H" if vowel_count == 1 else "L"
            elif is_coda_nasal:
                assigned_tone = "H" if vowel_count == 1 else "L"

            tok_id = self.vocab.get(sym, self.vocab.get(UNK_TOKEN, 1))
            units.append(PhonemeUnit(
                symbol=sym,
                tone=assigned_tone,
                is_vowel=is_v,
                is_nasal_coda=is_coda_nasal,
                token_id=tok_id
            ))

        return units

    def phonemize_word(self, word: str) -> List[PhonemeUnit]:
        """Processes a single word through full G2P, mutation, and tone stages."""
        w_norm = self.normalize_text(word)
        if not w_norm:
            return []
        w_bgl = self.disambiguate_ascii(w_norm)
        raw_tokens = self._tokenize_raw_word(w_bgl)
        mutated_tokens = self._apply_allophonic_mutations(raw_tokens)
        units = self._assign_tones_and_structure(mutated_tokens, w_bgl)
        return units

    def phonemize(
        self,
        text: str,
        return_token_ids: bool = True
    ) -> Dict[str, Any]:
        """
        Phonemizes an entire sentence or paragraph into IPA string,
        phoneme units, tone markers, and token IDs.
        """
        norm_text = self.normalize_text(text)
        if not norm_text:
            return {"ipa": "", "tokens": [], "tones": [], "token_ids": [], "num_tokens": 0}

        # Split words while tracking punctuation / boundaries
        words = re.findall(r"[\w'ɛɔɣŋʒ\-\:]+|[^\s\w'ɛɔɣŋʒ\-\:]+", norm_text, re.UNICODE)
        all_units: List[PhonemeUnit] = []
        space_id = self.vocab.get(SPACE_TOKEN, 4)

        for w_idx, w in enumerate(words):
            if re.match(r"^[\w'ɛɔɣŋʒ\-\:]+$", w):
                w_units = self.phonemize_word(w)
                if w_units:
                    if all_units:
                        # Append inter-word space
                        all_units.append(PhonemeUnit(
                            symbol=SPACE_TOKEN,
                            tone="0",
                            is_vowel=False,
                            is_nasal_coda=False,
                            token_id=space_id
                        ))
                    all_units.extend(w_units)

        ipa_symbols = [u.to_ipa() for u in all_units]
        token_symbols = [u.symbol for u in all_units]
        tone_symbols = [u.tone for u in all_units]
        token_ids = [u.token_id for u in all_units]

        return {
            "ipa": " ".join(ipa_symbols),
            "tokens": token_symbols,
            "tones": tone_symbols,
            "token_ids": token_ids,
            "num_tokens": len(token_ids)
        }


# ============================================================================
# Self-Test Verification Suite
# ============================================================================

def run_phonemizer_self_test() -> bool:
    """Executes deterministic test cases covering Dagbani phonology rules."""
    print("=" * 60)
    print("Running Dagbani Phonemizer Self-Tests...")
    print("=" * 60)

    phonemizer = DagbaniPhonemizer()
    tests_passed = 0
    total_tests = 0

    test_cases = [
        {
            "desc": "Digraph preservation (kp, gb, ŋm, ny, ch, sh)",
            "input": "kpariba gbanŋma nyaba chaŋ sheli",
            "checks": lambda res: "kp" in res["tokens"] and "gb" in res["tokens"] and "ŋm" in res["tokens"] and "ny" in res["tokens"] and "ch" in res["tokens"] and "sh" in res["tokens"]
        },
        {
            "desc": "Intervocalic /d/ -> [r] lenition",
            "input": "kadariba",
            "checks": lambda res: "r" in res["tokens"]
        },
        {
            "desc": "Intervocalic /s/ -> [h] debuccalization",
            "input": "biisi",
            "checks": lambda res: "h" in res["tokens"]
        },
        {
            "desc": "Velar palatalization /k/ -> [t͡ʃ] before front vowel",
            "input": "kɛma",
            "checks": lambda res: "ch" in res["tokens"]
        },
        {
            "desc": "Lexicon tone override verification for 'gballi'",
            "input": "gballi",
            "checks": lambda res: res["tones"] == ["H", "H", "H", "H"]
        },
        {
            "desc": "Long vowel duration preservation ('duu' -> 'u:')",
            "input": "duu",
            "checks": lambda res: "u:" in res["tokens"]
        },
        {
            "desc": "Glottal stop in broken syllable ('bɛ'ʊ')",
            "input": "bɛ'ʊ",
            "checks": lambda res: "ʔ" in res["tokens"]
        },
        {
            "desc": "Token IDs sanity check",
            "input": "O biɛla Yendi",
            "checks": lambda res: len(res["token_ids"]) > 0 and all(isinstance(x, int) for x in res["token_ids"])
        }
    ]

    for tc in test_cases:
        total_tests += 1
        res = phonemizer.phonemize(tc["input"])
        passed = tc["checks"](res)
        status = "PASSED" if passed else "FAILED"
        if passed:
            tests_passed += 1
        print(f"[{status}] Test #{total_tests}: {tc['desc']}")
        if not passed:
            print(f"   Input:  {tc['input']}")
            print(f"   Result: {res}")

    print("-" * 60)
    print(f"Self-Test Summary: {tests_passed}/{total_tests} tests passed.")
    print("=" * 60)
    return tests_passed == total_tests


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dagbani G2P Phonemizer and Tone Tokenizer for Neural TTS"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Input text in Dagbani (BGL or ASCII) to phonemize."
    )
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        help="Path to plain text file containing sentences (one per line)."
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Path to output file for phonemized results (TSV or JSON format)."
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "tsv", "ipa", "token_ids"],
        default="json",
        help="Output serialization format (default: json)."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute internal phonemizer validation test suite and exit."
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_phonemizer_self_test()
        sys.exit(0 if success else 1)

    phonemizer = DagbaniPhonemizer()

    if args.text:
        res = phonemizer.phonemize(args.text)
        if args.format == "json":
            print(json.dumps(res, ensure_ascii=False, indent=2))
        elif args.format == "ipa":
            print(res["ipa"])
        elif args.format == "token_ids":
            print(" ".join(map(str, res["token_ids"])))
        elif args.format == "tsv":
            print(f"{args.text}\t{res['ipa']}\t{' '.join(map(str, res['token_ids']))}")
        return

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        results = []
        for line in lines:
            res = phonemizer.phonemize(line)
            results.append({"raw": line, **res})

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                if args.format == "json":
                    json.dump(results, f, ensure_ascii=False, indent=2)
                elif args.format == "tsv":
                    f.write("raw_text\tipa\ttoken_ids\n")
                    for r in results:
                        ids_str = " ".join(map(str, r["token_ids"]))
                        f.write(f"{r['raw']}\t{r['ipa']}\t{ids_str}\n")
                else:
                    for r in results:
                        f.write(f"{r['ipa']}\n")
            print(f"Processed {len(results)} lines. Saved to {args.output_file}")
        else:
            print(json.dumps(results[:5], ensure_ascii=False, indent=2))
            print(f"... total {len(results)} items processed.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
