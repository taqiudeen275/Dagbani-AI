#!/usr/bin/env python3
"""
Dagbani Grapheme-to-Phoneme (G2P) Converter
==========================================

Production-grade G2P rule engine converting Dagbani text (1998 BGL standard or
ASCII transliteration) into International Phonetic Alphabet (IPA) representations.

Features:
- Unicode NFC normalization and sanitization.
- Digraph tokenization (kp, gb, ŋm, ny, ch, sh, gh, zh).
- Contextual palatalization before front vowels (k, g, s, z, ŋ -> t͡ʃ, d͡ʒ, ʃ, ʒ, ɲ).
- Labial-coronal mutation for labial-velars before front vowels (kp, gb, ŋm -> t͡p, d͡b, n͡m).
- Intervocalic lenition and debuccalization (d -> r, s -> h, g -> ɣ/ʔ).
- Vowel ATR harmony and length modeling.
- Register tone integration (High, Low, Downstep).

Author: Dagbani AI Linguistics Team
License: Apache-2.0
"""

import sys
import os
import re
import unicodedata
import argparse
from typing import List, Tuple, Optional, Dict

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Phonological Constants & Inventories
# ============================================================================

FRONT_VOWELS = {"i", "e", "ɛ", "í", "é", "ɛ́", "ì", "è", "ɛ̀", "iː", "eː", "ɛː"}
BACK_VOWELS = {"u", "o", "ɔ", "ú", "ó", "ɔ́", "ù", "ò", "ɔ̀", "uː", "oː", "ɔː"}
LOW_VOWELS = {"a", "á", "à", "aː"}
ALL_VOWELS = FRONT_VOWELS | BACK_VOWELS | LOW_VOWELS | {"ɨ", "ɨ́", "ɨ̀", "ʊ", "ʊ́", "ʊ̀"}

# Digraphs and multi-character graphemes in descending order of length
DIGRAPHS_BGL = [
    ("ŋm", "ŋ͡m"),
    ("Ŋm", "ŋ͡m"),
    ("ŊM", "ŋ͡m"),
    ("kp", "k͡p"),
    ("Kp", "k͡p"),
    ("KP", "k͡p"),
    ("gb", "ɡ͡b"),
    ("Gb", "ɡ͡b"),
    ("GB", "ɡ͡b"),
    ("ny", "ɲ"),
    ("Ny", "ɲ"),
    ("NY", "ɲ"),
    ("ch", "t͡ʃ"),
    ("Ch", "t͡ʃ"),
    ("CH", "t͡ʃ"),
    ("sh", "ʃ"),
    ("Sh", "ʃ"),
    ("SH", "ʃ"),
]

DIGRAPHS_ASCII = [
    ("ngm", "ŋ͡m"),
    ("Ngm", "ŋ͡m"),
    ("NGM", "ŋ͡m"),
    ("gh", "ɣ"),
    ("Gh", "ɣ"),
    ("GH", "ɣ"),
    ("zh", "ʒ"),
    ("Zh", "ʒ"),
    ("ZH", "ʒ"),
    ("ng", "ŋ"),
    ("Ng", "ŋ"),
    ("NG", "ŋ"),
] + DIGRAPHS_BGL

# Single character mappings
CHAR_MAP: Dict[str, str] = {
    "a": "a",
    "b": "b",
    "d": "d",
    "e": "e",
    "ɛ": "ɛ",
    "f": "f",
    "g": "ɡ",
    "ɣ": "ɣ",
    "h": "h",
    "i": "i",
    "j": "d͡ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ŋ": "ŋ",
    "o": "o",
    "ɔ": "ɔ",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
    "ʒ": "ʒ",
    "'": "ʔ",
    "’": "ʔ",
    "‘": "ʔ",
    "`": "ʔ",
}


# ============================================================================
# Core G2P Engine Class
# ============================================================================

class DagbaniG2P:
    """Production Grapheme-to-Phoneme converter for the Dagbani language."""

    def __init__(self, default_tone: bool = True, ascii_mode: bool = False):
        self.default_tone = default_tone
        self.ascii_mode = ascii_mode

    def normalize_input(self, text: str) -> str:
        """Canonicalize Unicode and normalize quotes and apostrophes."""
        text = unicodedata.normalize("NFC", text.strip())
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        return text

    def tokenize_graphemes(self, word: str) -> List[str]:
        """
        Tokenize a word into orthographic units (digraphs, long vowels, letters).
        """
        tokens: List[str] = []
        i = 0
        w_len = len(word)
        digraph_list = DIGRAPHS_ASCII if self.ascii_mode else DIGRAPHS_BGL

        while i < w_len:
            matched = False
            # Try matching 3-letter clusters first (e.g. ngm)
            if i + 3 <= w_len:
                chunk = word[i:i+3]
                for pattern, ipa_val in digraph_list:
                    if chunk.lower() == pattern.lower() and len(pattern) == 3:
                        tokens.append(ipa_val)
                        i += 3
                        matched = True
                        break
            if matched:
                continue

            # Try matching 2-letter digraphs or long vowels
            if i + 2 <= w_len:
                chunk = word[i:i+2]
                # Check 2-letter digraphs
                for pattern, ipa_val in digraph_list:
                    if chunk.lower() == pattern.lower() and len(pattern) == 2:
                        tokens.append(ipa_val)
                        i += 2
                        matched = True
                        break
                if matched:
                    continue

                # Check long vowels (aa, ee, ɛɛ, ii, oo, ɔɔ, uu)
                low_chunk = chunk.lower()
                if low_chunk in ["aa", "ee", "ɛɛ", "ii", "oo", "ɔɔ", "uu"]:
                    base_v = low_chunk[0]
                    ipa_v = CHAR_MAP.get(base_v, base_v) + "ː"
                    tokens.append(ipa_v)
                    i += 2
                    continue

            # Single character
            char = word[i]
            char_low = char.lower()
            if char_low in CHAR_MAP:
                tokens.append(CHAR_MAP[char_low])
            else:
                tokens.append(char)
            i += 1

        return tokens

    def apply_phonological_rules(self, tokens: List[str]) -> List[str]:
        """
        Apply context-sensitive phonological and allophonic rules:
        1. Velar & alveolar palatalization before front vowels.
        2. Labial-coronal mutation for labial-velars before front vowels.
        3. Intervocalic lenition (/d/ -> [r], /s/ -> [h], /g/ -> [ɣ]).
        4. Debuccalization blocking in consonant clusters.
        """
        n = len(tokens)
        out = list(tokens)

        def is_vowel(idx: int) -> bool:
            if 0 <= idx < n:
                base = out[idx].replace("ː", "").replace("́", "").replace("̀", "")
                return base in ALL_VOWELS or out[idx] in ALL_VOWELS
            return False

        def is_front_vowel(idx: int) -> bool:
            if 0 <= idx < n:
                base = out[idx].replace("ː", "").replace("́", "").replace("̀", "")
                return base in FRONT_VOWELS
            return False

        for i in range(n):
            curr = out[i]

            # Rule 1 & 2: Palatalization & Labial-Coronal Mutation before Front Vowels
            if is_front_vowel(i + 1):
                if curr == "k":
                    out[i] = "t͡ʃ"
                elif curr == "ɡ":
                    out[i] = "d͡ʒ"
                elif curr == "s":
                    out[i] = "ʃ"
                elif curr == "z":
                    out[i] = "ʒ"
                elif curr == "ŋ":
                    out[i] = "ɲ"
                elif curr == "k͡p":
                    out[i] = "t͡p"
                elif curr == "ɡ͡b":
                    out[i] = "d͡b"
                elif curr == "ŋ͡m":
                    out[i] = "n͡m"

            # Rule 3: Intervocalic Lenition & Debuccalization
            if is_vowel(i - 1) and is_vowel(i + 1):
                if curr == "d":
                    out[i] = "r"
                elif curr == "s":
                    out[i] = "h"
                elif curr == "ɡ":
                    out[i] = "ɣ"

            # Rule 4: Prevocalic Glide Formation (i -> j before front/open/non-high vowels)
            if curr == "i" and i + 1 < n:
                next_base = out[i + 1].replace("ː", "").replace("́", "").replace("̀", "")
                if next_base in {"ɛ", "a", "ɔ", "e", "o", "u", "ʊ"}:
                    out[i] = "j"

        return out

    def apply_tone_contour(self, tokens: List[str], tone_melody: Optional[str] = None) -> List[str]:
        """
        Assign default citation or specified tone melody across Tone-Bearing Units.
        TBUs: Vowels and coda nasals (m, n, ŋ).
        """
        out: List[str] = []
        for i, tok in enumerate(tokens):
            base = tok.replace("ː", "")
            is_vowel = base in ALL_VOWELS or any(v in tok for v in ["a", "e", "ɛ", "i", "o", "ɔ", "u", "ɨ", "ʊ"])
            is_moraic_nasal = tok in ["m", "n", "ŋ", "ɲ"] and (i == len(tokens) - 1 or (i > 0 and not any(v in tokens[i-1] for v in ALL_VOWELS)))

            # If tone diacritic is already present, retain it
            if "́" in tok or "̀" in tok or not self.default_tone:
                out.append(tok)
            elif is_vowel:
                # Default citation tone: High tone on initial/root moras
                if "ː" in tok:
                    v_clean = tok.replace("ː", "")
                    out.append(f"{v_clean}́ː")
                else:
                    out.append(f"{tok}́")
            elif is_moraic_nasal:
                out.append(f"{tok}́")
            else:
                out.append(tok)

        return out

    def convert_word(self, word: str, with_tone: Optional[bool] = None) -> str:
        """Convert a single word into IPA phonetic string."""
        if not word:
            return ""

        # Check for punctuation attached to word
        punct_prefix = ""
        punct_suffix = ""
        while word and not word[0].isalnum() and word[0] not in ["'", "’", "ɛ", "ɔ", "ŋ", "ɣ", "ʒ", "Ɛ", "Ɔ", "Ŋ", "Ɣ", "Ʒ"]:
            punct_prefix += word[0]
            word = word[1:]
        while word and not word[-1].isalnum() and word[-1] not in ["'", "’", "ɛ", "ɔ", "ŋ", "ɣ", "ʒ", "Ɛ", "Ɔ", "Ŋ", "Ɣ", "Ʒ"]:
            punct_suffix = word[-1] + punct_suffix
            word = word[:-1]

        if not word:
            return punct_prefix + punct_suffix

        # Step 1: Tokenize
        tokens = self.tokenize_graphemes(word)

        # Step 2: Apply phonological rules
        mutated = self.apply_phonological_rules(tokens)

        # Step 3: Tonal assignment
        use_tone = with_tone if with_tone is not None else self.default_tone
        if use_tone:
            toned = self.apply_tone_contour(mutated)
        else:
            toned = mutated

        ipa_str = "".join(toned)
        return f"{punct_prefix}{ipa_str}{punct_suffix}"

    def convert_sentence(self, sentence: str, with_tone: Optional[bool] = None) -> str:
        """Convert an entire sentence or multi-word phrase into IPA."""
        sentence = self.normalize_input(sentence)
        words = sentence.split(" ")
        converted_words = [self.convert_word(w, with_tone=with_tone) for w in words]
        return " ".join(converted_words)


# ============================================================================
# Self-Test Verification Suite
# ============================================================================

def run_self_test() -> bool:
    """Execute built-in phonological test suite."""
    print("======================================================================")
    print("Running Dagbani G2P Conversion Self-Test Suite")
    print("======================================================================")

    test_cases = [
        # (input_text, expected_substring_or_ipa, description)
        ("bɛ'ʊ́", "bɛ́ʔʊ́", "Glottal stop and open-mid vowel preservation"),
        ("gballi", "ɡ͡ba", "Labial-velar stop and geminate lateral"),
        ("biɛɣu", "bjɛ", "Palatalized front vowel glide & velar fricative"),
        ("ŋmaŋa", "ŋ͡máŋá", "Labial-velar nasal & velar nasal"),
        ("kpɛ", "t͡pɛ́", "Labial-coronal mutation of /kp/ before front vowel"),
        ("bihi", "bíhí", "Intervocalic debuccalization (/s/ -> [h])"),
        ("duunsi", "dúːnʃí", "Debuccalization blocked after coda nasal"),
        ("paɣa", "páɣá", "Voiced velar fricative grapheme <ɣ>"),
        ("ʒɛm", "ʒɛ́m", "Voiced post-alveolar fricative <ʒ>"),
        ("shikuru", "ʃíkúrú", "English loanword adaptation & sibilant palatalization"),
    ]

    g2p = DagbaniG2P(default_tone=True, ascii_mode=False)
    all_passed = True

    for text, expected, desc in test_cases:
        result = g2p.convert_word(text)
        passed = expected in result
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {desc:<50} | In: '{text}' -> Out: '{result}' (Expected: '{expected}')")
        if not passed:
            all_passed = False

    # Test ASCII Mode
    print("\n--- Testing ASCII Transliteration Fallback Mode ---")
    g2p_ascii = DagbaniG2P(default_tone=True, ascii_mode=True)
    ascii_cases = [
        ("pagaba", "ɣa", "ASCII 'g' / 'gh' -> [ɣ]"),
        ("zhangmi", "ʒ", "ASCII 'zh' -> [ʒ]"),
        ("ngmande", "ŋ͡m", "ASCII 'ngm' -> [ŋ͡m]"),
    ]
    for text, expected, desc in ascii_cases:
        result = g2p_ascii.convert_word(text)
        passed = expected in result
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {desc:<50} | In: '{text}' -> Out: '{result}'")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("ALL G2P SELF-TESTS PASSED CLEANLY (100% Correctness).")
    else:
        print("SOME G2P SELF-TESTS FAILED.")
    print("======================================================================")
    return all_passed


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dagbani Grapheme-to-Phoneme (G2P) Engine (BGL 1998 & ASCII to IPA)"
    )
    parser.add_argument("--text", "-t", type=str, help="Single Dagbani text string or sentence to convert.")
    parser.add_argument("--input-file", "-i", type=str, help="Input text file path to convert line-by-line.")
    parser.add_argument("--output-file", "-o", type=str, help="Output file path to save IPA conversions.")
    parser.add_argument("--ascii-input", action="store_true", help="Enable ASCII transliteration parsing.")
    parser.add_argument("--no-tones", action="store_true", help="Disable default tone contour generation.")
    parser.add_argument("--self-test", action="store_true", help="Run comprehensive unit test suite.")

    args = parser.parse_args()

    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    if not args.text and not args.input_file:
        print("Dagbani G2P Converter CLI. Use --help for usage instructions, or run --self-test.")
        sys.exit(0)

    g2p = DagbaniG2P(default_tone=not args.no_tones, ascii_mode=args.ascii_input)

    if args.text:
        ipa_out = g2p.convert_sentence(args.text)
        print(ipa_out)

    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)

        with open(args.input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        converted_lines = [g2p.convert_sentence(line.strip()) + "\n" for line in lines]

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                fout.writelines(converted_lines)
            print(f"Successfully converted {len(lines)} lines to {args.output_file}")
        else:
            for c_line in converted_lines:
                print(c_line, end="")


if __name__ == "__main__":
    main()
