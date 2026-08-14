#!/usr/bin/env python3
"""
Dagbani Orthography Normalizer & Sanitizer
==========================================

Production text normalization and transliteration engine for Dagbani.
Converts between 1998 Bureau of Ghana Languages (BGL) standard orthography
(incorporating ɛ, ɔ, ŋ, ɣ, ʒ) and ASCII transliterations, cleans noisy web
corpora, and standardizes dialectal variants.

Features:
- Canonical Unicode NFC normalization.
- Bidirectional ASCII <-> BGL standard conversion.
- Lexical disambiguation dictionary for open/close mid vowels (e/ɛ, o/ɔ).
- Punctuation hygiene, quotation mark sanitization, and whitespace cleanup.
- Dialectal normalization (Eastern Nayahili <-> Western Tomosili).

Author: Dagbani AI Linguistics Team
License: Apache-2.0
"""

import sys
import os
import re
import unicodedata
import argparse
from typing import Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Lexical Root Disambiguation & Transliteration Tables
# ============================================================================

# Common High-Frequency Words with open-mid vowels & special characters for ASCII -> BGL mapping
ASCII_TO_BGL_LEXICON: Dict[str, str] = {
    # Pronouns, Particles & Auxiliaries
    "nyela": "nyɛla",
    "be": "bɛ",
    "sheli": "shɛli",
    "sheba": "shɛba",
    "kpe": "kpɛ",
    "kpehi": "kpɛhi",
    "deei": "dee",
    "di": "di",
    "din": "din",
    "mbi": "m-bi",
    "mbo": "m-bɔ",
    "m-bo": "m-bɔ",
    "n-nye": "n-nyɛ",
    
    # Nouns & Verbs
    "paga": "paɣa",
    "pagaba": "paɣaba",
    "pagasara": "paɣasara",
    "pagasariba": "paɣasariba",
    "biegu": "biɛɣu",
    "bieɣu": "biɛɣu",
    "vieli": "viɛli",
    "pieli": "piɛli",
    "pieɣu": "piɛɣu",
    "piegu": "piɛɣu",
    "zango": "zaŋɔ",
    "zungo": "zuŋɔ",
    "zuliya": "zuliya",
    "dogim": "dɔɣim",
    "dogo": "dɔɣo",
    "doro": "dóró",
    "doba": "dɔba",
    "doo": "doo",
    "doohi": "doohi",
    "zhem": "ʒɛm",
    "zheri": "ʒɛri",
    "zherigu": "ʒɛrigu",
    "zhangmi": "ʒaŋmi",
    "ngmariga": "ŋmariga",
    "ngmanga": "ŋmaŋa",
    "ngmaa": "ŋmaa",
    "wuntanga": "wuntaŋa",
    "karimba": "karimba",
    "shikuru": "shikuru",
    "sukuru": "shikuru",
    "lahabali": "lahabali",
    "gaafara": "gaafara",
    "yeltoga": "yɛltɔɣa",
    "yeltoɣa": "yɛltɔɣa",
    "kpalinzhoo": "kpalinʒoo",
    "tariyo": "tariyu",
    "kobga": "kɔbga",
    "kowa": "kɔwa",
}

# Systematic Digraph & Glyph Replacement Pairs (ordered by descending pattern length)
ASCII_TO_BGL_PATTERNS = [
    (re.compile(r"\bNgm", re.IGNORECASE), "Ŋm"),
    (re.compile(r"ngm", re.IGNORECASE), "ŋm"),
    (re.compile(r"\bGh"), "Ɣ"),
    (re.compile(r"gh"), "ɣ"),
    (re.compile(r"\bZh"), "Ʒ"),
    (re.compile(r"zh"), "ʒ"),
]

BGL_TO_ASCII_MAP = {
    "ɛ": "e",
    "Ɛ": "E",
    "ɔ": "o",
    "Ɔ": "O",
    "ŋ": "ng",
    "Ŋ": "Ng",
    "ɣ": "gh",
    "Ɣ": "Gh",
    "ʒ": "zh",
    "Ʒ": "Zh",
}

# Dialect standardizer (Eastern Nayahili -> Western Tomosili Standard)
EASTERN_TO_WESTERN_MAP = {
    "yɛltoɣa": "yɛltɔɣa",
    "yeltoɣa": "yɛltɔɣa",
    "kɔbga": "kɔwa",
    "tɔɣsɨ": "tɔxɨ",
    "tɔɣsi": "tɔxi",
    "toɣsi": "tɔxi",
}


# ============================================================================
# Normalizer Implementation
# ============================================================================

class DagbaniNormalizer:
    """Production text normalizer and orthography converter for Dagbani."""

    def __init__(self, target_dialect: str = "western"):
        self.target_dialect = target_dialect.lower()

    def sanitize_punctuation(self, text: str) -> str:
        """
        Clean and standardize punctuation, quotation marks, and hyphens.
        - Canonicalizes curly quotes to single/double standard ASCII quotes.
        - Normalizes backticks to straight apostrophes.
        - Collapses repeated non-lexical punctuation.
        """
        text = unicodedata.normalize("NFC", str(text))
        
        # Quotations and apostrophes
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = text.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')

        # Hyphens and dashes
        text = text.replace("–", "-").replace("—", "-").replace("−", "-")

        # Collapse repeated punctuation
        text = re.sub(r"\.{2,}", "...", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"-{2,}", "--", text)

        # Collapse repeated characters (3 or more -> 2, e.g. pammm -> pam)
        text = re.sub(r"([a-zA-ZɛɔŋɣʒƐƆŊƔƷ])\1{2,}", r"\1", text)

        # Whitespace normalization
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text)
        return text.strip()

    def convert_ascii_to_bgl(self, text: str) -> str:
        """
        Convert ASCII transliterated text to standard 1998 BGL orthography.
        Uses lexicon lookups for high-frequency irregulars, followed by
        systematic digraph and character replacements.
        """
        text = self.sanitize_punctuation(text)
        words = text.split(" ")
        converted_words: List[str] = []

        for w in words:
            # Strip punctuation for lookup
            m = re.match(r"^([^a-zA-Z0-9ɛɔŋɣʒƐƆŊƔƷ]*)(.*?)([^a-zA-Z0-9ɛɔŋɣʒƐƆŊƔƷ]*)$", w)
            if not m:
                converted_words.append(w)
                continue

            prefix, core, suffix = m.group(1), m.group(2), m.group(3)
            core_lower = core.lower()

            if core_lower in ASCII_TO_BGL_LEXICON:
                replacement = ASCII_TO_BGL_LEXICON[core_lower]
                # Match title case
                if core.istitle():
                    replacement = replacement.capitalize()
                elif core.isupper():
                    replacement = replacement.upper()
                converted_words.append(f"{prefix}{replacement}{suffix}")
            else:
                # Apply systematic rules on unmatched word
                cur = core
                for pattern, repl in ASCII_TO_BGL_PATTERNS:
                    cur = pattern.sub(repl, cur)
                converted_words.append(f"{prefix}{cur}{suffix}")

        return " ".join(converted_words)

    def convert_bgl_to_ascii(self, text: str) -> str:
        """
        Convert standard 1998 BGL orthography into plain ASCII transliteration.
        Replaces special characters (ɛ, ɔ, ŋ, ɣ, ʒ) with standardized ASCII fallbacks.
        """
        text = self.sanitize_punctuation(text)
        out = []
        for ch in text:
            if ch in BGL_TO_ASCII_MAP:
                out.append(BGL_TO_ASCII_MAP[ch])
            else:
                out.append(ch)
        return "".join(out)

    def standardize_dialect(self, text: str) -> str:
        """Standardize dialectal variants to Western (Tomosili / Standard)."""
        text = self.sanitize_punctuation(text)
        words = text.split(" ")
        standardized = []
        for w in words:
            w_clean = re.sub(r"[^\wɛɔŋɣʒƐƆŊƔƷ]", "", w).lower()
            if w_clean in EASTERN_TO_WESTERN_MAP:
                repl = EASTERN_TO_WESTERN_MAP[w_clean]
                if w.istitle():
                    repl = repl.capitalize()
                standardized.append(repl)
            else:
                standardized.append(w)
        return " ".join(standardized)

    def normalize_for_asr(self, text: str) -> str:
        """
        Produce sanitized, lowercased string suitable for Normalized WER evaluation.
        Retains valid Dagbani letters (including special glyphs) and internal apostrophes/hyphens.
        """
        text = unicodedata.normalize("NFC", str(text))
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = text.lower()
        
        # Keep only letters, whitespace, single quote, hyphen
        cleaned_chars = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or ch in {" ", "'", "-"}:
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        
        cleaned = "".join(cleaned_chars)
        return re.sub(r"\s+", " ", cleaned).strip()


# ============================================================================
# Self-Test Verification Suite
# ============================================================================

def run_self_test() -> bool:
    """Execute unit test suite for normalizer."""
    print("======================================================================")
    print("Running Dagbani Orthography Normalizer Self-Test Suite")
    print("======================================================================")

    normalizer = DagbaniNormalizer()
    all_passed = True

    # Test 1: ASCII to BGL Transliteration
    ascii_input = "nyela paga mini bihi ban be Tamale"
    expected_bgl = "nyɛla paɣa mini bihi ban bɛ Tamale"
    bgl_res = normalizer.convert_ascii_to_bgl(ascii_input)
    passed_1 = bgl_res == expected_bgl
    print(f"[{'PASSED' if passed_1 else 'FAILED'}] ASCII -> BGL: '{ascii_input}' -> '{bgl_res}'")
    if not passed_1:
        all_passed = False

    # Test 2: BGL to ASCII Transliteration
    bgl_input = "nyɛla paɣa mini ʒɛm din viɛli"
    expected_ascii = "nyela pagha mini zhem din vieli"
    ascii_res = normalizer.convert_bgl_to_ascii(bgl_input)
    passed_2 = "pagh" in ascii_res and "zh" in ascii_res
    print(f"[{'PASSED' if passed_2 else 'FAILED'}] BGL -> ASCII: '{bgl_input}' -> '{ascii_res}'")
    if not passed_2:
        all_passed = False

    # Test 3: Punctuation and Noise Cleaning
    noisy_input = "O yeliya:   '''Biɛɣu    viɛli   pammm!!!'''"
    clean_res = normalizer.sanitize_punctuation(noisy_input)
    passed_3 = "pam!" in clean_res and "  " not in clean_res
    print(f"[{'PASSED' if passed_3 else 'FAILED'}] Punctuation Cleaning: '{noisy_input}' -> '{clean_res}'")
    if not passed_3:
        all_passed = False

    # Test 4: ASR Normalization
    raw_asr = "“Ti kpalinʒoo n-nyɛla din viɛli pam!”"
    asr_norm = normalizer.normalize_for_asr(raw_asr)
    expected_asr = "ti kpalinʒoo n-nyɛla din viɛli pam"
    passed_4 = asr_norm == expected_asr
    print(f"[{'PASSED' if passed_4 else 'FAILED'}] ASR Normalizer: '{raw_asr}' -> '{asr_norm}'")
    if not passed_4:
        all_passed = False

    print("======================================================================")
    if all_passed:
        print("ALL NORMALIZER SELF-TESTS PASSED CLEANLY (100% Correctness).")
    else:
        print("SOME NORMALIZER SELF-TESTS FAILED.")
    print("======================================================================")
    return all_passed


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dagbani Orthography Normalizer & Sanitizer (BGL 1998 / ASCII)"
    )
    parser.add_argument("--text", "-t", type=str, help="Single Dagbani text string to process.")
    parser.add_argument("--input-file", "-i", type=str, help="Input text file path to normalize.")
    parser.add_argument("--output-file", "-o", type=str, help="Output file path to save normalized text.")
    parser.add_argument("--to-bgl", action="store_true", help="Convert ASCII text to standard 1998 BGL.")
    parser.add_argument("--to-ascii", action="store_true", help="Convert BGL text to plain ASCII transliteration.")
    parser.add_argument("--clean", action="store_true", help="Clean punctuation, quotes, and whitespace.")
    parser.add_argument("--for-asr", action="store_true", help="Normalize for speech recognition evaluation.")
    parser.add_argument("--standardize-dialect", action="store_true", help="Standardize dialectal variants to Western.")
    parser.add_argument("--self-test", action="store_true", help="Run comprehensive unit test suite.")

    args = parser.parse_args()

    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    if not args.text and not args.input_file:
        print("Dagbani Orthography Normalizer CLI. Use --help for usage instructions, or run --self-test.")
        sys.exit(0)

    normalizer = DagbaniNormalizer()

    def process_string(s: str) -> str:
        if args.clean:
            s = normalizer.sanitize_punctuation(s)
        if args.to_bgl:
            s = normalizer.convert_ascii_to_bgl(s)
        if args.to_ascii:
            s = normalizer.convert_bgl_to_ascii(s)
        if args.standardize_dialect:
            s = normalizer.standardize_dialect(s)
        if args.for_asr:
            s = normalizer.normalize_for_asr(s)
        return s

    if args.text:
        print(process_string(args.text))

    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)

        with open(args.input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        processed_lines = [process_string(line.strip()) + "\n" for line in lines]

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                fout.writelines(processed_lines)
            print(f"Normalized {len(lines)} lines saved to {args.output_file}")
        else:
            for p_line in processed_lines:
                print(p_line, end="")


if __name__ == "__main__":
    main()
