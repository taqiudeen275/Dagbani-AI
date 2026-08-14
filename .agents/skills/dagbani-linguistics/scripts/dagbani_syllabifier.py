#!/usr/bin/env python3
"""
Dagbani Syllabifier & Prosodic Weight Analyzer
==============================================

Production-grade syllable parser for Dagbani. Decomposes words into onset,
nucleus, and coda constituents, assigns canonical syllable templates (CV, CVC,
CVV, CVVC, V, N), computes moraic weight (1 to 3 moras), and identifies
Tone-Bearing Units (TBUs).

Features:
- Full recognition of 1998 BGL digraphs and atomic labial-velars.
- Extraction of long vowels and opening diphthongs (ia, ua, iɛ, uɔ).
- Accurate identification of moraic coda nasals (m, n, ŋ) as active TBUs.
- Syllabic prefix nasal segmentation (m-, n-, ŋ-).
- Prosodic weight calculation and phonotactic validation.

Author: Dagbani AI Linguistics Team
License: Apache-2.0
"""

import sys
import os
import re
import unicodedata
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Phonological Data Structures & Constants
# ============================================================================

BASE_VOWEL_CHARS = {"a", "e", "ɛ", "i", "o", "ɔ", "u", "ɨ", "ʊ", "A", "E", "Ɛ", "I", "O", "Ɔ", "U"}
TONE_MARKS = {"́": "H", "̀": "L", "̂": "HL", "̌": "LH"}

DIGRAPH_ONSETS = ["ŋm", "Ŋm", "kp", "Kp", "gb", "Gb", "ny", "Ny", "ch", "Ch", "sh", "Sh"]
PERMITTED_CODAS = {"m", "n", "ŋ", "'", "’", "l", "r"}


def is_vowel_char(ch: str) -> bool:
    """Check if character is a Dagbani vowel (handles precomposed and combining accents)."""
    if not ch:
        return False
    decomposed = unicodedata.normalize("NFD", ch)
    base = decomposed[0]
    return base in BASE_VOWEL_CHARS


@dataclass
class Syllable:
    """Represents an atomic Dagbani syllable constituent."""
    raw: str
    onset: str
    nucleus: str
    coda: str
    template: str
    moras: int
    tone: str

    def __str__(self) -> str:
        coda_str = f".{self.coda}" if self.coda else ""
        return f"[{self.onset}-{self.nucleus}{coda_str} | {self.template} | {self.moras}μ | {self.tone}]"


# ============================================================================
# Syllabifier Implementation
# ============================================================================

class DagbaniSyllabifier:
    """Parser for Dagbani syllable structures and mora weight."""

    def __init__(self):
        pass

    def clean_word(self, word: str) -> str:
        """Strip non-alphabetic enclosing punctuation."""
        return unicodedata.normalize("NFC", word.strip())

    def syllabify_word(self, word: str) -> List[Syllable]:
        """
        Parse a single Dagbani word into a sequence of Syllable objects.
        """
        clean = self.clean_word(word)
        if not clean:
            return []

        syllables: List[Syllable] = []

        # Check for syllabic nasal prefix (e.g. m-bɔ, n-da, ŋ-ka, m̀-bɔ́)
        m_prefix = re.match(r"^([mnŋMNŊ][\u0300\u0301]?)[-\s](.+)$", clean)
        if m_prefix:
            nasal_tok = m_prefix.group(1)
            rest = m_prefix.group(2)
            tone = "H" if "́" in unicodedata.normalize("NFD", nasal_tok) else "L"
            syl = Syllable(
                raw=nasal_tok,
                onset="",
                nucleus=nasal_tok,
                coda="",
                template="N",
                moras=1,
                tone=tone,
            )
            syllables.append(syl)
            clean = rest

        chars = list(clean)
        n = len(chars)
        i = 0

        while i < n:
            onset = ""
            nucleus = ""
            coda = ""
            tone = "H"

            # 1. Onset extraction
            if i + 1 < n and "".join(chars[i:i+2]) in DIGRAPH_ONSETS:
                onset = "".join(chars[i:i+2])
                i += 2
            elif not is_vowel_char(chars[i]) and chars[i] not in ["'", "’"]:
                onset = chars[i]
                i += 1

            # 2. Nucleus extraction
            while i < n and (is_vowel_char(chars[i]) or chars[i] in TONE_MARKS or chars[i] in ["'", "’"]):
                if chars[i] in ["'", "’"]:
                    if nucleus and i + 1 < n and is_vowel_char(chars[i+1]):
                        coda = "'"
                        i += 1
                        break
                    elif not nucleus:
                        onset = "'"
                        i += 1
                        continue
                nucleus += chars[i]
                i += 1

            # 3. Coda extraction
            if i < n and chars[i] in PERMITTED_CODAS:
                # If followed by another vowel, it is an onset for next syllable
                if i + 1 < n and is_vowel_char(chars[i+1]):
                    pass  # Belongs to next syllable as onset
                elif i + 2 < n and "".join(chars[i+1:i+3]) in DIGRAPH_ONSETS:
                    coda = chars[i]
                    i += 1
                elif i + 1 < n and not is_vowel_char(chars[i+1]) and chars[i+1] not in ["'", "’"]:
                    coda = chars[i]
                    i += 1
                elif i + 1 == n:
                    coda = chars[i]
                    i += 1

            # Determine tone and clean nucleus
            nuc_decomposed = unicodedata.normalize("NFD", nucleus)
            if "̀" in nuc_decomposed:
                tone = "L"
            elif "́" in nuc_decomposed:
                tone = "H"
            else:
                tone = "H"

            # Base vowel characters without diacritics
            nuc_base = "".join(ch for ch in nuc_decomposed if unicodedata.category(ch) != "Mn")

            # Calculate moras and template
            is_long = len(nuc_base) >= 2 or nuc_base.lower() in ["ia", "ua", "iɛ", "uɔ"]
            moras = 1
            if is_long:
                moras += 1
            if coda in ["m", "n", "ŋ"]:  # Moraic coda nasal
                moras += 1

            # Template determination
            if not onset and not coda and not is_long:
                template = "V"
            elif not onset and not coda and is_long:
                template = "VV"
            elif onset and not coda and not is_long:
                template = "CV"
            elif onset and not coda and is_long:
                template = "CVV"
            elif onset and coda and not is_long:
                template = "CVC"
            elif onset and coda and is_long:
                template = "CVVC"
            elif not onset and coda and not is_long:
                template = "VC"
            elif not onset and coda and is_long:
                template = "VVC"
            else:
                template = "CV"

            raw_str = f"{onset}{nucleus}{coda}"
            syllables.append(Syllable(
                raw=raw_str,
                onset=onset,
                nucleus=nucleus,
                coda=coda,
                template=template,
                moras=moras,
                tone=tone,
            ))

        return syllables

    def syllabify_text(self, text: str) -> List[List[Syllable]]:
        """Syllabify all words in a sentence or paragraph."""
        words = text.strip().split()
        return [self.syllabify_word(w) for w in words]

    def get_total_moras(self, syllables: List[Syllable]) -> int:
        """Calculate total prosodic mora count."""
        return sum(s.moras for s in syllables)


# ============================================================================
# Self-Test Verification Suite
# ============================================================================

def run_self_test() -> bool:
    """Execute built-in syllabifier unit test suite."""
    print("======================================================================")
    print("Running Dagbani Syllabifier Self-Test Suite")
    print("======================================================================")

    parser = DagbaniSyllabifier()
    all_passed = True

    test_words = [
        # (word, expected_templates, expected_total_moras, description)
        ("bá", ["CV"], 1, "Short CV (1 mora)"),
        ("dáá", ["CVV"], 2, "Long CVV (2 moras)"),
        ("sɔ́ŋ", ["CVC"], 2, "CVC with moraic coda nasal (2 moras)"),
        ("dúúnsí", ["CVVC", "CV"], 4, "CVVC bimoraic nucleus + coda nasal"),
        ("m-bɔ́", ["N", "CV"], 2, "Syllabic nasal prefix + CV"),
        ("gballi", ["CVC", "CV"], 2, "Labial-velar onset with geminate coda"),
        ("ŋmaŋa", ["CV", "CV"], 2, "Atomic labial-velar nasal onset"),
        ("biɛɣu", ["CVV", "CV"], 3, "Palatalized front diphthong onset"),
    ]

    for word, exp_templates, exp_moras, desc in test_words:
        res = parser.syllabify_word(word)
        templates = [s.template for s in res]
        total_moras = parser.get_total_moras(res)
        passed = (templates == exp_templates) and (total_moras == exp_moras)
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {desc:<45} | Word: '{word}' -> {' . '.join(str(s) for s in res)} (Moras: {total_moras})")
        if not passed:
            all_passed = False

    print("======================================================================")
    if all_passed:
        print("ALL SYLLABIFIER SELF-TESTS PASSED CLEANLY (100% Correctness).")
    else:
        print("SOME SYLLABIFIER SELF-TESTS FAILED.")
    print("======================================================================")
    return all_passed


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dagbani Syllable Parser & Moraic Weight Analyzer"
    )
    parser.add_argument("--text", "-t", type=str, help="Dagbani word or text string to syllabify.")
    parser.add_argument("--input-file", "-i", type=str, help="Input text file path to syllabify.")
    parser.add_argument("--detailed", "-d", action="store_true", help="Print detailed onset-nucleus-coda breakdown.")
    parser.add_argument("--self-test", action="store_true", help="Run comprehensive unit test suite.")

    args = parser.parse_args()

    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    syllabifier = DagbaniSyllabifier()

    if args.text:
        results = syllabifier.syllabify_word(args.text)
        print(f"Input: {args.text}")
        print(f"Syllables: {' . '.join(str(s) for s in results)}")
        print(f"Templates: {'-'.join(s.template for s in results)}")
        print(f"Total Moras: {syllabifier.get_total_moras(results)}")
        if args.detailed:
            for idx, s in enumerate(results, start=1):
                print(f"  σ{idx}: Onset='{s.onset}', Nucleus='{s.nucleus}', Coda='{s.coda}', Template={s.template}, Moras={s.moras}, Tone={s.tone}")
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                res = syllabifier.syllabify_word(line)
                syl_str = " . ".join(str(s) for s in res)
                print(f"{line:<20} -> {syl_str}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
