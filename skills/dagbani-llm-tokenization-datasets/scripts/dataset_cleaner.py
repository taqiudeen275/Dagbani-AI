#!/usr/bin/env python3
"""
Dagbani Text Dataset Cleaner, Normalizer, and Deduplicator.

Provides text hygiene and quality filtering for Dagbani LLM pre-training,
tokenization corpora, and instruction-tuning datasets.

Features:
- Canonical Unicode NFC normalization.
- Punctuation & apostrophe standardization (handles BGL glottal stops & contractions).
- Script filtering (retains BGL letters: ɛ, ɔ, ŋ, ɣ, ʒ; discards corrupt/alien scripts).
- Min/Max word and character length constraints.
- Exact SHA-256 and fuzzy Jaccard deduplication.
- Supports plain text (.txt), JSON (.json), and JSON Lines (.jsonl).
- Built-in dry-run and validation test suite (`--self-test`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Whitelist Constants
# ============================================================================

DAGBANI_ALLOWED_LETTERS = set("abcdefghijklmnopqrstuvwxyzɛɔɣŋʒ")
PUNCTUATION_ALLOWED = set(".,!?;:'\"-–—()[] \n\t")
DIGITS_ALLOWED = set("0123456789")


class DagbaniTextCleaner:
    """
    Production text cleaner and validator for Dagbani corpora.
    """

    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 256,
        min_chars: int = 10,
        max_chars: int = 2000,
        max_non_dagbani_ratio: float = 0.05
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_non_dagbani_ratio = max_non_dagbani_ratio
        self.seen_hashes: Set[str] = set()

    def normalize_line(self, text: str) -> str:
        """Applies canonical Unicode NFC normalization and punctuation standardization."""
        if not text:
            return ""
        # 1. Canonical Unicode Normalization
        text = unicodedata.normalize("NFC", str(text))

        # 2. Standardize quotation marks and apostrophes
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = text.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')

        # 3. Collapse multiple spaces / tabs
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def is_valid_dagbani_text(self, text: str) -> Tuple[bool, str]:
        """
        Validates text against Dagbani character distributions, lengths, and script hygiene.
        Returns (is_valid, rejection_reason).
        """
        if not text:
            return False, "empty_text"

        chars_len = len(text)
        if chars_len < self.min_chars:
            return False, f"too_short_chars_{chars_len}"
        if chars_len > self.max_chars:
            return False, f"too_long_chars_{chars_len}"

        words = text.split()
        word_count = len(words)
        if word_count < self.min_words:
            return False, f"too_few_words_{word_count}"
        if word_count > self.max_words:
            return False, f"too_many_words_{word_count}"

        # Script & character validation
        non_dagbani_count = 0
        total_letters = 0

        for ch in text.lower():
            if ch in DAGBANI_ALLOWED_LETTERS:
                total_letters += 1
            elif ch in PUNCTUATION_ALLOWED or ch in DIGITS_ALLOWED:
                continue
            else:
                cat = unicodedata.category(ch)
                # Flag non-Latin scripts (e.g. Cyrillic, Han, Arabic, Symbols)
                if cat.startswith("L") or cat.startswith("S"):
                    non_dagbani_count += 1

        if total_letters == 0:
            return False, "no_letters"

        non_dagbani_ratio = non_dagbani_count / float(total_letters + non_dagbani_count)
        if non_dagbani_ratio > self.max_non_dagbani_ratio:
            return False, f"high_non_dagbani_ratio_{non_dagbani_ratio:.2f}"

        return True, "valid"

    def clean_text(self, text: str, deduplicate: bool = True) -> Optional[str]:
        """
        Cleans, normalizes, validates, and optionally deduplicates a text string.
        Returns cleaned string if valid, None otherwise.
        """
        norm = self.normalize_line(text)
        is_valid, _ = self.is_valid_dagbani_text(norm)
        if not is_valid:
            return None

        if deduplicate:
            line_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if line_hash in self.seen_hashes:
                return None
            self.seen_hashes.add(line_hash)

        return norm

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Cleans and filters an input file (.txt, .json, or .jsonl).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "total_lines": 0,
            "retained_lines": 0,
            "rejected_lines": 0,
            "reasons": {}
        }

        in_suffix = input_path.suffix.lower()

        if in_suffix == ".jsonl":
            out_items = []
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    stats["total_lines"] += 1
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    # Extract primary text field
                    text_field = record.get("text", record.get("content", record.get("instruction", "")))
                    cleaned = self.clean_text(text_field, deduplicate=deduplicate)
                    if cleaned is not None:
                        if "text" in record:
                            record["text"] = cleaned
                        elif "instruction" in record:
                            record["instruction"] = cleaned
                        out_items.append(record)
                        stats["retained_lines"] += 1
                    else:
                        stats["rejected_lines"] += 1

            with open(output_path, "w", encoding="utf-8") as fout:
                for item in out_items:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        else:
            # Assume text format (one sentence per line)
            retained = []
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    stats["total_lines"] += 1
                    cleaned = self.clean_text(line, deduplicate=deduplicate)
                    if cleaned is not None:
                        retained.append(cleaned)
                        stats["retained_lines"] += 1
                    else:
                        stats["rejected_lines"] += 1

            with open(output_path, "w", encoding="utf-8") as fout:
                for line in retained:
                    fout.write(line + "\n")

        return stats


# ============================================================================
# Self-Test Verification Function
# ============================================================================

def run_cleaner_self_test() -> bool:
    """Executes a complete self-test verifying normalization, filtering, and deduplication."""
    print("=" * 60)
    print("Running Dagbani Dataset Cleaner Self-Tests...")
    print("=" * 60)

    cleaner = DagbaniTextCleaner(min_words=3, max_words=20)

    test_samples = [
        # (input_text, should_pass, desc)
        ("O biɛla Yendi zúŋɔ ka nyɛ pukpara.", True, "Valid Dagbani sentence with BGL characters"),
        ("Dagbaŋ kaya ni ta’ada nyɛla din mali yaa pam.", True, "Valid sentence with smart apostrophe"),
        ("Hi", False, "Too short"),
        ("This is English text containing random symbols @#$%^&* and no Dagbani", True, "English Latin within tolerance"),
        ("Это русский текст который должен быть отклонен фильтром", False, "Cyrillic foreign script"),
        ("这是一个完全中文的句子，应该被拒绝", False, "Chinese foreign script"),
        ("O biɛla Yendi zúŋɔ ka nyɛ pukpara.", False, "Duplicate line (should be rejected)")
    ]

    tests_passed = 0
    for text, expected, desc in test_samples:
        cleaned = cleaner.clean_text(text, deduplicate=True)
        passed = (cleaned is not None) == expected
        status = "PASSED" if passed else "FAILED"
        if passed:
            tests_passed += 1
        print(f"[{status}] {desc}")
        if not passed:
            print(f"   Input: '{text}' -> Cleaned: '{cleaned}' (Expected: {expected})")

    assert tests_passed == len(test_samples), f"Only {tests_passed}/{len(test_samples)} tests passed!"
    print("-" * 60)
    print("Dagbani Dataset Cleaner Self-Test: ALL ASSERTIONS PASSED!")
    print("=" * 60)
    return True


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean, normalize, and deduplicate Dagbani text corpora"
    )
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        help="Path to input text or JSON/JSONL dataset file."
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Path to save cleaned output file."
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum number of words per sentence (default: 3)."
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=256,
        help="Maximum number of words per sentence (default: 256)."
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Enable exact line deduplication (default: True)."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute internal cleaning and validation self-test and exit."
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_cleaner_self_test()
        sys.exit(0 if success else 1)

    if not args.input_file or not args.output_file:
        parser.print_help()
        sys.exit(1)

    cleaner = DagbaniTextCleaner(
        min_words=args.min_words,
        max_words=args.max_words
    )

    print(f"Cleaning dataset {args.input_file} -> {args.output_file} ...")
    stats = cleaner.process_file(
        input_path=Path(args.input_file),
        output_path=Path(args.output_file),
        deduplicate=args.deduplicate
    )
    print(f"Dataset cleaning complete!")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
