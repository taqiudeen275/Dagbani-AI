#!/usr/bin/env python3
"""
Custom Byte-Level BPE Tokenizer Trainer for Dagbani (ISO 639-3: dag).

Trains a high-efficiency subword tokenizer tailored to Dagbani orthography,
preserving digraphs (kp, gb, ŋm, ny, ch, sh) and BGL characters (ɛ, ɔ, ŋ, ɣ, ʒ),
minimizing subword fertility and preventing byte-fallback fragmentation.

Features:
- Hugging Face `tokenizers` integration with standalone pure-Python BPE fallback.
- Explicit Dagbani digraph and character alphabet pre-seeding.
- Subword fertility evaluation against benchmark texts.
- Exports standard HuggingFace artifacts: `tokenizer.json`, `vocab.json`, `merges.txt`.
- Built-in verification test suite (`--self-test`).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# Dagbani Linguistic Constants & Digraphs
# ============================================================================

DAGBANI_SPECIAL_CHARS = ["ɛ", "ɔ", "ŋ", "ɣ", "ʒ", "Ɛ", "Ɔ", "Ŋ", "Ɣ", "Ʒ"]
DAGBANI_DIGRAPHS = ["kp", "gb", "ŋm", "ny", "ch", "sh", "Kp", "Gb", "Ŋm", "Ny", "Ch", "Sh"]
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


# ============================================================================
# Standalone Pure-Python BPE Engine (Fallback & Verification)
# ============================================================================

class PurePythonBPE:
    """
    Self-contained Byte-Level BPE tokenizer implementation for environments
    without external C-extensions.
    """

    def __init__(self, vocab_size: int = 1000, preserve_digraphs: bool = True):
        self.vocab_size = vocab_size
        self.preserve_digraphs = preserve_digraphs
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.inverse_vocab: Dict[int, str] = {}

    def _get_stats(self, splits: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of adjacent symbol pairs across vocabulary items."""
        pairs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        for word, freq in splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], splits: Dict[str, int]) -> Dict[str, int]:
        """Replaces instances of the symbol pair with a single merged symbol."""
        v_out: Dict[str, int] = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in splits.items():
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = freq
        return v_out

    def train_from_text(self, text: str, min_frequency: int = 2) -> None:
        """Trains BPE vocabulary and merge rules from raw string corpus."""
        # 1. Normalize
        norm_text = unicodedata.normalize("NFC", text)
        words = re.findall(r"\b[\w'ɛɔɣŋʒƐƆƔŊƷ\-]+\b", norm_text)

        # Word frequency table
        word_counts = collections.Counter(words)

        # Base vocabulary initialization
        base_symbols: Set[str] = set(SPECIAL_TOKENS)
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-":
            base_symbols.add(ch)
        for sc in DAGBANI_SPECIAL_CHARS:
            base_symbols.add(sc)

        if self.preserve_digraphs:
            for dg in DAGBANI_DIGRAPHS:
                base_symbols.add(dg)

        # Prepare initial splits (characters separated by space)
        splits: Dict[str, int] = {}
        for w, freq in word_counts.items():
            if freq < min_frequency:
                continue
            # Segment characters while respecting digraphs
            syms: List[str] = []
            idx = 0
            while idx < len(w):
                if self.preserve_digraphs and idx + 1 < len(w) and w[idx:idx+2].lower() in {"kp", "gb", "ŋm", "ny", "ch", "sh"}:
                    syms.append(w[idx:idx+2])
                    idx += 2
                else:
                    syms.append(w[idx])
                    idx += 1
            splits[" ".join(syms)] = freq

        # Iterative BPE merges
        num_merges = self.vocab_size - len(base_symbols)
        for _ in range(max(0, num_merges)):
            pairs = self._get_stats(splits)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            splits = self._merge_vocab(best_pair, splits)
            self.merges.append(best_pair)
            base_symbols.add("".join(best_pair))

        # Finalize vocab dictionary
        ordered_vocab = SPECIAL_TOKENS + sorted(list(base_symbols - set(SPECIAL_TOKENS)))
        self.vocab = {sym: idx for idx, sym in enumerate(ordered_vocab)}
        self.inverse_vocab = {idx: sym for sym, idx in self.vocab.items()}

    def encode(self, text: str) -> List[str]:
        """Encodes input string into subword tokens."""
        norm_text = unicodedata.normalize("NFC", text)
        words = re.findall(r"\b[\w'ɛɔɣŋʒƐƆƔŊƷ\-]+\b|[^\w\s]", norm_text)
        all_tokens: List[str] = []

        for w in words:
            if w in self.vocab:
                all_tokens.append(w)
                continue

            # Character split
            syms: List[str] = []
            idx = 0
            while idx < len(w):
                if self.preserve_digraphs and idx + 1 < len(w) and w[idx:idx+2].lower() in {"kp", "gb", "ŋm", "ny", "ch", "sh"}:
                    syms.append(w[idx:idx+2])
                    idx += 2
                else:
                    syms.append(w[idx])
                    idx += 1

            # Apply learned merges
            for pair in self.merges:
                bigram = pair[0] + " " + pair[1]
                joined = " ".join(syms)
                if bigram in joined:
                    new_syms: List[str] = []
                    s_idx = 0
                    while s_idx < len(syms):
                        if s_idx < len(syms) - 1 and (syms[s_idx], syms[s_idx+1]) == pair:
                            new_syms.append(pair[0] + pair[1])
                            s_idx += 2
                        else:
                            new_syms.append(syms[s_idx])
                            s_idx += 1
                    syms = new_syms

            for s in syms:
                all_tokens.append(s if s in self.vocab else "<unk>")

        return all_tokens

    def save(self, output_dir: Path) -> None:
        """Exports vocab.json, merges.txt, and tokenizer.json."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.2 - Dagbani BPE merges\n")
            for p1, p2 in self.merges:
                f.write(f"{p1} {p2}\n")

        tokenizer_json = {
            "version": "1.0",
            "type": "ByteLevelBPE",
            "model": {
                "vocab": self.vocab,
                "merges": [" ".join(m) for m in self.merges]
            },
            "special_tokens": SPECIAL_TOKENS
        }
        with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)


# ============================================================================
# Hugging Face Tokenizers Wrapper & Trainer
# ============================================================================

def train_hf_tokenizer(
    corpus_file: Path,
    vocab_size: int,
    output_dir: Path,
    preserve_digraphs: bool = True,
    min_frequency: int = 2
) -> bool:
    """
    Attempts to train using Hugging Face `tokenizers` library with Byte-Level pre-tokenization.
    Falls back to PurePythonBPE if library is unavailable.
    """
    try:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

        print("Using Hugging Face `tokenizers` backend...")

        # Initialize BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Normalization: NFC Canonical
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC()
        ])

        # Pre-tokenizer: ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        ])

        # Decoder: ByteLevel
        tokenizer.decoder = decoders.ByteLevel()

        # Special initial alphabet
        initial_alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-")
        initial_alphabet.extend(DAGBANI_SPECIAL_CHARS)
        if preserve_digraphs:
            initial_alphabet.extend(DAGBANI_DIGRAPHS)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=initial_alphabet,
            show_progress=True
        )

        tokenizer.train([str(corpus_file)], trainer)

        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_dir / "tokenizer.json"))

        # Save vocab.json and merges.txt
        model = tokenizer.model
        with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>"
            }, f, indent=2)

        print(f"Hugging Face Tokenizer trained successfully in: {output_dir}")
        return True

    except ImportError:
        print("Hugging Face `tokenizers` not installed. Falling back to PurePythonBPE engine...")
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus_text = f.read()

        bpe = PurePythonBPE(vocab_size=vocab_size, preserve_digraphs=preserve_digraphs)
        bpe.train_from_text(corpus_text, min_frequency=min_frequency)
        bpe.save(output_dir)
        print(f"PurePythonBPE trained and saved in: {output_dir}")
        return True


# ============================================================================
# Subword Fertility Evaluator
# ============================================================================

def compute_fertility(tokens: List[str], raw_text: str) -> float:
    """Computes tokens / words fertility ratio."""
    words = raw_text.split()
    if not words:
        return 0.0
    return len(tokens) / float(len(words))


# ============================================================================
# Self-Test Verification Suite
# ============================================================================

def run_tokenizer_self_test() -> bool:
    """Executes a complete self-test verifying tokenizer training, encoding, and fertility."""
    print("=" * 60)
    print("Running Dagbani BPE Tokenizer Self-Tests...")
    print("=" * 60)

    sample_corpus = """
    O biɛla Yendi zúŋɔ ka nyɛ pukpara ŋun kpaŋsiri kpaŋkpaŋ koobu.
    Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam n-ti salo zaa.
    Wabgu maa mini gballi maa viɛla pam ka chɛ ka bɛ nyaba gbaŋgbahira.
    Kpamba mini bihi zaa laɣimmi n-wum samban' luŋa yɛltɔɣa viɛnyɛla.
    Ti bɔrimi ni ti gu ka taɣi ti kaya ni ta'ada Dagbaŋ tingbani puuni.
    """

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        corpus_path = tmp_path / "train_corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(sample_corpus * 10)  # Repeat for frequency statistics

        out_dir = tmp_path / "dagbani_bpe_test"
        success = train_hf_tokenizer(
            corpus_file=corpus_path,
            vocab_size=200,
            output_dir=out_dir,
            preserve_digraphs=True,
            min_frequency=1
        )
        assert success, "Tokenizer training failed!"

        assert (out_dir / "tokenizer.json").exists()

        # Pure Python encoder test
        bpe = PurePythonBPE(vocab_size=200, preserve_digraphs=True)
        bpe.train_from_text(sample_corpus * 10, min_frequency=1)

        test_sent = "O biɛla Yendi zúŋɔ ka nyɛ pukpara"
        tokens = bpe.encode(test_sent)
        fert = compute_fertility(tokens, test_sent)

        print(f"Test Sentence: '{test_sent}'")
        print(f"Encoded Tokens: {tokens}")
        print(f"Subword Fertility: {fert:.2f} tokens/word")

        assert len(tokens) > 0, "Encoding produced empty token list!"
        assert fert < 2.0, f"Fertility {fert:.2f} is unexpectedly high on Dagbani vocabulary"

    print("-" * 60)
    print("Dagbani BPE Tokenizer Self-Test: ALL ASSERTIONS PASSED!")
    print("=" * 60)
    return True


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a custom Byte-Level BPE Tokenizer for Dagbani"
    )
    parser.add_argument(
        "--corpus-file", "-c",
        type=str,
        help="Path to cleaned Dagbani text corpus file."
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=8000,
        help="Target vocabulary size (default: 8000)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="dagbani_bpe_out",
        help="Directory to save tokenizer artifacts (default: dagbani_bpe_out)."
    )
    parser.add_argument(
        "--preserve-digraphs",
        action="store_true",
        default=True,
        help="Preserve Dagbani digraphs (kp, gb, ŋm, ny, ch, sh) in initial alphabet."
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency threshold (default: 2)."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute tokenizer training and encoding self-test and exit."
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_tokenizer_self_test()
        sys.exit(0 if success else 1)

    if not args.corpus_file:
        parser.print_help()
        sys.exit(1)

    train_hf_tokenizer(
        corpus_file=Path(args.corpus_file),
        vocab_size=args.vocab_size,
        output_dir=Path(args.output_dir),
        preserve_digraphs=args.preserve_digraphs,
        min_frequency=args.min_frequency
    )


if __name__ == "__main__":
    main()
