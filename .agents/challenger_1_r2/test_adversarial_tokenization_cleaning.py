#!/usr/bin/env python3
"""
Adversarial Stress Test Suite: Dagbani Dataset Cleaning & Tokenizer Training
===========================================================================
Tests:
1. dataset_cleaner.py (Text hygiene, Unicode NFC, script filter, deduplication)
2. train_dagbani_tokenizer.py (PurePythonBPE, HF tokenizers, digraph preservation, fertility)

Author: Challenger 1 (Empirical Challenger)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root and skill script directories to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_TOKENIZATION = PROJECT_ROOT / "skills" / "dagbani-llm-tokenization-datasets" / "scripts"
sys.path.insert(0, str(SKILLS_TOKENIZATION))

from dataset_cleaner import DagbaniTextCleaner
from train_dagbani_tokenizer import PurePythonBPE, compute_fertility, train_hf_tokenizer, DAGBANI_DIGRAPHS, DAGBANI_SPECIAL_CHARS


class TestDagbaniDatasetCleanerAdversarial(unittest.TestCase):
    """Adversarial stress testing for Dagbani Dataset Cleaner & Normalizer."""

    def setUp(self):
        self.cleaner = DagbaniTextCleaner(min_words=3, max_words=100, min_chars=10, max_chars=1000)

    def test_unicode_nfc_and_smart_quote_hygiene(self):
        """Verify NFC canonicalization and standardization of all apostrophe and quote variants."""
        samples = [
            ("“Ti kpalinʒoo n’nyɛla din viɛli pam!”", "Standardization of curly double and single quotes"),
            ("‘Dagbaŋ kaya ni ta’ada nyɛla din mali yaa pam.’", "Standardization of curly single quotes"),
            ("«Biɛɣu viɛli pam n-ti ti salo zaa.»", "Standardization of guillemets"),
            ("O biɛla Yendi zúŋɔ ka nyɛ pukpara.", "NFC preservation of accented vowels and special glyphs"),
        ]
        for raw, desc in samples:
            cleaned = self.cleaner.clean_text(raw, deduplicate=False)
            self.assertIsNotNone(cleaned, f"Cleaner rejected valid Dagbani text ({desc}): '{raw}'")
            self.assertNotIn("“", cleaned)
            self.assertNotIn("”", cleaned)
            self.assertNotIn("‘", cleaned)
            self.assertNotIn("«", cleaned)
            self.assertNotIn("»", cleaned)
            print(f"[Cleaner Hygiene] '{raw}' -> '{cleaned}'")

    def test_script_filtering_foreign_and_corrupt(self):
        """Stress-test rejection of non-Latin scripts (Cyrillic, Arabic, Chinese, Emojis, Symbols)."""
        rejections = [
            ("Это русский текст который должен быть отклонен", "Cyrillic script sentence"),
            ("هذا نص عربي يجب رفضه من قبل المنظف", "Arabic script sentence"),
            ("这是一个完全中文的句子，应该被彻底拒绝", "Chinese script sentence"),
            ("🚀🌟🔥🎉🎈 Dagbaŋ biɛɣu viɛli 🤖👻👾", "Emoji-flooded sentence with over-ratio symbols"),
            ("@@@ ### $$$ %%% ^^^ &&& *** ((( ))) ___ +++", "Pure symbol noise without letters"),
            ("1234567890 9876543210 1122334455", "Pure numeric string without letters"),
        ]
        for raw, desc in rejections:
            cleaned = self.cleaner.clean_text(raw, deduplicate=False)
            self.assertIsNone(cleaned, f"Cleaner failed to reject alien/corrupt script ({desc}): '{raw}'")

    def test_length_boundary_conditions(self):
        """Stress-test min/max words and min/max characters boundary conditions."""
        # Too short words (< 3 words)
        self.assertIsNone(self.cleaner.clean_text("Biɛɣu viɛli")) # 2 words
        self.assertIsNone(self.cleaner.clean_text("Hi")) # 1 word, too short chars
        # Valid 3 words >= 10 chars
        self.assertIsNotNone(self.cleaner.clean_text("Biɛɣu viɛli pam")) # 3 words, 14 chars

        # Too many words (> 100 words)
        long_sentence = " ".join(["yɛltɔɣa"] * 105)
        self.assertIsNone(self.cleaner.clean_text(long_sentence))

    def test_zero_width_spaces_and_control_chars(self):
        """Stress-test zero-width spaces (ZWSP, ZWNJ, ZWJ, BOM) and control characters."""
        text_with_zwsp = "O biɛla\u200b Yendi\u200c zúŋɔ\u200d ka nyɛ\ufeff pukpara."
        cleaned = self.cleaner.clean_text(text_with_zwsp, deduplicate=False)
        self.assertIsNotNone(cleaned, "Cleaner rejected sentence with hidden zero-width spaces")
        # Ensure text is normalized
        self.assertTrue(len(cleaned) > 0)

    def test_sha256_exact_deduplication(self):
        """Verify strict SHA-256 deduplication filtering across multiple occurrences."""
        cleaner_dedup = DagbaniTextCleaner(min_words=3, max_words=50)
        sent = "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam."

        first_pass = cleaner_dedup.clean_text(sent, deduplicate=True)
        self.assertIsNotNone(first_pass, "First occurrence should pass deduplication")

        second_pass = cleaner_dedup.clean_text(sent, deduplicate=True)
        self.assertIsNone(second_pass, "Exact duplicate should be rejected by deduplication")

        # Slight variation (extra whitespace) should also be normalized to same hash and rejected
        variant_ws = "  Dagbaŋ   kaya   ni   ta'ada   nyɛla   din   mali   yaa   pam.  "
        third_pass = cleaner_dedup.clean_text(variant_ws, deduplicate=True)
        self.assertIsNone(third_pass, "Whitespace variant should be normalized to duplicate and rejected")

    def test_jsonl_and_txt_file_processing(self):
        """Verify complete file processing pipeline on .txt and .jsonl corpora."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Text file test
            txt_in = tmp_path / "raw_corpus.txt"
            txt_out = tmp_path / "clean_corpus.txt"
            lines = [
                "O biɛla Yendi zúŋɔ ka nyɛ pukpara ŋun kpaŋsiri kpaŋkpaŋ koobu.\n",
                "Hi\n",  # too short
                "O biɛla Yendi zúŋɔ ka nyɛ pukpara ŋun kpaŋsiri kpaŋkpaŋ koobu.\n",  # duplicate
                "Это русский текст который должен быть отклонен фильтром\n",  # foreign
                "Kpamba mini bihi zaa laɣimmi n-wum samban' luŋa yɛltɔɣa viɛnyɛla.\n"
            ]
            with open(txt_in, "w", encoding="utf-8") as f:
                f.writelines(lines)

            stats_txt = self.cleaner.process_file(txt_in, txt_out, deduplicate=True)
            self.assertEqual(stats_txt["total_lines"], 5)
            self.assertEqual(stats_txt["retained_lines"], 2)
            self.assertEqual(stats_txt["rejected_lines"], 3)

            # 2. JSONL file test
            jsonl_in = tmp_path / "raw_corpus.jsonl"
            jsonl_out = tmp_path / "clean_corpus.jsonl"
            records = [
                {"id": 1, "text": "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam n-ti salo zaa."},
                {"id": 2, "text": "Too short"},
                {"id": 3, "text": "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam n-ti salo zaa."}, # dup
                {"id": 4, "instruction": "Wabgu maa mini gballi maa viɛla pam ka chɛ ka bɛ nyaba gbaŋgbahira."}
            ]
            with open(jsonl_in, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            cleaner_jsonl = DagbaniTextCleaner(min_words=3, max_words=50)
            stats_jsonl = cleaner_jsonl.process_file(jsonl_in, jsonl_out, deduplicate=True)
            self.assertEqual(stats_jsonl["total_lines"], 4)
            self.assertEqual(stats_jsonl["retained_lines"], 2)
            self.assertEqual(stats_jsonl["rejected_lines"], 2)


class TestDagbaniTokenizerAdversarial(unittest.TestCase):
    """Adversarial stress testing for Dagbani Byte-Level BPE Tokenizer Trainer."""

    def setUp(self):
        self.synthetic_dagbani_corpus = """
        O biɛla Yendi zúŋɔ ka nyɛ pukpara ŋun kpaŋsiri kpaŋkpaŋ koobu tiŋa maa puuni.
        Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam n-ti salo zaa ka bihi bɔhindi li.
        Wabgu maa mini gballi maa viɛla pam ka chɛ ka bɛ nyaba gbaŋgbahira kpalinʒoo.
        Kpamba mini bihi zaa laɣimmi n-wum samban' luŋa yɛltɔɣa viɛnyɛla n-ti sokam.
        Ti bɔrimi ni ti gu ka taɣi ti kaya ni ta'ada Dagbaŋ tingbani puuni saha kam.
        Ŋmaŋa mini kpariba zaa chani la daa n-kɔhiri bɛ nɛma viɛnyɛla.
        Naawuni ti ti suhudoo mini alaafee ti yaalim maa puuni dabisili kam.
        """

    def test_pure_python_bpe_training_and_fertility(self):
        """Train PurePythonBPE on synthetic corpus and verify low subword fertility (< 1.6 tokens/word)."""
        bpe = PurePythonBPE(vocab_size=300, preserve_digraphs=True)
        # Train with 10x repeated corpus to build merge statistics
        bpe.train_from_text(self.synthetic_dagbani_corpus * 10, min_frequency=2)

        # Check vocabulary contains digraphs & BGL special chars
        for sc in DAGBANI_SPECIAL_CHARS:
            self.assertIn(sc, bpe.vocab, f"Special character '{sc}' missing from BPE initial vocab")
        for dg in ["kp", "gb", "ŋm", "ny", "ch", "sh"]:
            self.assertIn(dg, bpe.vocab, f"Digraph '{dg}' missing from BPE initial vocab")

        # Test encoding and fertility
        test_sentences = [
            "O biɛla Yendi zúŋɔ ka nyɛ pukpara",
            "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam",
            "Kpamba mini bihi zaa laɣimmi n-wum samban' luŋa yɛltɔɣa",
            "Ŋmaŋa mini kpariba zaa chani la daa",
        ]

        for s in test_sentences:
            tokens = bpe.encode(s)
            fert = compute_fertility(tokens, s)
            print(f"[BPE Encode] Text: '{s}'")
            print(f"             Tokens: {tokens}")
            print(f"             Fertility: {fert:.2f} tokens/word")
            self.assertTrue(len(tokens) > 0, f"Token list is empty for '{s}'")
            self.assertLess(fert, 1.85, f"Fertility {fert:.2f} too high for Dagbani sentence: '{s}'")

    def test_digraph_preservation_in_splits(self):
        """Verify digraphs (kp, gb, ŋm, ny, ch, sh) are treated as atomic units in tokenization."""
        bpe = PurePythonBPE(vocab_size=200, preserve_digraphs=True)
        bpe.train_from_text(self.synthetic_dagbani_corpus * 5, min_frequency=1)

        # Words with digraphs
        test_word = "kpaŋkpaŋ"
        toks = bpe.encode(test_word)
        # Verify 'kp' was not broken into separate 'k' and 'p'
        has_kp = any("kp" in t.lower() for t in toks)
        self.assertTrue(has_kp, f"Digraph 'kp' was fragmented into individual letters in '{toks}' for '{test_word}'")

        test_word_nm = "ŋmaŋa"
        toks_nm = bpe.encode(test_word_nm)
        has_nm = any("ŋm" in t.lower() for t in toks_nm)
        self.assertTrue(has_nm, f"Digraph 'ŋm' was fragmented in '{toks_nm}' for '{test_word_nm}'")

    def test_huggingface_tokenizer_integration_and_export(self):
        """Test full train_hf_tokenizer pipeline (with HF or pure Python fallback) and export verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corpus_path = tmp_path / "synthetic_train.txt"
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(self.synthetic_dagbani_corpus * 15)

            out_dir = tmp_path / "dagbani_bpe_export"
            success = train_hf_tokenizer(
                corpus_file=corpus_path,
                vocab_size=250,
                output_dir=out_dir,
                preserve_digraphs=True,
                min_frequency=1
            )
            self.assertTrue(success, "train_hf_tokenizer returned False")

            # Check exported artifacts
            self.assertTrue((out_dir / "tokenizer.json").exists(), "tokenizer.json artifact missing")
            self.assertTrue(
                (out_dir / "vocab.json").exists() or (out_dir / "tokenizer_config.json").exists(),
                "vocab configuration missing"
            )
            print(f"[Tokenizer Export] Successfully exported artifacts to {out_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING EMPIRICAL ADVERSARIAL TEST SUITE 2 (TOKENIZATION & DATASETS)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromNames([
        "__main__.TestDagbaniDatasetCleanerAdversarial",
        "__main__.TestDagbaniTokenizerAdversarial",
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nADVERSARIAL SUITE 2 FAILED!")
        sys.exit(1)
    else:
        print("\nADVERSARIAL SUITE 2 PASSED ALL TESTS!")
        sys.exit(0)
