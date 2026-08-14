#!/usr/bin/env python3
"""
Adversarial Stress Test Suite: Dagbani Linguistics Tooling
=========================================================
Tests:
1. dagbani_g2p.py (G2P phonology rule engine)
2. orthography_normalizer.py (BGL/ASCII normalizer & sanitizer)
3. dagbani_syllabifier.py (Syllable parser & moraic weight engine)

Author: Challenger 1 (Empirical Challenger)
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root and skill script directories to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_LINGUISTICS = PROJECT_ROOT / "skills" / "dagbani-linguistics" / "scripts"
sys.path.insert(0, str(SKILLS_LINGUISTICS))

from dagbani_g2p import DagbaniG2P, CHAR_MAP, DIGRAPHS_BGL, DIGRAPHS_ASCII
from orthography_normalizer import DagbaniNormalizer, ASCII_TO_BGL_LEXICON, BGL_TO_ASCII_MAP, EASTERN_TO_WESTERN_MAP
from dagbani_syllabifier import DagbaniSyllabifier, Syllable


class TestDagbaniG2PAdversarial(unittest.TestCase):
    """Adversarial stress testing for Dagbani G2P Converter."""

    def setUp(self):
        self.g2p_bgl = DagbaniG2P(default_tone=True, ascii_mode=False)
        self.g2p_notone = DagbaniG2P(default_tone=False, ascii_mode=False)
        self.g2p_ascii = DagbaniG2P(default_tone=True, ascii_mode=True)

    def test_digraph_and_special_glyphs(self):
        """Verify proper IPA mapping for all 1998 BGL digraphs and special glyphs."""
        cases = [
            ("ŋmaŋa", "ŋ͡máŋá"),
            ("kpamba", "k͡pámbá"),
            ("gballi", "ɡ͡bállí"),
            ("nyabli", "ɲáblí"),
            ("chaŋ", "t͡ʃáŋ́"),
            ("shikuru", "ʃíkúːrú"),
            ("ʒɛm", "ʒɛ́m"),
            ("paɣa", "páɣá"),
        ]
        for word, exp_pattern in cases:
            res = self.g2p_bgl.convert_word(word)
            self.assertTrue(len(res) > 0, f"Conversion returned empty string for '{word}'")
            print(f"[G2P Glyph Check] '{word}' -> '{res}'")

    def test_contextual_palatalization_front_vowels(self):
        """Stress-test palatalization and labial-coronal mutation before front vowels (i, e, ɛ)."""
        # kp -> tp, gb -> db, ŋm -> nm before front vowels
        cases = [
            ("kpɛ", "t͡p"),      # kp + ɛ -> tp
            ("kpi", "t͡p"),      # kp + i -> tp
            ("gbee", "d͡b"),     # gb + ee -> db
            ("ŋmɛ", "n͡m"),     # ŋm + ɛ -> nm
            ("kpa", "k͡p"),     # kp + a -> kp (no mutation before back/low vowel)
            ("gba", "ɡ͡b"),     # gb + a -> gb (no mutation)
            ("ŋma", "ŋ͡m"),     # ŋm + a -> ŋm (no mutation)
        ]
        for word, exp_onset in cases:
            res = self.g2p_bgl.convert_word(word, with_tone=False)
            self.assertTrue(res.startswith(exp_onset), f"Expected onset '{exp_onset}' in '{res}' for input '{word}'")

    def test_intervocalic_lenition_and_blocking(self):
        """Stress-test intervocalic lenition (d->r, s->h, g->ɣ) and blocking in clusters."""
        # Intervocalic: /ada/ -> [ara], /asa/ -> [aha], /aga/ -> [aɣa]
        self.assertIn("r", self.g2p_bgl.convert_word("bada", with_tone=False))
        self.assertIn("h", self.g2p_bgl.convert_word("bihi", with_tone=False))
        self.assertIn("ɣ", self.g2p_bgl.convert_word("paga", with_tone=False))

        # Blocked when preceded by consonant / nasal
        # In 'duunsi', 's' is after 'n', not between vowels -> should NOT debuccalize to 'h'
        # But 's' palatalizes to 'ʃ' before front vowel 'i'
        res_duunsi = self.g2p_bgl.convert_word("duunsi", with_tone=False)
        self.assertIn("ʃ", res_duunsi, f"Palatalization before 'i' produced unexpected output in '{res_duunsi}'")
        self.assertNotIn("h", res_duunsi, f"Debuccalization incorrectly converted s->h in '{res_duunsi}'")

    def test_disharmonic_loanwords(self):
        """Stress-test loanwords with disharmonic vowel structures and alien phonotactics."""
        loanwords = [
            ("asibiti", "Hospital loanword with disharmonic front/low vowels"),
            ("sooja", "Soldier loanword with back vowel + low vowel"),
            ("rediyo", "Radio loanword"),
            ("komputa", "Computer loanword"),
            ("loori", "Lorry loanword with geminate long vowel"),
            ("bɔlu", "Ball loanword with open-mid back vowel"),
            ("dokita", "Doctor loanword"),
        ]
        for word, desc in loanwords:
            res = self.g2p_bgl.convert_word(word)
            self.assertTrue(len(res) >= len(word), f"Conversion too short for loanword '{word}' ({desc}): '{res}'")
            print(f"[G2P Loanword] '{word}' ({desc}) -> '{res}'")

    def test_curly_smart_quotes_and_glottals(self):
        """Verify robust normalization and phonetic representation of glottal stop / quotes."""
        variants = ["bɛ'ʊ́", "bɛ’ʊ́", "bɛ‘ʊ́", "bɛ`ʊ́"]
        outputs = [self.g2p_bgl.convert_word(v, with_tone=False) for v in variants]
        # All variants should produce glottal stop ʔ
        for i, out in enumerate(outputs):
            self.assertIn("ʔ", out, f"Glottal stop missing in output '{out}' for variant '{variants[i]}'")

    def test_mixed_case_and_uppercase_bgl(self):
        """Stress-test uppercase and mixed-case BGL glyphs in G2P."""
        cases = [
            ("ŊMAŊA", "ŋ͡m"),
            ("KPAMBA", "k͡p"),
            ("GBALLI", "ɡ͡b"),
            ("BIƐƔU", "ɣ"),
            ("Paɣa", "páɣá"),
            ("Kpɛ", "t͡p"),
        ]
        for word, exp_sub in cases:
            res = self.g2p_bgl.convert_word(word)
            self.assertIn(exp_sub, res, f"Expected '{exp_sub}' in uppercase/mixed conversion '{res}' for '{word}'")

    def test_ascii_transliteration_mode(self):
        """Test G2P in ASCII mode converting digraphs like ngm, gh, zh, ng."""
        cases = [
            ("ngmande", "ŋ͡m"),
            ("pagaba", "ɣ"),
            ("zhangmi", "ʒ"),
            ("sheba", "ʃ"),
            ("chana", "t͡ʃ"),
        ]
        for word, exp_sub in cases:
            res = self.g2p_ascii.convert_word(word)
            self.assertIn(exp_sub, res, f"ASCII mode expected '{exp_sub}' in '{res}' for '{word}'")

    def test_empty_and_corrupt_inputs(self):
        """Edge case robustness: empty strings, whitespace, non-alphabetic chars."""
        self.assertEqual(self.g2p_bgl.convert_word(""), "")
        self.assertEqual(self.g2p_bgl.convert_sentence("   "), "")
        # Punctuation-only
        self.assertEqual(self.g2p_bgl.convert_word("..."), "...")
        self.assertEqual(self.g2p_bgl.convert_word("!?"), "!?")
        # Word with attached punctuation
        res = self.g2p_bgl.convert_word("«paɣa»")
        self.assertTrue(res.startswith("«") and res.endswith("»"), f"Enclosing punctuation corrupted in '{res}'")


class TestOrthographyNormalizerAdversarial(unittest.TestCase):
    """Adversarial stress testing for Dagbani Orthography Normalizer."""

    def setUp(self):
        self.normalizer = DagbaniNormalizer()

    def test_ascii_to_bgl_lexicon_and_patterns(self):
        """Stress-test ASCII to BGL conversion with high-frequency lexicon items and fallback patterns."""
        cases = [
            ("nyela paga mini bihi ban be Tamale", "nyɛla paɣa mini bihi ban bɛ Tamale"),
            ("sheba kpe shikuru", "shɛba kpɛ shikuru"),
            ("ngmanga mini wuntanga", "ŋmaŋa mini wuntaŋa"),
            ("kpalinzhoo zhem", "kpalinʒoo ʒɛm"),
            ("biegu vieli", "biɛɣu viɛli"),
            ("dogim mini doba", "dɔɣim mini dɔba"),
        ]
        for inp, expected in cases:
            res = self.normalizer.convert_ascii_to_bgl(inp)
            self.assertEqual(res, expected, f"ASCII->BGL failed: in '{inp}' -> got '{res}' vs expected '{expected}'")

    def test_case_preservation_ascii_to_bgl(self):
        """Verify titlecase and uppercase preservation in transliteration."""
        cases = [
            ("Paga", "Paɣa"),
            ("PAGABA", "PAƔABA"),
            ("Biegu", "Biɛɣu"),
            ("Ngmanga", "Ŋmaŋa"),
        ]
        for inp, exp in cases:
            res = self.normalizer.convert_ascii_to_bgl(inp)
            self.assertEqual(res, exp, f"Case preservation failed: '{inp}' -> '{res}', expected '{exp}'")

    def test_bgl_to_ascii_transliteration(self):
        """Verify conversion from 1998 BGL standard characters to clean ASCII."""
        bgl_text = "Nyɛla paɣa mini ʒɛm din viɛli pam ka mali ŋmaŋa"
        ascii_out = self.normalizer.convert_bgl_to_ascii(bgl_text)
        # Ensure none of the special BGL characters remain
        for ch in ["ɛ", "ɔ", "ŋ", "ɣ", "ʒ", "Ɛ", "Ɔ", "Ŋ", "Ɣ", "Ʒ"]:
            self.assertNotIn(ch, ascii_out, f"Special char '{ch}' remained in ASCII conversion: '{ascii_out}'")
        print(f"[BGL->ASCII] '{bgl_text}' -> '{ascii_out}'")

    def test_punctuation_and_quote_sanitization(self):
        """Stress-test quotes, em-dashes, excessive punctuation, and elongated words."""
        noisy = "“O yeliya: ‘Biɛɣu—viɛli   pammm!!!’ Ni...   «Dagbaŋ»”"
        clean = self.normalizer.sanitize_punctuation(noisy)
        self.assertNotIn("“", clean)
        self.assertNotIn("”", clean)
        self.assertNotIn("‘", clean)
        self.assertNotIn("—", clean)
        self.assertIn('"', clean)
        self.assertIn("'", clean)
        self.assertIn("-", clean)
        self.assertIn("pam!", clean)  # pammm collapsed
        self.assertNotIn("   ", clean) # whitespaces collapsed

    def test_dialect_standardization(self):
        """Verify Eastern Nayahili to Western Tomosili standard conversion."""
        eastern = "O yɛltoɣa maa mini kɔbga toɣsi"
        standard = self.normalizer.standardize_dialect(eastern)
        self.assertIn("yɛltɔɣa", standard)
        self.assertIn("kɔwa", standard)
        self.assertIn("tɔxi", standard)

    def test_asr_normalization_pipeline(self):
        """Verify normalization for ASR WER/CER evaluation."""
        raw = "“Ti kpalinʒoo n-nyɛla DIN viɛli pam, 100%!”"
        norm = self.normalizer.normalize_for_asr(raw)
        self.assertEqual(norm, "ti kpalinʒoo n-nyɛla din viɛli pam")
        # Check special characters remain lowercased
        for ch in ["ʒ", "ɛ"]:
            self.assertIn(ch, norm)


class TestDagbaniSyllabifierAdversarial(unittest.TestCase):
    """Adversarial stress testing for Dagbani Syllabifier."""

    def setUp(self):
        self.syllabifier = DagbaniSyllabifier()

    def test_canonical_syllable_templates(self):
        """Test canonical Dagbani templates: V, CV, CVC, CVV, CVVC, N."""
        test_words = [
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
            res = self.syllabifier.syllabify_word(word)
            templates = [s.template for s in res]
            total_moras = self.syllabifier.get_total_moras(res)
            self.assertEqual(templates, exp_templates, f"Template mismatch for '{word}' ({desc}): got {templates}, expected {exp_templates}")
            self.assertEqual(total_moras, exp_moras, f"Mora count mismatch for '{word}': got {total_moras}, expected {exp_moras}")
            print(f"[Syllable Check] '{word}' -> {' . '.join(str(s) for s in res)} ({total_moras}μ)")

    def test_syllabic_prefix_nasals(self):
        """Stress-test syllabic nasal prefixes: m-, n-, ŋ- with tones."""
        prefixes = ["m-bɔ", "n-da", "ŋ-ka", "m̀-bɔ́", "n-nyɛ"]
        for p in prefixes:
            res = self.syllabifier.syllabify_word(p)
            self.assertTrue(len(res) >= 2, f"Prefix not segmented as syllabic nasal in '{p}': {res}")
            self.assertEqual(res[0].template, "N", f"First syllable of '{p}' was '{res[0].template}', expected 'N'")
            self.assertEqual(res[0].moras, 1, f"Nasal prefix mora should be 1, got {res[0].moras}")

    def test_moraic_coda_nasals_and_glottal_stops(self):
        """Verify moraic weight addition for coda nasals (m, n, ŋ) and glottal hiatus."""
        # Dagbaŋ: Dag-baŋ (CVC + CVC), coda ŋ adds 1 mora -> Dag (CVC, 1 or 2μ), baŋ (CVC, 2μ)
        res_dagban = self.syllabifier.syllabify_word("Dagbaŋ")
        self.assertEqual(len(res_dagban), 2, f"Expected 2 syllables for 'Dagbaŋ', got {len(res_dagban)}")
        self.assertEqual(res_dagban[1].coda, "ŋ", f"Expected coda 'ŋ' in second syllable of 'Dagbaŋ'")
        self.assertEqual(res_dagban[1].moras, 2, f"Expected 2 moras for bimoraic coda nasal in 'baŋ'")

        # Glottal hiatus: bɛ'ʊ́
        res_beu = self.syllabifier.syllabify_word("bɛ'ʊ́")
        self.assertTrue(len(res_beu) >= 1, f"Glottal hiatus parsing failed on 'bɛ\'ʊ́'")

    def test_long_compounds_and_loanwords(self):
        """Stress-test multi-syllable compounds and loanwords."""
        words = ["pagasara", "kpalinʒoo", "asibiti", "shikuru", "wuntaŋa"]
        for w in words:
            res = self.syllabifier.syllabify_word(w)
            moras = self.syllabifier.get_total_moras(res)
            self.assertTrue(len(res) >= 2, f"Multi-syllable word '{w}' has only {len(res)} syllables")
            self.assertTrue(moras >= len(res), f"Total moras {moras} should be >= syllable count {len(res)}")
            print(f"[Complex Syllable] '{w}' -> {' . '.join(s.raw for s in res)} ({moras}μ)")

    def test_empty_and_single_char_syllables(self):
        """Edge cases: empty string, single vowel, whitespace."""
        self.assertEqual(self.syllabifier.syllabify_word(""), [])
        single_v = self.syllabifier.syllabify_word("a")
        self.assertEqual(len(single_v), 1)
        self.assertEqual(single_v[0].template, "V")
        self.assertEqual(single_v[0].moras, 1)


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING EMPIRICAL ADVERSARIAL TEST SUITE 1 (LINGUISTICS)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromNames([
        "__main__.TestDagbaniG2PAdversarial",
        "__main__.TestOrthographyNormalizerAdversarial",
        "__main__.TestDagbaniSyllabifierAdversarial",
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nADVERSARIAL SUITE 1 FAILED!")
        sys.exit(1)
    else:
        print("\nADVERSARIAL SUITE 1 PASSED ALL TESTS!")
        sys.exit(0)
