#!/usr/bin/env python3
"""
Master Adversarial and Self-Test Runner for Challenger 1
========================================================
Runs:
1. dagbani_g2p.py self-test
2. orthography_normalizer.py self-test
3. dagbani_syllabifier.py self-test
4. dataset_cleaner.py self-test
5. train_dagbani_tokenizer.py self-test
6. test_adversarial_linguistics.py (Adversarial Suite 1: 19 test cases)
7. test_adversarial_tokenization_cleaning.py (Adversarial Suite 2: 9 test cases)
"""

import sys
import os
import unittest
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_LING = PROJECT_ROOT / "skills" / "dagbani-linguistics" / "scripts"
SKILLS_LLM = PROJECT_ROOT / "skills" / "dagbani-llm-tokenization-datasets" / "scripts"
sys.path.insert(0, str(SKILLS_LING))
sys.path.insert(0, str(SKILLS_LLM))

import dagbani_g2p
import orthography_normalizer
import dagbani_syllabifier
import dataset_cleaner
import train_dagbani_tokenizer

import test_adversarial_linguistics
import test_adversarial_tokenization_cleaning

def run_all():
    print("=" * 80)
    print("CHALLENGER 1: COMPREHENSIVE EMPIRICAL TEST HARNESS & AUDIT EXECUTION")
    print("=" * 80)

    results = {}

    # 1. G2P Built-in Self-Test
    print("\n>>> [1/7] Testing dagbani_g2p.py built-in self-test...")
    g2p_res = dagbani_g2p.run_self_test()
    results["dagbani_g2p_self_test"] = g2p_res

    # 2. Orthography Normalizer Built-in Self-Test
    print("\n>>> [2/7] Testing orthography_normalizer.py built-in self-test...")
    norm_res = orthography_normalizer.run_self_test()
    results["orthography_normalizer_self_test"] = norm_res

    # 3. Syllabifier Built-in Self-Test
    print("\n>>> [3/7] Testing dagbani_syllabifier.py built-in self-test...")
    syl_res = dagbani_syllabifier.run_self_test()
    results["dagbani_syllabifier_self_test"] = syl_res

    # 4. Dataset Cleaner Built-in Self-Test
    print("\n>>> [4/7] Testing dataset_cleaner.py built-in self-test...")
    cleaner_res = dataset_cleaner.run_cleaner_self_test()
    results["dataset_cleaner_self_test"] = cleaner_res

    # 5. Tokenizer Built-in Self-Test
    print("\n>>> [5/7] Testing train_dagbani_tokenizer.py built-in self-test...")
    tok_res = train_dagbani_tokenizer.run_tokenizer_self_test()
    results["train_dagbani_tokenizer_self_test"] = tok_res

    # 6. Adversarial Suite 1: Linguistics (19 unit tests)
    print("\n>>> [6/7] Running Adversarial Suite 1 (Linguistics, G2P, Normalizer, Syllables)...")
    suite1 = unittest.TestLoader().loadTestsFromNames([
        "test_adversarial_linguistics.TestDagbaniG2PAdversarial",
        "test_adversarial_linguistics.TestOrthographyNormalizerAdversarial",
        "test_adversarial_linguistics.TestDagbaniSyllabifierAdversarial",
    ])
    runner1 = unittest.TextTestRunner(verbosity=2)
    res1 = runner1.run(suite1)
    results["adversarial_suite_1_linguistics"] = res1.wasSuccessful()

    # 7. Adversarial Suite 2: Tokenization & Cleaning (9 unit tests)
    print("\n>>> [7/7] Running Adversarial Suite 2 (Cleaning, Scripts, Tokenizer, Digraphs)...")
    suite2 = unittest.TestLoader().loadTestsFromNames([
        "test_adversarial_tokenization_cleaning.TestDagbaniDatasetCleanerAdversarial",
        "test_adversarial_tokenization_cleaning.TestDagbaniTokenizerAdversarial",
    ])
    runner2 = unittest.TextTestRunner(verbosity=2)
    res2 = runner2.run(suite2)
    results["adversarial_suite_2_tokenization_cleaning"] = res2.wasSuccessful()

    print("\n" + "=" * 80)
    print("MASTER TEST SUMMARY")
    print("=" * 80)
    all_ok = True
    for test_name, status in results.items():
        pass_str = "PASS" if status else "FAIL"
        print(f"  - {test_name:<45}: [{pass_str}]")
        if not status:
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("ALL TESTS AND ADVERSARIAL STRESS SUITES PASSED EMPIRICALLY (100% PASS RATE)")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! AUDIT IDENTIFIED DEFECTS.")
        sys.exit(1)

if __name__ == "__main__":
    run_all()
