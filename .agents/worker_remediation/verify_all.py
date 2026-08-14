#!/usr/bin/env python3
"""
Comprehensive Verification Test Runner for Dagbani AI Remediation
=================================================================
Runs:
1. dagbani_g2p.py self-test
2. orthography_normalizer.py self-test
3. dagbani_syllabifier.py self-test
4. audio_preprocessor.py self-test
5. whisper_dagbani_trainer.py self-test
6. evaluate_asr.py self-test
7. dagbani_phonemizer.py self-test
8. prepare_tts_dataset.py self-test
9. synthesize_tts.py self-test
10. train_dagbani_tokenizer.py self-test
11. dataset_cleaner.py self-test
12. llm_lora_finetuner.py self-test
13. Challenger 1 Master Adversarial Suite
14. Challenger 2 Master Adversarial Suite
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path("d:/ATS Tech/Dagbani AI")
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-linguistics/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-asr-whisper/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-tts-synthesis/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-llm-tokenization-datasets/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / ".agents/challenger_1_r2"))
sys.path.insert(0, str(PROJECT_ROOT / ".agents/challenger_2_r2"))

# Import all scripts
import dagbani_g2p
import orthography_normalizer
import dagbani_syllabifier
import audio_preprocessor
import whisper_dagbani_trainer
import evaluate_asr
import dagbani_phonemizer
import prepare_tts_dataset
import synthesize_tts
import train_dagbani_tokenizer
import dataset_cleaner
import llm_lora_finetuner

import run_all_adversarial_and_self_tests as ch1_runner
import adversarial_test_suite as ch2_runner


def main():
    print("=" * 80)
    print("STARTING REMEDIATION WORKER COMPLETE VERIFICATION HARNESS")
    print("=" * 80)

    results = {}

    # 1. dagbani_g2p
    print("\n--- [1/12] Testing dagbani_g2p.py --self-test ---")
    results["dagbani_g2p"] = dagbani_g2p.run_self_test()

    # 2. orthography_normalizer
    print("\n--- [2/12] Testing orthography_normalizer.py --self-test ---")
    results["orthography_normalizer"] = orthography_normalizer.run_self_test()

    # 3. dagbani_syllabifier
    print("\n--- [3/12] Testing dagbani_syllabifier.py --self-test ---")
    results["dagbani_syllabifier"] = dagbani_syllabifier.run_self_test()

    # 4. audio_preprocessor
    print("\n--- [4/12] Testing audio_preprocessor.py --self-test ---")
    results["audio_preprocessor"] = audio_preprocessor.run_self_test()

    # 5. whisper_dagbani_trainer
    print("\n--- [5/12] Testing whisper_dagbani_trainer.py --self-test ---")
    results["whisper_dagbani_trainer"] = whisper_dagbani_trainer.run_self_test()

    # 6. evaluate_asr
    print("\n--- [6/12] Testing evaluate_asr.py --self-test ---")
    results["evaluate_asr"] = evaluate_asr.run_self_test()

    # 7. dagbani_phonemizer
    print("\n--- [7/12] Testing dagbani_phonemizer.py --self-test ---")
    results["dagbani_phonemizer"] = dagbani_phonemizer.run_phonemizer_self_test()

    # 8. prepare_tts_dataset
    print("\n--- [8/12] Testing prepare_tts_dataset.py --self-test ---")
    results["prepare_tts_dataset"] = prepare_tts_dataset.run_self_test()

    # 9. synthesize_tts
    print("\n--- [9/12] Testing synthesize_tts.py --self-test ---")
    results["synthesize_tts"] = synthesize_tts.run_self_test()

    # 10. train_dagbani_tokenizer
    print("\n--- [10/12] Testing train_dagbani_tokenizer.py --self-test ---")
    results["train_dagbani_tokenizer"] = train_dagbani_tokenizer.run_tokenizer_self_test()

    # 11. dataset_cleaner
    print("\n--- [11/12] Testing dataset_cleaner.py --self-test ---")
    results["dataset_cleaner"] = dataset_cleaner.run_cleaner_self_test()

    # 12. llm_lora_finetuner
    print("\n--- [12/12] Testing llm_lora_finetuner.py --self-test ---")
    results["llm_lora_finetuner"] = llm_lora_finetuner.run_self_test()

    # 13. Challenger 2 Adversarial Test Suite
    print("\n--- [13] Running Challenger 2 Master Adversarial Test Suite ---")
    suite2 = ch2_runner.AdversarialTestSuite()
    ch2_summary = suite2.run_all()
    results["challenger_2_adversarial_suite"] = (ch2_summary["failed"] == 0)

    # Master Summary
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION SUMMARY REPORT")
    print("=" * 80)
    all_ok = True
    for name, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"  * {name:<35}: [{status_str}]")
        if not status:
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("ALL 12 SCRIPT SELF-TESTS AND ADVERSARIAL SUITES PASSED (100% SUCCESS RATE)!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
