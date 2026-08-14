## 2026-08-14T21:28:26Z
You are the Remediation Worker for the Dagbani AI project.
Your working directory is: d:/ATS Tech/Dagbani AI/.agents/worker_remediation/
Project Root: d:/ATS Tech/Dagbani AI
Read ORIGINAL_REQUEST.md: d:/ATS Tech/Dagbani AI/ORIGINAL_REQUEST.md
Read PROJECT.md: d:/ATS Tech/Dagbani AI/PROJECT.md

Your inputs:
1. Reviewer 1 Handoff: `d:/ATS Tech/Dagbani AI/.agents/reviewer_1/handoff.md`
2. Challenger 2 Handoff: `d:/ATS Tech/Dagbani AI/.agents/challenger_2_r2/handoff.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your mission:
Apply the exact, targeted code fixes across the skills scripts, run all self-tests and test suites, and mirror the updated files to `.agents/skills/`:

1. `skills/dagbani-linguistics/scripts/dagbani_g2p.py` (and mirror in `.agents/skills/dagbani-linguistics/scripts/dagbani_g2p.py`):
   - Ensure prevocalic `i` before front/open vowels transforms to glide `[j]` (e.g. `biɛɣu` -> `[bjɛ́ɣʊ́]` or `[bjɛ́ɣú]`).
   - Fix self-test assertions so `shikuru` and `biɛɣu` tests pass cleanly.

2. `skills/dagbani-linguistics/scripts/dagbani_syllabifier.py` (and mirror in `.agents/skills/dagbani-linguistics/scripts/dagbani_syllabifier.py`):
   - Align the self-test assertion for `biɛɣu` with the syllabifier's structure.

3. `skills/dagbani-linguistics/scripts/orthography_normalizer.py` (and mirror in `.agents/skills/dagbani-linguistics/scripts/orthography_normalizer.py`):
   - Remove the duplicate dictionary key `"nyela": "nyɛla"`.

4. `skills/dagbani-asr-whisper/scripts/audio_preprocessor.py` (and mirror in `.agents/skills/dagbani-asr-whisper/scripts/audio_preprocessor.py`):
   - Line 112: Fix 8-bit multi-channel audio downmixing `TypeError` by converting bytes to ints before subtraction: `sample = (int(b) - 128) / 128.0`.
   - Lines 176-193: Fix VAD segment truncation by ensuring any active speech segment is flushed when EOF is reached even if no trailing silence occurs.

5. `skills/dagbani-asr-whisper/scripts/evaluate_asr.py` (and mirror in `.agents/skills/dagbani-asr-whisper/scripts/evaluate_asr.py`):
   - Ensure `normalize_dagbani_text` removes Unicode combining tone diacritics (`Mn`, like `\u0301`, `\u0300`, `\u0304`) directly without replacing them with spaces (so `"zúŋɔ"` becomes `"zuŋɔ"`, not `"zu ŋɔ"`).

6. `skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py` (and mirror in `.agents/skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py`):
   - Ensure `phonemize_text("")` on empty string returns a valid dictionary including `"num_tokens": 0` and empty phoneme sequences without crashing.

7. Verification:
   Run all self-tests across all scripts:
   - `python skills/dagbani-linguistics/scripts/dagbani_g2p.py --self-test`
   - `python skills/dagbani-linguistics/scripts/orthography_normalizer.py --self-test`
   - `python skills/dagbani-linguistics/scripts/dagbani_syllabifier.py --self-test`
   - `python skills/dagbani-asr-whisper/scripts/audio_preprocessor.py --self-test`
   - `python skills/dagbani-asr-whisper/scripts/whisper_dagbani_trainer.py --self-test`
   - `python skills/dagbani-asr-whisper/scripts/evaluate_asr.py --self-test`
   - `python skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py --self-test`
   - `python skills/dagbani-tts-synthesis/scripts/prepare_tts_dataset.py --self-test`
   - `python skills/dagbani-tts-synthesis/scripts/synthesize_tts.py --self-test`
   - `python skills/dagbani-llm-tokenization-datasets/scripts/train_dagbani_tokenizer.py --self-test`
   - `python skills/dagbani-llm-tokenization-datasets/scripts/dataset_cleaner.py --self-test`
   - `python skills/dagbani-llm-tokenization-datasets/scripts/llm_lora_finetuner.py --self-test`
   - Also run `.agents/challenger_1_r2/run_all_adversarial_and_self_tests.py` and `.agents/challenger_2_r2/adversarial_test_suite.py` if available.

Ensure all 12 scripts pass 100% and mirror all modified files to `.agents/skills/`.

Document your changes and verification commands in `d:/ATS Tech/Dagbani AI/.agents/worker_remediation/handoff.md` and send a message when done.
