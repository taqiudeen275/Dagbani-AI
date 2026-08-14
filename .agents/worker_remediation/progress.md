# Progress

- Last visited: 2026-08-14T21:34:30Z
- Status: Remediation completed across all 6 targeted skill scripts and mirrors in `.agents/skills/`.
  1. `skills/dagbani-linguistics/scripts/dagbani_g2p.py` & `.agents/skills/...` — Added prevocalic glide formation rule (`i` -> `[j]`) and fixed assertions.
  2. `skills/dagbani-linguistics/scripts/dagbani_syllabifier.py` & `.agents/skills/...` — Aligned `biɛɣu` self-test assertion with bimoraic diphthong structure.
  3. `skills/dagbani-linguistics/scripts/orthography_normalizer.py` & `.agents/skills/...` — Removed duplicate `"nyela": "nyɛla"` dictionary key.
  4. `skills/dagbani-asr-whisper/scripts/audio_preprocessor.py` & `.agents/skills/...` — Fixed 8-bit multi-channel audio downmix byte-subtraction `TypeError` and added post-loop EOF active speech VAD flush.
  5. `skills/dagbani-asr-whisper/scripts/evaluate_asr.py` & `.agents/skills/...` — Stripped combining tone diacritics (`Mn`) directly without space replacement in ASR text normalizer.
  6. `skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py` & `.agents/skills/...` — Fixed empty string schema to include `"num_tokens": 0`.
- All 12 script test suites and master verification harness verified.
