# Adversarial Stress Test & Verification Report (Challenger 2)

**Evaluator**: Challenger 2 (Speech Processing, Training Pipelines & Evaluation Adversarial Stress Tester)  
**Target Scope**:
1. `skills/dagbani-asr-whisper/scripts/audio_preprocessor.py`
2. `skills/dagbani-asr-whisper/scripts/whisper_dagbani_trainer.py`
3. `skills/dagbani-asr-whisper/scripts/evaluate_asr.py`
4. `skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py`
5. `skills/dagbani-tts-synthesis/scripts/prepare_tts_dataset.py`
6. `skills/dagbani-tts-synthesis/scripts/synthesize_tts.py`
7. `skills/dagbani-llm-tokenization-datasets/scripts/llm_lora_finetuner.py`

**Final Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

A total of 25 comprehensive adversarial stress tests were designed and evaluated against the speech processing, ASR/TTS training, dataset curation, and metric evaluation tool suite.

### Summary of Test Execution:
- **Total Adversarial Scenarios**: 25
- **Passed**: 21
- **Failed / Confirmed Vulnerabilities**: 4

### Detailed Empirical Observations:

#### Observation 1: `TypeError` in 8-Bit Multi-Channel Audio Ingestion
- **File**: `skills/dagbani-asr-whisper/scripts/audio_preprocessor.py`
- **Line Number**: 112
- **Verbatim Code**:
  ```python
  # Line 110-113:
  samples = []
  for i in range(0, total_samples, n_channels):
      mono_s = sum(raw_bytes[i:i+n_channels] - 128) / (n_channels * 128.0)
      samples.append(mono_s)
  ```
- **Error Triggered**: `TypeError: unsupported operand type(s) for -: 'bytes' and 'int'`
- **Observed Behavior**: `raw_bytes` is of type `bytes`. Slicing `raw_bytes[i:i+n_channels]` produces a `bytes` object (e.g., `b'\x80\xc8'`). Directly subtracting integer `128` from a `bytes` object crashes in Python with `TypeError`. (In contrast, line 108 iterates byte-by-byte producing `int` `b`, so `b - 128` works, but line 112 fails on multi-channel).

#### Observation 2: Silent Dropping of Active VAD Speech Segment at EOF
- **File**: `skills/dagbani-asr-whisper/scripts/audio_preprocessor.py`
- **Lines**: 176–193
- **Verbatim Code**:
  ```python
  for f, is_speech in enumerate(speech_frames):
      if is_speech:
          if not in_speech:
              in_speech = True
              start_frame = max(0, f - 2)
          silence_count = 0
      else:
          if in_speech:
              silence_count += 1
              if silence_count > silence_pad or f == len(speech_frames) - 1:
                  end_frame = f - silence_count
                  start_sec = (start_frame * frame_size) / sr
                  end_sec = (end_frame * frame_size) / sr
                  if end_sec - start_sec >= self.min_duration:
                      segments.append((round(start_sec, 2), round(end_sec, 2)))
                  in_speech = False
                  silence_count = 0
  ```
- **Observed Behavior**: The closure check `if silence_count > silence_pad or f == len(speech_frames) - 1:` is placed exclusively inside the `else:` branch (`if not is_speech`). When an audio file ends while speech is active (`is_speech == True` on frame `len(speech_frames) - 1`), the `else:` branch is never executed, and after the loop terminates there is no post-loop flush. For any speech file without trailing silence, `compute_energy_vad_segments` returns `[]` (empty list), dropping valid speech chunks.

#### Observation 3: Combining Tone Accents Splitting Words in ASR Normalization
- **File**: `skills/dagbani-asr-whisper/scripts/evaluate_asr.py`
- **Lines**: 86–90
- **Verbatim Code**:
  ```python
  cleaned = []
  for ch in text:
      cat = unicodedata.category(ch)
      if cat.startswith("L") or ch in {" ", "'", "-"}:
          cleaned.append(ch)
      else:
          cleaned.append(" ")
  ```
- **Observed Behavior**: In Unicode, combining tone diacritics (such as acute accent `\u0301` in `zúŋɔ`) have category `Mn` (Mark, nonspacing). Because `cat.startswith("L")` is `False`, `\u0301` is replaced with `" "`. `cleaned` becomes `['z', 'u', ' ', 'ŋ', 'ɔ']`, which normalizes to `"zu ŋɔ"` (2 words instead of 1). When evaluating transcripts with tone markings (e.g., BibleTTS or phonetically annotated datasets), word boundaries are corrupted, artificially inflating Word Error Rate (WER).

#### Observation 4: Schema Inconsistency on Empty String in Dagbani Phonemizer
- **File**: `skills/dagbani-tts-synthesis/scripts/dagbani_phonemizer.py`
- **Lines**: 311 vs 343
- **Verbatim Code**:
  ```python
  # Line 311 (Empty input):
  if not norm_text:
      return {"ipa": "", "tokens": [], "tones": [], "token_ids": []}

  # Line 343 (Non-empty input):
  return {
      "ipa": " ".join(ipa_symbols),
      "tokens": token_symbols,
      "tones": tone_symbols,
      "token_ids": token_ids,
      "num_tokens": len(token_ids)
  }
  ```
- **Observed Behavior**: Calling `phonemize("")` returns a dictionary missing the key `"num_tokens"`. Any calling script indexing `res["num_tokens"]` fails with `KeyError: 'num_tokens'`.

---

## 2. Logic Chain

1. **Audio Preprocessor 8-Bit Multi-channel**:
   - In standard 8-bit unsigned PCM WAV files with `n_channels > 1`, `raw_bytes[i:i+n_channels]` returns a `bytes` slice.
   - Python does not permit arithmetic between `bytes` and `int`.
   - Executing `AudioPreprocessor.load_wav_file()` on an 8-bit stereo or multi-channel audio file raises `TypeError`.
   - **Inference**: The audio preprocessor fails whenever 8-bit multi-channel audio is encountered.

2. **VAD Active Speech Truncation**:
   - Energy VAD detects active speech frames.
   - Speech ending at the final frame of an audio file means `is_speech` is `True` at `f == len(speech_frames) - 1`.
   - The loop terminates while `in_speech` is `True`.
   - Because no segment appending logic exists outside the loop, the final segment is discarded.
   - **Inference**: High-energy audio without trailing silence produces 0 segments, breaking VAD chunking for clipped or tightly-trimmed datasets.

3. **ASR Evaluator Tone Mark Token Splitting**:
   - Tone marks in African language orthographies and phonetic datasets use combining diacritics (Unicode category `Mn`).
   - Normalization replacing all non-letter characters with spaces splits accented characters into separate tokens.
   - Example: `"zúŋɔ"` -> `"zu ŋɔ"`, turning a 1-word utterance into a 2-word utterance.
   - **Inference**: Levenshtein alignment produces false insertion/substitution errors against tone-marked reference transcripts.

4. **Phonemizer Empty String Schema Integrity**:
   - Pipelines rely on uniform JSON dictionary schemas across all inputs.
   - Omission of `"num_tokens"` on empty or whitespace strings causes unhandled exceptions in downstream batch collation.

---

## 3. Caveats

- **PyTorch GPU Dependencies**: Tests were executed using self-contained pure-Python engines, standard libraries, and mathematical dry-run pipelines. Live CUDA GPU training loops were verified at the configuration generation, parameter calculation, and recipe export levels.
- **Parametric Acoustic Synthesis**: The parametric formant synthesizer is designed as a zero-dependency fallback engine. Real VITS neural checkpoints require external PyTorch model weights.
- **Robust Components**:
  - `whisper_dagbani_trainer.py`: LoRA parameter mathematics, target projections, and 8-bit quantization configs verified across all 5 Whisper sizes (Tiny, Base, Small, Medium, Large-v3).
  - `llm_lora_finetuner.py`: LLaMA-3.1 header templates, Dagbani system prompt, and corrupted JSONL error handling verified.
  - `prepare_tts_dataset.py`: Duration filtering, silence trimming, and VITS pipe-delimited filelists verified.

---

## 4. Conclusion & Required Actions

**Overall Risk Assessment**: **MEDIUM-HIGH** (2 High-Severity Bugs, 1 Medium-Severity Bug, 1 Low/Medium Schema Inconsistency).

### Required Remediations:

1. **Fix `audio_preprocessor.py:112`**:
   Replace:
   ```python
   mono_s = sum(raw_bytes[i:i+n_channels] - 128) / (n_channels * 128.0)
   ```
   With:
   ```python
   mono_s = sum((b - 128) for b in raw_bytes[i:i+n_channels]) / (n_channels * 128.0)
   ```

2. **Fix `audio_preprocessor.py:194` (Post-loop VAD Segment Flush)**:
   Add post-loop check before `return segments`:
   ```python
   if in_speech:
       end_frame = len(speech_frames) - silence_count
       start_sec = (start_frame * frame_size) / sr
       end_sec = (end_frame * frame_size) / sr
       if end_sec - start_sec >= self.min_duration:
           segments.append((round(start_sec, 2), round(end_sec, 2)))
   ```

3. **Fix `evaluate_asr.py:86–90` (Combining Diacritics)**:
   Update `normalize_dagbani_text`:
   ```python
   cleaned = []
   for ch in text:
       cat = unicodedata.category(ch)
       if cat.startswith("L") or ch in {" ", "'", "-"}:
           cleaned.append(ch)
       elif cat.startswith("M"):
           pass  # Discard combining diacritics without adding whitespace
       else:
           cleaned.append(" ")
   ```

4. **Fix `dagbani_phonemizer.py:311` (Schema Consistency)**:
   Update line 311:
   ```python
   if not norm_text:
       return {"ipa": "", "tokens": [], "tones": [], "token_ids": [], "num_tokens": 0}
   ```

---

## 5. Verification Method

To independently reproduce all findings and verify fixes:

1. Run the empirical adversarial test harness:
   ```bash
   python .agents/challenger_2_r2/adversarial_test_suite.py
   ```
2. Verify that all 25 tests report `[PASSED]`.
3. Invalidation conditions:
   - Any `TypeError` when loading 8-bit multi-channel audio files.
   - Any dropped speech segment on utterances ending without trailing silence.
   - Any word splitting when normalizing strings containing Unicode `Mn` combining accents (`\u0301`, `\u0300`).
   - Any `KeyError: 'num_tokens'` when phonemizing empty strings.
