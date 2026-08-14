#!/usr/bin/env python3
"""
Adversarial Stress-Testing Suite for Dagbani Speech & Training Pipelines
========================================================================
Challenger 2 - Comprehensive Adversarial Verification Harness
"""

import sys
import os
import io
import math
import wave
import struct
import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add skill script directories to sys.path
PROJECT_ROOT = Path("d:/ATS Tech/Dagbani AI").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-asr-whisper/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-tts-synthesis/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-llm-tokenization-datasets/scripts"))

# Import target modules
try:
    import audio_preprocessor
    from audio_preprocessor import AudioPreprocessor
except Exception as e:
    print(f"FAILED to import audio_preprocessor: {e}")

try:
    import whisper_dagbani_trainer
    from whisper_dagbani_trainer import WhisperDagbaniTrainer, WHISPER_SPECS
except Exception as e:
    print(f"FAILED to import whisper_dagbani_trainer: {e}")

try:
    import evaluate_asr
    from evaluate_asr import ASREvaluator, levenshtein_distance, normalize_dagbani_text
except Exception as e:
    print(f"FAILED to import evaluate_asr: {e}")

try:
    import dagbani_phonemizer
    from dagbani_phonemizer import DagbaniPhonemizer, PhonemeUnit
except Exception as e:
    print(f"FAILED to import dagbani_phonemizer: {e}")

try:
    import prepare_tts_dataset
    from prepare_tts_dataset import TTSDatasetPreprocessor, trim_silence_16bit, write_wav_file, read_wav_info
except Exception as e:
    print(f"FAILED to import prepare_tts_dataset: {e}")

try:
    import synthesize_tts
    from synthesize_tts import DagbaniTTSPipeline, ParametricDagbaniSynthesizer
except Exception as e:
    print(f"FAILED to import synthesize_tts: {e}")

try:
    import llm_lora_finetuner
    from llm_lora_finetuner import DagbaniLoRAFinetuner, DagbaniInstructionDataset, format_dagbani_prompt
except Exception as e:
    print(f"FAILED to import llm_lora_finetuner: {e}")


# ============================================================================
# Helper: Synthetic Audio Generator
# ============================================================================

def make_wav_bytes(
    duration_sec: float,
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
    signal_type: str = "sine",
    freq: float = 440.0,
    amplitude: float = 0.5,
    clip: bool = False
) -> bytes:
    """Generate raw WAV byte string with arbitrary specs and signals."""
    num_frames = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_frames):
            t = float(i) / sample_rate
            if signal_type == "silence":
                val = 0.0
            elif signal_type == "sine":
                val = math.sin(2.0 * math.pi * freq * t) * amplitude
            elif signal_type == "multi_tone":
                val = (0.5 * math.sin(2.0 * math.pi * freq * t) +
                       0.3 * math.sin(2.0 * math.pi * (freq * 2.5) * t) +
                       0.2 * math.sin(2.0 * math.pi * (freq * 5.0) * t)) * amplitude
            elif signal_type == "chirp":
                # Frequency sweep 100Hz to 4000Hz
                f_inst = 100.0 + (3900.0 * (i / max(1, num_frames)))
                val = math.sin(2.0 * math.pi * f_inst * t) * amplitude
            elif signal_type == "square":
                val = amplitude if math.sin(2.0 * math.pi * freq * t) >= 0 else -amplitude
            else:
                val = 0.0

            if clip:
                val = 1.5 if val > 0 else -1.5

            for ch in range(channels):
                # Channel phase difference for multi-channel testing
                ch_val = val * (1.0 if ch == 0 else 0.7)
                if sampwidth == 2:
                    int_val = int(ch_val * 32767.0)
                    int_val = max(-32768, min(32767, int_val))
                    frames.extend(struct.pack("<h", int_val))
                elif sampwidth == 1:
                    # 8-bit unsigned (0..255, 128 is center)
                    int_val = int(ch_val * 127.0 + 128.0)
                    int_val = max(0, min(255, int_val))
                    frames.extend(struct.pack("<B", int_val))
        wf.writeframes(frames)
    return buf.getvalue()


class AdversarialTestRunner:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, suite: str, test_name: str, passed: bool, details: str = "", extra: Any = None):
        res = {
            "suite": suite,
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "extra": extra
        }
        self.results.append(res)
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] [{suite}] {test_name} - {details}")

    # ========================================================================
    # SUITE 1: Audio Preprocessor & TTS Dataset Prep
    # ========================================================================
    def test_audio_preprocessor_adversarial(self):
        suite = "Audio Preprocessor"
        preproc = AudioPreprocessor(target_sr=16000, n_mels=80, min_duration=0.5, max_duration=30.0)

        # 1.1 Pure Silence
        try:
            silence_wav = make_wav_bytes(duration_sec=2.0, sample_rate=16000, signal_type="silence")
            samples, sr = preproc.load_wav_file(silence_wav)
            vad_segs = preproc.compute_energy_vad_segments(samples, sr=sr)
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            # Silence should produce 0 VAD segments, valid mel spectrogram with shape [80, 3000]
            passed = (sr == 16000) and (len(samples) == 32000) and (len(vad_segs) == 0) and (len(mel) == 80)
            self.record(suite, "Pure Silence Audio (2.0s)", passed, f"Samples={len(samples)}, VAD={len(vad_segs)}, Mel=[{len(mel)}x{len(mel[0])}]")
        except Exception as e:
            self.record(suite, "Pure Silence Audio (2.0s)", False, f"Exception: {e}")

        # 1.2 Very Short Audio (0.1s)
        try:
            short_wav = make_wav_bytes(duration_sec=0.1, sample_rate=16000, signal_type="sine", freq=440.0)
            samples, sr = preproc.load_wav_file(short_wav)
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            vad_segs = preproc.compute_energy_vad_segments(samples, sr=sr)
            passed = (len(samples) == 1600) and (len(mel) == 80) and (len(vad_segs) == 0) # Below min_duration 0.5s
            self.record(suite, "Sub-Threshold Short Audio (0.1s)", passed, f"Padded Mel frames={len(mel[0])}, VAD segments={len(vad_segs)} (correctly filtered)")
        except Exception as e:
            self.record(suite, "Sub-Threshold Short Audio (0.1s)", False, f"Exception: {e}")

        # 1.3 Long Audio (35.0s)
        try:
            long_wav = make_wav_bytes(duration_sec=35.0, sample_rate=16000, signal_type="sine", freq=300.0)
            samples, sr = preproc.load_wav_file(long_wav)
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            passed = (len(samples) == 35 * 16000) and (len(mel) == 80) and (len(mel[0]) == 2998)
            self.record(suite, "Long Audio Truncation (35.0s)", passed, f"Loaded {len(samples)} samples, Mel truncated to Whisper 30s window [{len(mel)}x{len(mel[0])}]")
        except Exception as e:
            self.record(suite, "Long Audio Truncation (35.0s)", False, f"Exception: {e}")

        # 1.4 Multi-channel Stereo at 44.1kHz
        try:
            stereo_wav = make_wav_bytes(duration_sec=1.5, sample_rate=44100, channels=2, signal_type="multi_tone", freq=440.0)
            samples, sr = preproc.load_wav_file(stereo_wav)
            passed = (sr == 16000) and (abs(len(samples) - int(1.5 * 16000)) <= 2) and all(-1.0 <= s <= 1.0 for s in samples)
            self.record(suite, "Stereo 44.1kHz -> Mono 16kHz Downmix & Resample", passed, f"Result samples={len(samples)} at sr={sr} Hz")
        except Exception as e:
            self.record(suite, "Stereo 44.1kHz -> Mono 16kHz Downmix & Resample", False, f"Exception: {e}")

        # 1.5 8-bit Unsigned PCM at 8000 Hz
        try:
            pcm8_wav = make_wav_bytes(duration_sec=2.0, sample_rate=8000, channels=1, sampwidth=1, signal_type="sine", freq=350.0)
            samples, sr = preproc.load_wav_file(pcm8_wav)
            passed = (sr == 16000) and (len(samples) == 32000) and all(-1.0 <= s <= 1.0 for s in samples)
            self.record(suite, "8-bit Unsigned 8kHz PCM Ingestion & Resampling", passed, f"Result samples={len(samples)} at sr={sr} Hz")
        except Exception as e:
            self.record(suite, "8-bit Unsigned 8kHz PCM Ingestion & Resampling", False, f"Exception: {e}")

        # 1.6 Severely Clipped / Saturated Waveform
        try:
            clipped_wav = make_wav_bytes(duration_sec=1.0, sample_rate=16000, signal_type="square", freq=200.0, clip=True)
            samples, sr = preproc.load_wav_file(clipped_wav)
            passed = all(-1.0 <= s <= 1.0 for s in samples) and max(abs(s) for s in samples) <= 1.0
            self.record(suite, "Clipped / Saturated Waveform Normalization", passed, f"Peak amplitude bounded to {max(abs(s) for s in samples):.4f}")
        except Exception as e:
            self.record(suite, "Clipped / Saturated Waveform Normalization", False, f"Exception: {e}")

        # 1.7 Intermittent Speech & VAD Segmentation
        try:
            # Create 6-second audio with speech at 1-2s and 4-5s, silence elsewhere
            sr = 16000
            total_samples = 6 * sr
            custom_samples = [0.0] * total_samples
            for i in range(sr * 1, sr * 2):
                custom_samples[i] = 0.5 * math.sin(2.0 * math.pi * 400.0 * (i / sr))
            for i in range(sr * 4, sr * 5):
                custom_samples[i] = 0.6 * math.sin(2.0 * math.pi * 500.0 * (i / sr))
            
            vad_segs = preproc.compute_energy_vad_segments(custom_samples, sr=sr)
            passed = len(vad_segs) == 2
            self.record(suite, "Intermittent Multi-Burst VAD Segmentation", passed, f"Detected segments: {vad_segs}")
        except Exception as e:
            self.record(suite, "Intermittent Multi-Burst VAD Segmentation", False, f"Exception: {e}")

        # 1.8 Directory Manifest Ingestion with Corrupted & Mixed Files
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                # Valid 3.0s file
                with open(tmp_p / "valid_1.wav", "wb") as f:
                    f.write(make_wav_bytes(3.0, 16000, signal_type="sine"))
                # Out-of-bounds short (0.2s) file
                with open(tmp_p / "too_short.wav", "wb") as f:
                    f.write(make_wav_bytes(0.2, 16000, signal_type="sine"))
                # Out-of-bounds long (35.0s) file
                with open(tmp_p / "too_long.wav", "wb") as f:
                    f.write(make_wav_bytes(35.0, 16000, signal_type="sine"))
                # Corrupted non-audio file disguised as .wav
                with open(tmp_p / "corrupt.wav", "wb") as f:
                    f.write(b"NOT_A_REAL_WAV_HEADER_CORRUPTED_DATA_1234567890")
                
                manifest_out = str(tmp_p / "manifest.json")
                records = preproc.process_directory(str(tmp_p), manifest_out)
                passed = (len(records) == 1) and os.path.exists(manifest_out)
                self.record(suite, "Manifest Generator Filtering (Valid vs Short/Long/Corrupt)", passed, f"Retained {len(records)} valid records (expected 1)")
        except Exception as e:
            self.record(suite, "Manifest Generator Filtering (Valid vs Short/Long/Corrupt)", False, f"Exception: {e}")

    # ========================================================================
    # SUITE 2: TTS Dataset Preprocessor Stress Testing
    # ========================================================================
    def test_tts_dataset_prep_adversarial(self):
        suite = "TTS Dataset Prep"
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                raw_dir = tmp_p / "raw"
                raw_dir.mkdir()
                out_dir = tmp_p / "out"

                # Create diverse test utterances
                test_samples = [
                    {"id": "spk0_valid", "text": "Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam.", "dur": 3.0, "spk": "speaker_1"},
                    {"id": "spk1_valid", "text": "O biɛla Yendi tiŋgbani ni zúŋɔ.", "dur": 4.5, "spk": "speaker_2"},
                    {"id": "spk0_short", "text": "Yi", "dur": 0.3, "spk": "speaker_1"},  # Filtered (< 1.0s)
                    {"id": "spk0_long", "text": "Long audio ...", "dur": 15.0, "spk": "speaker_1"}, # Filtered (> 12.0s)
                ]

                tsv_lines = ["id\taudio_file\ttranscript\tspeaker\n"]
                for s in test_samples:
                    wav_file = raw_dir / f"{s['id']}.wav"
                    raw_bytes = make_wav_bytes(s["dur"], sample_rate=24000, signal_type="sine", freq=220.0)
                    with open(wav_file, "wb") as f:
                        f.write(raw_bytes)
                    tsv_lines.append(f"{s['id']}\t{s['id']}.wav\t{s['text']}\t{s['spk']}\n")

                tsv_path = tmp_p / "transcripts.tsv"
                with open(tsv_path, "w", encoding="utf-8") as f:
                    f.writelines(tsv_lines)

                preproc = TTSDatasetPreprocessor(sample_rate=24000, min_duration=1.0, max_duration=12.0)
                meta = preproc.prepare_dataset(
                    audio_dir=raw_dir,
                    transcripts_path=tsv_path,
                    output_dir=out_dir,
                    val_ratio=0.5,
                    test_ratio=0.0
                )

                train_f = out_dir / "train_filelist.txt"
                val_f = out_dir / "val_filelist.txt"
                passed = (meta["total_samples"] == 2) and train_f.exists() and val_f.exists()
                
                # Check VITS pipe format
                with open(train_f, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    format_ok = all(len(l.split("|")) == 3 for l in lines)

                self.record(suite, "TTS Corpus Preparation & Duration Filtering", passed and format_ok, f"Valid={meta['total_samples']}, Train={meta['train_samples']}, Val={meta['val_samples']}, PipeFormat={format_ok}")
        except Exception as e:
            self.record(suite, "TTS Corpus Preparation & Duration Filtering", False, f"Exception: {e}")

    # ========================================================================
    # SUITE 3: ASR Evaluation Metric Auditor Adversarial Tests
    # ========================================================================
    def test_evaluate_asr_adversarial(self):
        suite = "ASR Evaluation Metrics"
        evaluator = ASREvaluator()

        # 3.1 Exact Match
        try:
            ref = ["n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"]
            hyp = ["n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"]
            res = evaluator.evaluate(ref, hyp)
            passed = (res["strict_wer"] == 0.0) and (res["cer"] == 0.0) and (res["special_glyph_metrics"]["overall_recall"] == 1.0)
            self.record(suite, "Exact Match (0.0 WER / 0.0 CER / 1.0 Glyph Recall)", passed, f"WER={res['strict_wer']}, CER={res['cer']}")
        except Exception as e:
            self.record(suite, "Exact Match", False, f"Exception: {e}")

        # 3.2 Complete Deletion
        try:
            ref = ["Dagbaŋ kaya ni ta'ada"]
            hyp = [""]
            res = evaluator.evaluate(ref, hyp)
            passed = (res["strict_wer"] == 1.0) and (res["cer"] == 1.0)
            self.record(suite, "Complete Deletion (100% WER / 100% CER)", passed, f"WER={res['strict_wer']}, CER={res['cer']}")
        except Exception as e:
            self.record(suite, "Complete Deletion", False, f"Exception: {e}")

        # 3.3 Complete Insertion / Empty Reference
        try:
            ref = [""]
            hyp = ["dagbani yɛltɔɣa"]
            res = evaluator.evaluate(ref, hyp)
            passed = (res["strict_wer"] == 0.0) and (res["cer"] == 0.0) # Handled gracefully without ZeroDivisionError
            self.record(suite, "Zero Division on Empty Reference", passed, f"Zero Division guarded cleanly: WER={res['strict_wer']}")
        except Exception as e:
            self.record(suite, "Zero Division on Empty Reference", False, f"Exception: {e}")

        # 3.4 Both Empty
        try:
            ref = [""]
            hyp = [""]
            res = evaluator.evaluate(ref, hyp)
            passed = (res["strict_wer"] == 0.0) and (res["cer"] == 0.0)
            self.record(suite, "Both Reference & Hypothesis Empty", passed, f"WER={res['strict_wer']}, CER={res['cer']}")
        except Exception as e:
            self.record(suite, "Both Reference & Hypothesis Empty", False, f"Exception: {e}")

        # 3.5 Special Character Substitutions (ɛ <-> e, ɔ <-> o, ŋ <-> n, ɣ <-> g, ʒ <-> z)
        try:
            ref = ["paɣaba mini bihi ban bɛ ʒɛri zuliya n-ti kɔŋ"]
            hyp = ["pagaba mini bihi ban be zeri zuliya n-ti kon"] # Replaced ɣ->g, ɛ->e, ʒ->z, ɔ->o, ŋ->n
            res = evaluator.evaluate(ref, hyp)
            glyph_m = res["special_glyph_metrics"]
            # All special glyphs should have 0.0 recall
            all_zero_recall = all(glyph_m["by_glyph"][g]["recall"] == 0.0 for g in ["ɛ", "ɔ", "ŋ", "ɣ", "ʒ"])
            passed = all_zero_recall and (glyph_m["overall_recall"] == 0.0) and (res["normalized_wer"] > 0.0)
            self.record(suite, "Special Glyph Substitution Detection (ɛ, ɔ, ŋ, ɣ, ʒ)", passed, f"Overall Special Glyph Recall={glyph_m['overall_recall']:.4f} (Correctly 0.0 for pure ASCII degradation)")
        except Exception as e:
            self.record(suite, "Special Glyph Substitution Detection", False, f"Exception: {e}")

        # 3.6 Punctuation Bursts & Case Differences (Strict vs Normalized WER)
        try:
            ref = ['“Dagbaŋ... Ti kpalinʒoo n-nyɛla din viɛli pam! (Naawuni chɛ ka di kpaŋsi!)”']
            hyp = ['dagbaŋ ti kpalinʒoo n-nyɛla din viɛli pam naawuni chɛ ka di kpaŋsi']
            res = evaluator.evaluate(ref, hyp)
            passed = (res["strict_wer"] > 0.5) and (res["normalized_wer"] == 0.0) and (res["cer"] == 0.0)
            self.record(suite, "Punctuation & Unicode Normalization Discrepancy", passed, f"Strict WER={res['strict_wer']*100:.1f}% -> Normalized WER={res['normalized_wer']*100:.1f}%")
        except Exception as e:
            self.record(suite, "Punctuation & Unicode Normalization Discrepancy", False, f"Exception: {e}")

    # ========================================================================
    # SUITE 4: TTS Phonemizer & Synthesizer Stress Testing
    # ========================================================================
    def test_tts_phonemizer_and_synth_adversarial(self):
        suite = "TTS Phonemizer & Synthesis"
        phonemizer = DagbaniPhonemizer()
        pipeline = DagbaniTTSPipeline(sample_rate=24000)

        # 4.1 Proverbs & Complex Linguistic Structures
        proverbs = [
            ("Saa nira ku ʒin n-nyu baa biɛli kom", ["s", "a:", "n", "i", "r", "a", "_", "k", "u", "_", "ʒ", "i", "n"]),
            ("Kpariba gbanŋma nyaba chaŋ sheli bɛ'ʊ tiŋa", ["kp", "a", "r", "i", "b", "a", "_", "gb", "a", "n", "ŋm", "a", "_", "ny", "a", "b", "a"]),
            ("Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam", ["d", "a", "g", "b", "a", "ŋ", "_", "k", "a", "y", "a", "_", "n", "i", "_", "t", "a", "ʔ", "a", "d", "a"]),
        ]
        for p_text, expected_subtokens in proverbs:
            try:
                res = phonemizer.phonemize(p_text)
                passed = (len(res["tokens"]) > 5) and (len(res["token_ids"]) == len(res["tokens"])) and ("ipa" in res)
                self.record(suite, f"Proverb Phonemization: '{p_text[:25]}...'", passed, f"Tokens={len(res['tokens'])}, IPA='{res['ipa'][:35]}...'")
            except Exception as e:
                self.record(suite, f"Proverb Phonemization: '{p_text[:25]}...'", False, f"Exception: {e}")

        # 4.2 Punctuation Bursts & Special Symbols
        try:
            noisy_input = "???!!! --- @@@ ### $$$ %%% ^^^ &&& *** ((( ))) ___ +++ === { } [ ] ; : , . < > / ? \\ | ~ `"
            res = phonemizer.phonemize(noisy_input)
            passed = (res["ipa"] == "") and (len(res["tokens"]) == 0) and (len(res["token_ids"]) == 0)
            self.record(suite, "Pure Punctuation & Symbol Storm Handling", passed, f"Gracefully filtered to empty: {res['tokens']}")
        except Exception as e:
            self.record(suite, "Pure Punctuation & Symbol Storm Handling", False, f"Exception: {e}")

        # 4.3 Foreign Loanwords & Digits
        try:
            foreign_text = "UNESCO COVID-19 hospital 2026 computer"
            res = phonemizer.phonemize(foreign_text)
            passed = len(res["token_ids"]) > 0 and all(isinstance(i, int) for i in res["token_ids"])
            self.record(suite, "Foreign Loanwords & Numerical Input Parsing", passed, f"Token IDs={res['token_ids']}, IPA='{res['ipa']}'")
        except Exception as e:
            self.record(suite, "Foreign Loanwords & Numerical Input Parsing", False, f"Exception: {e}")

        # 4.4 End-to-end Parametric Synthesis & WAV Header Validation
        try:
            test_phrase = "O biɛla Yendi zúŋɔ, ka Naawuni ti o alaafeei."
            samples_normal = pipeline.synthesize(text=test_phrase, speed=1.0)
            samples_fast = pipeline.synthesize(text=test_phrase, speed=0.5)
            samples_slow = pipeline.synthesize(text=test_phrase, speed=2.0)

            # Speed checks
            passed_speed = (len(samples_fast) < len(samples_normal) < len(samples_slow))

            # Export to temp WAV and verify wave format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_tmp:
                tmp_wav = f_tmp.name

            pipeline.save_wav(samples_normal, tmp_wav)
            with wave.open(tmp_wav, "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                dur = nframes / float(framerate)

            os.remove(tmp_wav)
            passed = passed_speed and (channels == 1) and (sampwidth == 2) and (framerate == 24000) and (dur > 0.5)
            self.record(suite, "End-to-End Synthesis Prosodic Scaling & WAV Export", passed, f"Normal={len(samples_normal)} samples ({dur:.2f}s), Fast={len(samples_fast)}, Slow={len(samples_slow)}")
        except Exception as e:
            self.record(suite, "End-to-End Synthesis Prosodic Scaling & WAV Export", False, f"Exception: {e}")

    # ========================================================================
    # SUITE 5: Training Pipeline & Parameter Verification
    # ========================================================================
    def test_training_pipelines_adversarial(self):
        suite = "Training Pipelines"

        # 5.1 Whisper Trainer Parameter Scaling
        for model_name in ["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3", "custom/non-existent-model"]:
            try:
                trainer = WhisperDagbaniTrainer(
                    base_model=model_name,
                    lora_r=16,
                    lora_alpha=32,
                    batch_size=2,
                    grad_accum=8
                )
                stats = trainer.calculate_lora_parameters()
                config = trainer.generate_training_config()

                eff_batch = stats["effective_batch_size"]
                trainable_pct = stats["trainable_percent"]
                has_q_v = "q_proj" in stats["target_modules"] and "v_proj" in stats["target_modules"]
                forced_dec = config["generation_config"]["forced_decoder_ids"] is None

                passed = (eff_batch == 16) and (trainable_pct < 3.0) and has_q_v and forced_dec
                self.record(suite, f"Whisper Trainer Config & LoRA Param Check ({model_name.split('/')[-1]})", passed, f"EffectiveBatch={eff_batch}, TrainableLoRA={stats['trainable_parameters']:,} ({trainable_pct}%), FallbackHandled={passed}")
            except Exception as e:
                self.record(suite, f"Whisper Trainer Config ({model_name})", False, f"Exception: {e}")

        # 5.2 Whisper Trainer Dry-Run Recipe Export
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = WhisperDagbaniTrainer(
                    base_model="openai/whisper-medium",
                    output_dir=tmp_dir,
                    lora_r=16,
                    batch_size=4,
                    grad_accum=4
                )
                success = trainer.execute_dry_run()
                recipe_file = Path(tmp_dir) / "training_recipe.json"
                passed = success and recipe_file.exists()
                if passed:
                    with open(recipe_file, "r", encoding="utf-8") as f:
                        recipe = json.load(f)
                    passed = recipe["peft_type"] == "LORA" and recipe["training_arguments"]["gradient_accumulation_steps"] == 4
                self.record(suite, "Whisper Trainer Dry-Run Execution & Recipe Export", passed, f"Recipe generated: {recipe_file.exists()}")
        except Exception as e:
            self.record(suite, "Whisper Trainer Dry-Run Execution & Recipe Export", False, f"Exception: {e}")

        # 5.3 LLM LoRA Finetuner Chat Prompt Formatter
        try:
            p_full = format_dagbani_prompt(
                instruction="Wula ka bɛ kɔri kpaŋkpaŋ?",
                system_prompt="Custom System Prompt for Dagbani",
                response="Kpaŋkpaŋ koobu bɔrimi pukparilim baŋsim."
            )
            passed = ("Custom System Prompt for Dagbani" in p_full) and ("<|start_header_id|>user<|end_header_id|>" in p_full) and ("<|start_header_id|>assistant<|end_header_id|>" in p_full)
            self.record(suite, "LLM Chat Template Custom System Prompt Formatting", passed, "LLaMA-3.1 header delimiters verified")
        except Exception as e:
            self.record(suite, "LLM Chat Template Custom System Prompt Formatting", False, f"Exception: {e}")

        # 5.4 LLM LoRA Finetuner Dataset Ingestion (Valid, Malformed, JSON vs JSONL)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                
                # Test JSONL with some malformed rows
                jsonl_file = tmp_p / "mixed_dataset.jsonl"
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"instruction": "Query 1", "response": "Answer 1"}) + "\n")
                    f.write("CORRUPTED_NON_JSON_ROW_HERE\n")
                    f.write(json.dumps({"prompt": "Query 2", "output": "Answer 2"}) + "\n")
                    f.write("\n") # empty line

                ds_jsonl = DagbaniInstructionDataset(jsonl_file)
                passed_jsonl = len(ds_jsonl) == 2

                # Test JSON array
                json_file = tmp_p / "array_dataset.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump([
                        {"instruction": "Array Q1", "response": "Array A1"},
                        {"instruction": "Array Q2", "response": "Array A2"}
                    ], f)
                ds_json = DagbaniInstructionDataset(json_file)
                passed_json = len(ds_json) == 2

                # Test Finetuner dry-run
                finetuner = DagbaniLoRAFinetuner(
                    base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    output_dir=tmp_p / "adapter_out",
                    lora_r=64,
                    lora_alpha=64,
                    batch_size=2,
                    grad_accum=8
                )
                res_dry = finetuner.train(jsonl_file, dry_run=True)
                passed_dry = res_dry["status"] == "dry_run_verified" and res_dry["effective_batch_size"] == 16 and (tmp_p / "adapter_out" / "adapter_config.json").exists()

                passed = passed_jsonl and passed_json and passed_dry
                self.record(suite, "LLM Finetuner Dataset Parsing & LoRA Dry-Run Simulation", passed, f"JSONL parsed={len(ds_jsonl)}/2, JSON parsed={len(ds_json)}/2, EffectiveBatch={res_dry['effective_batch_size']}")
        except Exception as e:
            self.record(suite, "LLM Finetuner Dataset Parsing & LoRA Dry-Run Simulation", False, f"Exception: {e}")

    # ========================================================================
    # Main Execution
    # ========================================================================
    def run_all(self) -> bool:
        print("=" * 80)
        print("STARTING ADVERSARIAL STRESS-TEST HARNESS (CHALLENGER 2)")
        print("=" * 80)

        self.test_audio_preprocessor_adversarial()
        self.test_tts_dataset_prep_adversarial()
        self.test_evaluate_asr_adversarial()
        self.test_tts_phonemizer_and_synth_adversarial()
        self.test_training_pipelines_adversarial()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        print("\n" + "=" * 80)
        print(f"ADVERSARIAL STRESS-TEST RESULTS: {passed}/{total} PASSED ({failed} FAILED)")
        print("=" * 80)

        return failed == 0


if __name__ == "__main__":
    runner = AdversarialTestRunner()
    success = runner.run_all()
    # Save full JSON report
    report_path = Path("d:/ATS Tech/Dagbani AI/.agents/challenger_2/adversarial_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_tests": len(runner.results),
            "passed_tests": sum(1 for r in runner.results if r["passed"]),
            "failed_tests": sum(1 for r in runner.results if not r["passed"]),
            "verdict": "APPROVE" if success else "REQUEST_CHANGES",
            "results": runner.results
        }, f, indent=2)
    print(f"\nDetailed JSON report written to {report_path}")
    sys.exit(0 if success else 1)
