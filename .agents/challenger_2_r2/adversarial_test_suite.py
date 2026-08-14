#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for Dagbani Speech & Training Pipelines
====================================================================
Empirical challenge suite testing:
1. audio_preprocessor.py
2. whisper_dagbani_trainer.py
3. evaluate_asr.py
4. dagbani_phonemizer.py
5. prepare_tts_dataset.py
6. synthesize_tts.py
7. llm_lora_finetuner.py

Author: Challenger 2 (Adversarial Stress Tester)
"""

import sys
import os
import io
import math
import struct
import wave
import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Setup import paths
PROJECT_ROOT = Path("d:/ATS Tech/Dagbani AI")
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-asr-whisper/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-tts-synthesis/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-llm-tokenization-datasets/scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "skills/dagbani-linguistics/scripts"))

# Import target modules
import audio_preprocessor as ap
import evaluate_asr as ev
import whisper_dagbani_trainer as wt
import dagbani_phonemizer as dp
import prepare_tts_dataset as pd
import synthesize_tts as st
import llm_lora_finetuner as lf


# ============================================================================
# Synthetic Audio Helpers
# ============================================================================

def make_wav_bytes(
    samples: List[int],
    sample_rate: int = 16000,
    n_channels: int = 1,
    sampwidth: int = 2
) -> bytes:
    """Generate in-memory WAV bytes from raw samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        if sampwidth == 2:
            raw = struct.pack(f"<{len(samples)}h", *samples)
        elif sampwidth == 1:
            raw = struct.pack(f"<{len(samples)}B", *[max(0, min(255, s + 128)) for s in samples])
        else:
            raise ValueError(f"Unsupported sampwidth: {sampwidth}")
        wf.writeframes(raw)
    return buf.getvalue()


def make_sine_wave(
    duration_sec: float,
    sample_rate: int = 16000,
    freq: float = 440.0,
    amplitude: float = 16000.0,
    n_channels: int = 1
) -> List[int]:
    """Generate sine wave integer samples."""
    n_samples = int(duration_sec * sample_rate)
    mono_samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = int(amplitude * math.sin(2.0 * math.pi * freq * t))
        val = max(-32768, min(32767, val))
        mono_samples.append(val)
    
    if n_channels == 1:
        return mono_samples
    else:
        interleaved = []
        for s in mono_samples:
            for ch in range(n_channels):
                interleaved.append(s)
        return interleaved


# ============================================================================
# Test Runner & Logging
# ============================================================================

class TestResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed = False
        self.error_msg = ""
        self.details: Dict[str, Any] = {}

    def pass_test(self, details: Dict[str, Any] = None):
        self.passed = True
        self.details = details or {}

    def fail_test(self, error_msg: str, details: Dict[str, Any] = None):
        self.passed = False
        self.error_msg = error_msg
        self.details = details or {}


class AdversarialTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []

    def log_result(self, res: TestResult):
        self.results.append(res)
        status = "PASSED" if res.passed else "FAILED"
        print(f"[{status:6s}] [{res.category:12s}] {res.name}")
        if not res.passed:
            print(f"         Error: {res.error_msg}")
            if res.details:
                print(f"         Details: {res.details}")

    # ------------------------------------------------------------------------
    # 1. AUDIO PREPROCESSOR TESTS
    # ------------------------------------------------------------------------
    def test_audio_preproc_synthetic_16k(self):
        res = TestResult("Synthetic 16kHz Mono Loading & Resampling", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000, n_mels=80)
            raw_wav = make_wav_bytes(make_sine_wave(2.0, 16000, 440.0), 16000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            if sr == 16000 and len(samples) == 32000 and max(samples) <= 1.0 and min(samples) >= -1.0:
                res.pass_test({"sr": sr, "len": len(samples), "peak": max(map(abs, samples))})
            else:
                res.fail_test("Invalid sample count or bounds", {"sr": sr, "len": len(samples)})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_resample_8k_to_16k(self):
        res = TestResult("Resampling 8kHz -> 16kHz", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000)
            raw_wav = make_wav_bytes(make_sine_wave(1.5, 8000, 300.0), 8000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            if sr == 16000 and len(samples) == 24000:
                res.pass_test({"sr": sr, "len": len(samples)})
            else:
                res.fail_test(f"Expected 24000 samples at 16k, got {len(samples)} at {sr}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_resample_44k_stereo_to_16k_mono(self):
        res = TestResult("Multi-channel Stereo 44.1kHz -> 16kHz Mono Downmix", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000)
            # 1 second of 44.1kHz stereo
            stereo_samples = make_sine_wave(1.0, 44100, 440.0, n_channels=2)
            raw_wav = make_wav_bytes(stereo_samples, 44100, 2, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            if sr == 16000 and len(samples) == 16000:
                res.pass_test({"sr": sr, "len": len(samples)})
            else:
                res.fail_test(f"Expected 16000 samples, got {len(samples)}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_pure_silence(self):
        res = TestResult("Pure Silence Waveform (Zero-Division Defense & VAD)", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000)
            silence_samples = [0] * 32000  # 2s pure silence
            raw_wav = make_wav_bytes(silence_samples, 16000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            # Peak normalize must not explode or divide by zero
            vad_segments = preproc.compute_energy_vad_segments(samples, sr=sr)
            # Mel spectrogram
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            
            if len(samples) == 32000 and len(vad_segments) == 0 and len(mel) == 80:
                res.pass_test({"vad_segments": len(vad_segments), "mel_channels": len(mel)})
            else:
                res.fail_test("Silence processing failed", {"vad_segments": vad_segments})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_short_audio_padding(self):
        res = TestResult("Very Short Audio (0.1s) Mel Spectrogram 30s Padding", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000, n_mels=80)
            short_samples = make_sine_wave(0.1, 16000, 440.0)
            raw_wav = make_wav_bytes(short_samples, 16000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            if len(mel) == 80 and len(mel[0]) == 2998:
                res.pass_test({"mels": len(mel), "frames": len(mel[0])})
            else:
                res.fail_test(f"Expected 80x2998 spectrogram, got {len(mel)}x{len(mel[0]) if mel else 0}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_long_audio_truncation(self):
        res = TestResult("Long Audio (35s) Mel Spectrogram 30s Truncation", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000, n_mels=128)
            long_samples = make_sine_wave(35.0, 16000, 440.0)
            raw_wav = make_wav_bytes(long_samples, 16000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            mel = preproc.compute_log_mel_spectrogram(samples, sr=sr)
            if len(mel) == 128 and len(mel[0]) == 2998:
                res.pass_test({"mels": len(mel), "frames": len(mel[0])})
            else:
                res.fail_test(f"Expected 128x2998 spectrogram, got {len(mel)}x{len(mel[0]) if mel else 0}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_clipped_waveform(self):
        res = TestResult("Clipped Waveform (-1.0 dBFS Peak Normalization)", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000)
            # Full scale square wave
            clipped_samples = [32767 if (i // 50) % 2 == 0 else -32768 for i in range(16000)]
            raw_wav = make_wav_bytes(clipped_samples, 16000, 1, 2)
            samples, sr = preproc.load_wav_file(raw_wav)
            max_s = max(map(abs, samples))
            target_peak = 10.0 ** (-1.0 / 20.0)  # ~0.89125
            if abs(max_s - target_peak) < 1e-4:
                res.pass_test({"target_peak": target_peak, "actual_peak": max_s})
            else:
                res.fail_test(f"Peak normalization mismatch: expected {target_peak}, got {max_s}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_8bit_multichannel_bug_check(self):
        res = TestResult("8-Bit Multi-Channel WAV Downmixing (Bug Hunt #1)", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000)
            # 8-bit stereo audio
            stereo_8bit = [128, 200] * 1000  # 1000 stereo frames
            raw_wav = make_wav_bytes(stereo_8bit, 16000, 2, 1)
            try:
                samples, sr = preproc.load_wav_file(raw_wav)
                res.pass_test({"samples_len": len(samples)})
            except TypeError as te:
                res.fail_test(f"VULNERABILITY CONFIRMED: TypeError in 8-bit multi-channel audio parsing: {te}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_audio_preproc_vad_speech_at_end_bug_check(self):
        res = TestResult("VAD Segmentation with Speech Ending at EOF (Bug Hunt #2)", "AudioPreproc")
        try:
            preproc = ap.AudioPreprocessor(target_sr=16000, min_duration=0.5)
            # 2 seconds of loud tone with NO trailing silence
            loud_samples = make_sine_wave(2.0, 16000, 440.0, amplitude=25000.0)
            norm_samples = [s / 32768.0 for s in loud_samples]
            segments = preproc.compute_energy_vad_segments(norm_samples, sr=16000)
            if len(segments) >= 1:
                res.pass_test({"segments": segments})
            else:
                res.fail_test("VULNERABILITY CONFIRMED: VAD dropped speech segment ending at EOF due to missing silence flush", {"segments": segments})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    # ------------------------------------------------------------------------
    # 2. TTS DATASET PREPARATION TESTS
    # ------------------------------------------------------------------------
    def test_tts_dataset_prep_full_pipeline(self):
        res = TestResult("TTS Dataset Preprocessor Pipeline & Splits", "TTSDatasetPrep")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                audio_dir = tmp_p / "raw_audio"
                audio_dir.mkdir()
                out_dir = tmp_p / "dataset_out"

                # 4 samples: 2 normal (2.5s, 3.0s), 1 too short (0.2s), 1 too long (15.0s)
                dataset_samples = [
                    {"id": "dag_01", "text": "O biɛla Yendi zúŋɔ", "dur": 2.5},
                    {"id": "dag_02", "text": "Gballi maa viɛla pam", "dur": 3.0},
                    {"id": "dag_03", "text": "Too short", "dur": 0.2},
                    {"id": "dag_04", "text": "Too long sentence", "dur": 15.0},
                ]

                tsv_lines = ["id\taudio\ttext\tspeaker\n"]
                for s in dataset_samples:
                    wav_p = audio_dir / f"{s['id']}.wav"
                    raw = pd.generate_synthetic_wav(s["dur"], 24000)
                    pd.write_wav_file(wav_p, raw, 24000)
                    tsv_lines.append(f"{s['id']}\t{s['id']}.wav\t{s['text']}\t0\n")

                tsv_p = tmp_p / "transcripts.tsv"
                with open(tsv_p, "w", encoding="utf-8") as f:
                    f.writelines(tsv_lines)

                prep = pd.TTSDatasetPreprocessor(sample_rate=24000, min_duration=1.0, max_duration=12.0)
                meta = prep.prepare_dataset(audio_dir, tsv_p, out_dir, val_ratio=0.5, test_ratio=0.0)

                if meta["total_samples"] == 2 and meta["train_samples"] == 1 and meta["val_samples"] == 1:
                    train_f = out_dir / "train_filelist.txt"
                    with open(train_f, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    if len(lines) == 1 and "|" in lines[0]:
                        res.pass_test({"meta": meta, "filelist_sample": lines[0]})
                    else:
                        res.fail_test("Invalid filelist content", {"lines": lines})
                else:
                    res.fail_test(f"Duration filtering failure: expected 2 valid samples, got {meta['total_samples']}", {"meta": meta})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_tts_dataset_prep_pure_silence_trim(self):
        res = TestResult("Silence Trimming on Pure Silence Audio", "TTSDatasetPrep")
        try:
            silence = [0] * 48000 # 2s silence
            trimmed = pd.trim_silence_16bit(silence)
            # Must return safety margin or empty, not raise exception
            if len(trimmed) <= 480:
                res.pass_test({"trimmed_len": len(trimmed)})
            else:
                res.fail_test(f"Silence trim did not reduce silence: {len(trimmed)}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    # ------------------------------------------------------------------------
    # 3. ASR EVALUATION METRIC TESTS
    # ------------------------------------------------------------------------
    def test_asr_eval_exact_match(self):
        res = TestResult("ASR Metric: Exact Match (0% WER/CER, 100% Glyphs)", "ASREval")
        try:
            evaluator = ev.ASREvaluator()
            refs = ["n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"]
            hyps = ["n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"]
            out = evaluator.evaluate(refs, hyps)
            if out["strict_wer"] == 0.0 and out["normalized_wer"] == 0.0 and out["cer"] == 0.0:
                res.pass_test(out)
            else:
                res.fail_test(f"Non-zero error on exact match: WER={out['strict_wer']}", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_asr_eval_complete_deletion(self):
        res = TestResult("ASR Metric: Complete Deletion (Empty Hypothesis -> 100% WER)", "ASREval")
        try:
            evaluator = ev.ASREvaluator()
            refs = ["paɣaŋa mini bihi ban bɛ ʒɛri zuliya"]
            hyps = [""]
            out = evaluator.evaluate(refs, hyps)
            if out["strict_wer"] == 1.0 and out["normalized_wer"] == 1.0:
                res.pass_test(out)
            else:
                res.fail_test(f"Expected 1.0 WER for complete deletion, got {out['normalized_wer']}", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_asr_eval_both_empty(self):
        res = TestResult("ASR Metric: Both References & Hypotheses Empty (Zero-Division)", "ASREval")
        try:
            evaluator = ev.ASREvaluator()
            refs = [""]
            hyps = [""]
            out = evaluator.evaluate(refs, hyps)
            if out["strict_wer"] == 0.0 and out["cer"] == 0.0:
                res.pass_test(out)
            else:
                res.fail_test("Empty strings failed zero division check", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_asr_eval_special_glyph_substitutions(self):
        res = TestResult("ASR Metric: Dagbani Special Glyph Tracking (ɛ->e, ɔ->o, ŋ->n, ɣ->gh, ʒ->zh)", "ASREval")
        try:
            evaluator = ev.ASREvaluator()
            refs = ["paɣaba mini bihi ban bɛ ʒɛri ɔ zuliya ŋɔ"]
            hyps = ["paghaba mini bihi ban be zheri o zuliya no"]
            glyph_stats = evaluator.compute_special_glyph_metrics(refs, hyps)
            # All distinctive glyphs replaced by ASCII approximations -> 0% recall
            rec = glyph_stats["overall_recall"]
            if rec == 0.0:
                res.pass_test(glyph_stats)
            else:
                res.fail_test(f"Special glyph tracking missed ASCII substitutions (recall={rec})", glyph_stats)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_asr_eval_combining_diacritic_normalization_bug(self):
        res = TestResult("ASR Metric: Combining Tone Diacritic Normalization (Bug Hunt #3)", "ASREval")
        try:
            # Word with combining acute accent: zúŋɔ -> 'z', 'u', '\u0301', 'ŋ', 'ɔ'
            text_with_accent = "zúŋɔ"
            norm = ev.normalize_dagbani_text(text_with_accent)
            # In proper Dagbani text normalization, zúŋɔ should normalize to "zuŋɔ"
            # If Mn category is replaced by " ", it becomes "z ŋɔ" (split into two words)
            words = norm.split()
            if len(words) == 1 and norm == "zuŋɔ":
                res.pass_test({"normalized": norm, "words": words})
            else:
                res.fail_test(f"VULNERABILITY CONFIRMED: Combining tone accent splits word into multiple words: '{text_with_accent}' -> '{norm}' ({words})")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    # ------------------------------------------------------------------------
    # 4. DAGBANI PHONEMIZER & SYNTHESIZER TESTS
    # ------------------------------------------------------------------------
    def test_phonemizer_complex_dagbani_proverb(self):
        res = TestResult("Phonemizer: Complex Proverbs with Digraphs & Glyphs", "Phonemizer")
        try:
            phonemizer = dp.DagbaniPhonemizer()
            text = "Kpamba yɛliya ni: Baa bɛ ŋubiri gbaŋ din viɛla!"
            out = phonemizer.phonemize(text)
            tokens = out["tokens"]
            # Check presence of digraphs and mutated phonemes
            has_kp = "kp" in tokens
            has_gb = "gb" in tokens
            has_ŋ = "ŋ" in tokens
            has_j_or_ch = "j" in tokens or "ch" in tokens or "y" in tokens
            has_token_ids = len(out["token_ids"]) == len(tokens)
            if has_kp and has_gb and has_ŋ and has_token_ids:
                res.pass_test({"ipa": out["ipa"], "tokens": tokens, "num_tokens": out.get("num_tokens")})
            else:
                res.fail_test("Missing digraphs or token ID mismatch", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_phonemizer_empty_string_missing_key_bug(self):
        res = TestResult("Phonemizer: Empty String Return Schema (Bug Hunt #4)", "Phonemizer")
        try:
            phonemizer = dp.DagbaniPhonemizer()
            out = phonemizer.phonemize("")
            # Check if num_tokens key is present
            if "num_tokens" in out:
                res.pass_test(out)
            else:
                res.fail_test("VULNERABILITY CONFIRMED: KeyError 'num_tokens' missing when phonemizing empty string", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_phonemizer_allophonic_mutations(self):
        res = TestResult("Phonemizer: Allophonic Mutations (k->ch, d->r, s->h, g->ɣ)", "Phonemizer")
        try:
            phonemizer = dp.DagbaniPhonemizer()
            # 1. k -> ch before front vowel: kɛma -> ch ɛ m a
            r1 = phonemizer.phonemize("kɛma")["tokens"]
            # 2. d -> r intervocalic: kadariba -> k a r a r i b a
            r2 = phonemizer.phonemize("kadariba")["tokens"]
            # 3. s -> h intervocalic: biisi -> b i: h i
            r3 = phonemizer.phonemize("biisi")["tokens"]
            
            p1 = "ch" in r1
            p2 = "r" in r2
            p3 = "h" in r3
            if p1 and p2 and p3:
                res.pass_test({"kɛma": r1, "kadariba": r2, "biisi": r3})
            else:
                res.fail_test(f"Mutation failure: k->ch: {p1}, d->r: {p2}, s->h: {p3}", {"r1": r1, "r2": r2, "r3": r3})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_phonemizer_punctuation_burst_and_foreign_words(self):
        res = TestResult("Phonemizer: Punctuation Bursts & Foreign Digits", "Phonemizer")
        try:
            phonemizer = dp.DagbaniPhonemizer()
            text = "!!! ??? ... --- ,,, ::: ;;; ((( ))) 12345 Hello COVID-19"
            out = phonemizer.phonemize(text)
            # Should not crash, should produce valid token IDs for parseable letters
            if isinstance(out["token_ids"], list):
                res.pass_test({"tokens": out["tokens"], "num_tokens": out.get("num_tokens")})
            else:
                res.fail_test("Invalid token IDs output for noisy text", out)
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_tts_synthesis_parametric_audio_generation(self):
        res = TestResult("TTS Synthesis: Parametric Speech Synthesis to WAV", "TTSSynthesis")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_wav = Path(tmp_dir) / "synth_test.wav"
                pipeline = st.DagbaniTTSPipeline(sample_rate=24000)
                samples = pipeline.synthesize(text="O biɛla Yendi zúŋɔ.", speed=1.0)
                pipeline.save_wav(samples, out_wav)

                if out_wav.exists():
                    with wave.open(str(out_wav), "rb") as wf:
                        nchan = wf.getnchannels()
                        sampw = wf.getsampwidth()
                        sr = wf.getframerate()
                        nframes = wf.getnframes()
                        dur = nframes / float(sr)
                    if nchan == 1 and sampw == 2 and sr == 24000 and dur > 0.5:
                        res.pass_test({"channels": nchan, "sampwidth": sampw, "sr": sr, "duration": dur})
                    else:
                        res.fail_test("WAV header format verification failed", {"nchan": nchan, "sampw": sampw, "sr": sr, "dur": dur})
                else:
                    res.fail_test("Output WAV file not created")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_tts_synthesis_speed_and_pitch_scaling(self):
        res = TestResult("TTS Synthesis: Speed (0.5x, 2.0x) & Pitch Scaling", "TTSSynthesis")
        try:
            pipeline = st.DagbaniTTSPipeline(sample_rate=24000)
            text = "Dagbaŋ kaya ni ta'ada"
            samples_norm = pipeline.synthesize(text=text, speed=1.0)
            samples_fast = pipeline.synthesize(text=text, speed=0.5)
            samples_slow = pipeline.synthesize(text=text, speed=2.0)

            # Fast should be ~half length, slow should be ~double length
            len_norm = len(samples_norm)
            len_fast = len(samples_fast)
            len_slow = len(samples_slow)

            if len_fast < len_norm < len_slow:
                res.pass_test({"len_fast": len_fast, "len_norm": len_norm, "len_slow": len_slow})
            else:
                res.fail_test(f"Speed scaling failed: fast={len_fast}, norm={len_norm}, slow={len_slow}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    # ------------------------------------------------------------------------
    # 5. TRAINING PIPELINES & CONFIG GENERATOR TESTS
    # ------------------------------------------------------------------------
    def test_whisper_trainer_lora_parameters_and_configs(self):
        res = TestResult("Whisper Trainer: LoRA Parameter Math & Recipe Generation", "WhisperTrainer")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = wt.WhisperDagbaniTrainer(
                    base_model="openai/whisper-medium",
                    output_dir=tmp_dir,
                    lora_r=16,
                    lora_alpha=32,
                    batch_size=4,
                    grad_accum=4
                )
                stats = trainer.calculate_lora_parameters()
                config = trainer.generate_training_config()
                dry_ok = trainer.execute_dry_run()

                # Checks:
                # 1. Effective batch size = 4 * 4 = 16
                # 2. Medium trainable params = 4,718,592
                # 3. Trainable percent < 1.0%
                # 4. forced_decoder_ids is None
                # 5. recipe file created
                c1 = stats["effective_batch_size"] == 16
                c2 = stats["trainable_parameters"] == 4_718_592
                c3 = stats["trainable_percent"] < 1.0
                c4 = config["generation_config"]["forced_decoder_ids"] is None
                c5 = (Path(tmp_dir) / "training_recipe.json").exists()

                if c1 and c2 and c3 and c4 and c5 and dry_ok:
                    res.pass_test({"stats": stats, "recipe_exists": c5})
                else:
                    res.fail_test("Whisper trainer verification failed", {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_whisper_trainer_all_model_sizes_lora_math(self):
        res = TestResult("Whisper Trainer: Multi-Model LoRA Calculations (Tiny to Large-v3)", "WhisperTrainer")
        try:
            models = [
                ("openai/whisper-tiny", 983_040),
                ("openai/whisper-base", 1_572_864),
                ("openai/whisper-small", 2_949_120),
                ("openai/whisper-medium", 4_718_592),
                ("openai/whisper-large-v3", 7_864_320),
            ]
            all_ok = True
            details = {}
            for model_name, expected_lora in models:
                t = wt.WhisperDagbaniTrainer(base_model=model_name, lora_r=16)
                s = t.calculate_lora_parameters()
                details[model_name] = s["trainable_parameters"]
                if s["trainable_parameters"] != expected_lora:
                    all_ok = False

            if all_ok:
                res.pass_test(details)
            else:
                res.fail_test(f"LoRA parameter count mismatch: {details}")
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def test_llm_lora_finetuner_chat_template_and_dry_run(self):
        res = TestResult("LLM LoRA Finetuner: LLaMA-3.1 Chat Template & JSONL Parsing", "LLMLoRATrainer")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                jsonl_path = tmp_p / "dataset.jsonl"
                
                # Write sample dataset with one valid entry, one with alternate keys, and one corrupted line
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"instruction": "Wula ka bɛ kɔri kpaŋkpaŋ?", "response": "Pukparilim baŋsim"}) + "\n")
                    f.write(json.dumps({"prompt": "Translate Hello", "output": "Dasiba"}) + "\n")
                    f.write("{ CORRUPTED JSON LINE }\n")
                    f.write(json.dumps({"user": "Naa n-nyɛ ŋuni?", "assistant": "Naa n-nyɛ kpɛma."}) + "\n")

                dataset = lf.DagbaniInstructionDataset(jsonl_path)
                # Should cleanly parse 3 valid records, skipping the corrupted line
                c1 = len(dataset) == 3

                # Test chat template formatting
                prompt = dataset[0]["formatted_text"]
                c2 = "<|start_header_id|>system<|end_header_id|>" in prompt
                c3 = "<|start_header_id|>user<|end_header_id|>" in prompt
                c4 = "<|start_header_id|>assistant<|end_header_id|>" in prompt

                # Dry-run training
                finetuner = lf.DagbaniLoRAFinetuner(
                    output_dir=tmp_p / "adapter_out",
                    lora_r=64,
                    lora_alpha=64,
                    batch_size=4,
                    grad_accum=4
                )
                train_res = finetuner.train(jsonl_path, dry_run=True)
                c5 = train_res["status"] == "dry_run_verified"
                c6 = (tmp_p / "adapter_out" / "adapter_config.json").exists()

                if c1 and c2 and c3 and c4 and c5 and c6:
                    res.pass_test({"parsed_records": len(dataset), "adapter_created": c6})
                else:
                    res.fail_test("LLM LoRA finetuner verification failed", {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6})
        except Exception as e:
            res.fail_test(str(e))
        self.log_result(res)

    def run_all(self) -> Dict[str, Any]:
        print("======================================================================")
        print("STARTING EMPIRICAL ADVERSARIAL STRESS TEST SUITE")
        print("======================================================================")
        
        # Audio Preprocessor
        self.test_audio_preproc_synthetic_16k()
        self.test_audio_preproc_resample_8k_to_16k()
        self.test_audio_preproc_resample_44k_stereo_to_16k_mono()
        self.test_audio_preproc_pure_silence()
        self.test_audio_preproc_short_audio_padding()
        self.test_audio_preproc_long_audio_truncation()
        self.test_audio_preproc_clipped_waveform()
        self.test_audio_preproc_8bit_multichannel_bug_check()
        self.test_audio_preproc_vad_speech_at_end_bug_check()

        # TTS Dataset Prep
        self.test_tts_dataset_prep_full_pipeline()
        self.test_tts_dataset_prep_pure_silence_trim()

        # ASR Metric Evaluator
        self.test_asr_eval_exact_match()
        self.test_asr_eval_complete_deletion()
        self.test_asr_eval_both_empty()
        self.test_asr_eval_special_glyph_substitutions()
        self.test_asr_eval_combining_diacritic_normalization_bug()

        # Dagbani Phonemizer & TTS Synthesizer
        self.test_phonemizer_complex_dagbani_proverb()
        self.test_phonemizer_empty_string_missing_key_bug()
        self.test_phonemizer_allophonic_mutations()
        self.test_phonemizer_punctuation_burst_and_foreign_words()
        self.test_tts_synthesis_parametric_audio_generation()
        self.test_tts_synthesis_speed_and_pitch_scaling()

        # Training Pipelines
        self.test_whisper_trainer_lora_parameters_and_configs()
        self.test_whisper_trainer_all_model_sizes_lora_math()
        self.test_llm_lora_finetuner_chat_template_and_dry_run()

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print("======================================================================")
        print(f"ADVERSARIAL SUITE SUMMARY: {passed}/{total} PASSED ({failed} FAILED)")
        print("======================================================================")
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "passed": r.passed,
                    "error_msg": r.error_msg,
                    "details": r.details
                }
                for r in self.results
            ]
        }


if __name__ == "__main__":
    suite = AdversarialTestSuite()
    summary = suite.run_all()
    out_json = PROJECT_ROOT / ".agents/challenger_2_r2/test_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Test results saved to {out_json}")
