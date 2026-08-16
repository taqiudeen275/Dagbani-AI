"""Build the audited Dagbani ASR Kaggle notebooks.

The notebooks are generated from this file so their large JSON representation stays
reviewable and reproducible.  Run this script after editing a cell and then run the
contract tests in ``tests/test_dagbani_asr_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
PHASE1_PATH = NOTEBOOK_DIR / "01_Dagbani_ASR_Data_Audit_and_Baselines_Kaggle.ipynb"
BASELINE_PATH = NOTEBOOK_DIR / "01b_Dagbani_ASR_Baseline_Only_Kaggle.ipynb"
PHASE2_PATH = NOTEBOOK_DIR / "02_Dagbani_ASR_Whisper_Small_Training_Kaggle.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str, *tags: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict], title: str, accelerator: str = "none") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": accelerator, "title": title},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PHASE1_CELLS = [
    markdown(
        r"""
        # Dagbani ASR Phase 1 — Data Audit and Comparable Baselines

        This is the **CPU-first evidence notebook**. It discovers real corpora, creates
        the canonical manifest, audits audio/transcript quality and split leakage, then
        optionally benchmarks Whisper models on one frozen WAXAL evaluation set.

        Safety rules:

        - Missing data is reported as `unavailable`, `gated`, or `rejected`; it is never
          replaced with synthetic audio.
        - `transcript_raw` is immutable. Evaluation normalization only applies Unicode
          NFC, lowercase, whitespace cleanup, and punctuation removal.
        - WAXAL's official train/validation/test split is preserved. The test split is
          sealed and is never used for training or pseudo-labelling.
        - Run `quick` mode first. Use `full` only after inspecting the source report.
        - Keep the accelerator **off** until the optional baseline section.
        """
    ),
    markdown("## 1. Install a bounded, Kaggle-compatible environment"),
    code(
        r"""
        import importlib.util
        import subprocess
        import sys

        INSTALL_DEPS = True
        REQUIRED = {
            "datasets": "datasets[audio]>=3.2,<5",
            "transformers": "transformers>=4.48,<6",
            "accelerate": "accelerate>=1.2,<3",
            "huggingface_hub": "huggingface_hub>=0.27,<2",
            "jiwer": "jiwer>=3.0,<5",
            "soundfile": "soundfile>=0.12,<1",
            "librosa": "librosa>=0.10,<1",
            "pyarrow": "pyarrow>=17,<25",
            "pandas": "pandas>=2.1,<4",
        }
        missing = [spec for module, spec in REQUIRED.items() if importlib.util.find_spec(module) is None]
        if INSTALL_DEPS and missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("Environment ready. PyTorch is intentionally not replaced.")
        """,
        "setup",
    ),
    markdown("## 2. Configuration — quick mode is intentionally the default"),
    code(
        r"""
        from __future__ import annotations

        import dataclasses
        import gc
        import gzip
        import hashlib
        import importlib.metadata
        import io
        import json
        import os
        import platform
        import re
        import string
        import subprocess
        import textwrap
        import time
        import unicodedata
        from collections import Counter, defaultdict
        from dataclasses import asdict, dataclass, field
        from pathlib import Path
        from typing import Any, Iterable, Iterator

        import numpy as np
        import pandas as pd
        import requests
        import soundfile as sf
        from IPython.display import display


        def kaggle_secret(name: str) -> str | None:
            value = os.getenv(name)
            if value:
                return value
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return None


        @dataclass
        class AuditConfig:
            audit_mode: str = "quick"  # quick | full
            work_dir: str = "/kaggle/working/dagbani_asr_phase1"
            hf_token: str | None = field(default_factory=lambda: kaggle_secret("HF_TOKEN"))
            manifest_repo_id: str = os.getenv("DAGBANI_MANIFEST_REPO", "")
            publish_artifacts: bool = False
            run_baselines: bool = False
            baseline_limit: int | None = 256
            metadata_rows_per_split: int = 2_000
            audio_rows_per_split: int = 100
            unlabeled_metadata_rows: int = 5_000
            unlabeled_audio_rows: int = 200
            spell4wiki_quick_limit: int = 500
            seed: int = 42
            min_duration_s: float = 0.5
            max_duration_s: float = 30.0
            min_words_per_s: float = 0.25
            max_words_per_s: float = 8.0
            waxal_benchmark_min_duration_s: float = 1.5
            waxal_benchmark_max_words_per_s: float = 4.0
            common_voice_root: str = os.getenv("COMMON_VOICE_DAG_ROOT", "")
            local_manifest_paths: tuple[str, ...] = ()
            # Empty means every discoverable source. For the first supervised
            # milestone use ("waxal_dag_asr",) and audit Bible data later.
            enabled_sources: tuple[str, ...] = ()
            allow_network_audio_downloads: bool = False

            def validate(self) -> None:
                if self.audit_mode not in {"quick", "full"}:
                    raise ValueError("audit_mode must be 'quick' or 'full'")
                if self.publish_artifacts and not self.manifest_repo_id:
                    raise ValueError("Set DAGBANI_MANIFEST_REPO before publishing")
                if self.publish_artifacts and not self.hf_token:
                    raise ValueError("HF_TOKEN is required for private publishing")

            @property
            def metadata_limit(self) -> int | None:
                return None if self.audit_mode == "full" else self.metadata_rows_per_split

            @property
            def audio_limit(self) -> int | None:
                return None if self.audit_mode == "full" else self.audio_rows_per_split


        cfg = AuditConfig()
        cfg.validate()
        WORK_DIR = Path(cfg.work_dir)
        ARTIFACT_DIR = WORK_DIR / "artifacts"
        CACHE_DIR = WORK_DIR / "hf_cache"
        for directory in (WORK_DIR, ARTIFACT_DIR, CACHE_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(CACHE_DIR)
        os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
        print(json.dumps(asdict(cfg) | {"hf_token": "<set>" if cfg.hf_token else "<missing>"}, indent=2))
        """
    ),
    markdown("## 3. Canonical text and audio helpers"),
    code(
        r"""
        CANONICAL_COLUMNS = [
            "sample_id", "source", "source_version", "audio_locator",
            "transcript_raw", "transcript_eval", "speaker_id", "duration_s",
            "domain", "split", "license", "audio_hash", "is_pseudo",
            "pseudo_confidence",
        ]
        DAGBANI_GLYPHS = set("ɛɔŋɣʒƐƆŊƔƷ")
        PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation + "“”‘’…–—"})


        def normalize_eval(text: Any) -> str:
            '''Minimal evaluation-only normalization; never transliterate letters.'''
            value = unicodedata.normalize("NFC", str(text or "")).lower()
            value = value.translate(PUNCT_TRANSLATION)
            return " ".join(value.split())


        def strict_text(text: Any) -> str:
            return unicodedata.normalize("NFC", str(text or "")).strip()


        def unexpected_characters(text: str) -> list[str]:
            allowed_categories = {"Ll", "Lu", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd", "Zs", "Po", "Pd"}
            return sorted({ch for ch in text if unicodedata.category(ch) not in allowed_categories and ch not in "\n\t"})


        def stable_split(group_id: str, train: int = 80, validation: int = 10) -> str:
            bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            if bucket < train:
                return "train"
            if bucket < train + validation:
                return "validation"
            return "test"


        def bible_group(record: dict[str, Any], locator: str) -> str | None:
            for key in ("book_chapter", "chapter", "book", "group_id"):
                value = record.get(key)
                if value not in (None, ""):
                    book = record.get("book", "book")
                    return f"{book}:{value}"
            match = re.search(r"(?i)(genesis|exodus|leviticus|numbers|deuteronomy|[1-3]?\s*[a-z]+)[_\-/ ]*(\d{1,3})", locator)
            return f"{match.group(1).lower()}:{match.group(2)}" if match else None


        def audio_payload(value: Any) -> tuple[np.ndarray, int, bytes | None]:
            '''Decode HF Audio dicts, torchcodec decoders, local paths, or byte payloads.'''
            raw_bytes = None
            if hasattr(value, "get_all_samples"):
                samples = value.get_all_samples()
                array = samples.data.detach().cpu().numpy()
                sample_rate = int(samples.sample_rate)
            elif isinstance(value, dict) and value.get("array") is not None:
                array = np.asarray(value["array"], dtype=np.float32)
                sample_rate = int(value.get("sampling_rate") or 16_000)
                raw_bytes = value.get("bytes")
            else:
                path = None
                if isinstance(value, dict):
                    raw_bytes = value.get("bytes")
                    path = value.get("path")
                elif isinstance(value, (bytes, bytearray)):
                    raw_bytes = bytes(value)
                elif isinstance(value, (str, os.PathLike)):
                    path = str(value)
                if raw_bytes:
                    array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                elif path and re.match(r"https?://", path):
                    if not cfg.allow_network_audio_downloads:
                        raise PermissionError("Network audio download is disabled")
                    response = requests.get(path, timeout=60)
                    response.raise_for_status()
                    raw_bytes = response.content
                    array, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                elif path:
                    array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
                else:
                    raise ValueError("Unsupported or empty audio payload")
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:
                array = array.mean(axis=0 if array.shape[0] <= 8 else 1)
            return array.reshape(-1), int(sample_rate), raw_bytes


        def canonical_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
            import librosa
            if sample_rate != 16_000:
                array = librosa.resample(array, orig_sr=sample_rate, target_sr=16_000)
            peak = float(np.max(np.abs(array))) if len(array) else 0.0
            if peak > 1.0:
                array = array / peak
            return np.clip(array, -1.0, 1.0).astype(np.float32)


        def exact_audio_hash(array: np.ndarray) -> str:
            pcm16 = np.round(np.clip(array, -1, 1) * 32767).astype("<i2")
            return hashlib.sha256(pcm16.tobytes()).hexdigest()


        def spectral_fingerprint(array: np.ndarray) -> str:
            '''A compact codec-tolerant log-mel fingerprint for near-duplicate grouping.'''
            import librosa
            if len(array) < 800:
                return ""
            mel = librosa.feature.melspectrogram(y=array, sr=16_000, n_fft=400, hop_length=320, n_mels=24)
            log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
            old_x = np.linspace(0.0, 1.0, log_mel.shape[1])
            new_x = np.linspace(0.0, 1.0, 32)
            resized = np.vstack([np.interp(new_x, old_x, row) for row in log_mel])
            quantized = np.clip(np.round((resized + 80.0) * 3.0), 0, 255).astype(np.uint8)
            return hashlib.sha256(quantized.tobytes()).hexdigest()


        assert normalize_eval("  N NYƐLA—BƐ!  ") == "n nyɛla bɛ"
        assert strict_text("e\u0301") == "é"
        print("Canonical schema and conservative normalization checks passed.")
        """
    ),
    markdown("## 4. Source registry and availability discovery"),
    code(
        r"""
        from datasets import Audio, get_dataset_config_info, get_dataset_config_names, load_dataset


        @dataclass(frozen=True)
        class SourceSpec:
            name: str
            repo_id: str
            config_name: str | None
            domain: str
            expected_license: str
            required: bool = False
            text_fields: tuple[str, ...] = ("transcription", "sentence", "text", "transcript")
            audio_fields: tuple[str, ...] = ("audio", "speech")
            speaker_fields: tuple[str, ...] = ("speaker_id", "client_id", "speaker", "user_id")
            id_fields: tuple[str, ...] = ("id", "sample_id", "path", "file", "filename")


        HF_SOURCES = [
            SourceSpec("waxal_dag_asr", "google/WaxalNLP", "dag_asr", "spontaneous", "cc-by-4.0", True),
            SourceSpec("dagbani_bible", "ghananlpcommunity/dagbani-bible-audio-text-tts", None, "bible_read", "cc-by-nc-4.0"),
            SourceSpec("ghana_speech_dag", "ghananlpcommunity/ghana-speech", "Dagbani_dag", "read_speech", "unknown"),
            SourceSpec("navigation_dagbani", "ghananlpcommunity/navigation-corpus-dagbani-speech", None, "navigation", "cc-by-nc-4.0"),
            SourceSpec("health_unicef_dagbani", "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani", None, "health", "unknown"),
            SourceSpec("youth_conversations_dagbani", "ghananlpcommunity/youth-conversations-dag", None, "conversation", "unknown"),
        ]


        def split_info_value(value: Any, field_name: str) -> int | None:
            '''Read Hugging Face split metadata across datasets library versions.'''
            if isinstance(value, dict):
                raw = value.get(field_name)
            else:
                raw = getattr(value, field_name, None)
            return int(raw) if raw is not None else None


        def discover_hf_source(spec: SourceSpec) -> dict[str, Any]:
            started = time.time()
            report = asdict(spec) | {"status": "unavailable", "splits": {}, "reported_license": ""}
            try:
                configs = get_dataset_config_names(spec.repo_id, token=cfg.hf_token)
                config_name = spec.config_name
                if config_name and config_name not in configs:
                    raise ValueError(f"config {config_name!r} not found; available={configs[:20]}")
                if config_name is None and len(configs) == 1:
                    config_name = configs[0]
                info = get_dataset_config_info(spec.repo_id, config_name, token=cfg.hf_token)
                report["resolved_config"] = config_name
                report["reported_license"] = str(getattr(info, "license", "") or "")
                report["splits"] = {
                    name: {
                        "num_examples": split_info_value(value, "num_examples"),
                        "num_bytes": split_info_value(value, "num_bytes"),
                    }
                    for name, value in (getattr(info, "splits", {}) or {}).items()
                }
                explicit_license = report["reported_license"] or spec.expected_license
                report["status"] = "available" if explicit_license != "unknown" else "gated"
                report["license"] = explicit_license
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
            report["elapsed_s"] = round(time.time() - started, 2)
            return report


        source_reports = [discover_hf_source(spec) for spec in HF_SOURCES]

        cv_root = Path(cfg.common_voice_root) if cfg.common_voice_root else None
        cv_report = {
            "name": "common_voice_25_dag",
            "status": "available" if cv_root and (cv_root / "validated.tsv").exists() else "gated",
            "domain": "scripted_read",
            "license": "cc0-1.0",
            "locator": str(cv_root or "Set COMMON_VOICE_DAG_ROOT after accepting Mozilla terms"),
        }
        source_reports.append(cv_report)

        local_reports = []
        for raw_path in cfg.local_manifest_paths:
            path = Path(raw_path)
            local_reports.append({
                "name": f"local:{path.stem}", "status": "available" if path.exists() else "unavailable",
                "domain": "user_supplied", "license": "gated", "locator": str(path),
            })
        source_reports.extend(local_reports)
        source_status_df = pd.DataFrame(source_reports)
        display(source_status_df[[column for column in ["name", "status", "domain", "license", "locator", "error"] if column in source_status_df]])
        print(source_status_df[[column for column in ["name", "status", "resolved_config", "license", "error"] if column in source_status_df]].to_string(index=False))
        """
    ),
    markdown("## 5. Bounded adapters build the canonical manifest"),
    code(
        r"""
        def first_field(record: dict[str, Any], candidates: Iterable[str]) -> str | None:
            return next((name for name in candidates if name in record), None)


        def source_split(spec: SourceSpec, original_split: str, record: dict[str, Any], locator: str, speaker: str) -> str:
            if spec.name == "waxal_dag_asr":
                return original_split
            if "bible" in spec.name:
                group = bible_group(record, locator)
                return stable_split(group) if group else "external_test"
            if speaker and speaker != "unknown":
                return stable_split(f"{spec.name}:{speaker}")
            return "external_test"


        def audio_locator(spec: SourceSpec, split: str, sample_id: str, value: Any) -> str:
            if isinstance(value, dict) and value.get("path"):
                return str(value["path"])
            if isinstance(value, (str, os.PathLike)):
                return str(value)
            return f"hf://{spec.repo_id}/{spec.config_name or 'default'}/{split}/{sample_id}"


        def canonical_record(
            spec: SourceSpec,
            original_split: str,
            record: dict[str, Any],
            row_index: int,
            decode_audio: bool,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
            text_field = first_field(record, spec.text_fields)
            audio_field = first_field(record, spec.audio_fields)
            id_field = first_field(record, spec.id_fields)
            speaker_field = first_field(record, spec.speaker_fields)
            sample_id = strict_text(record.get(id_field)) if id_field else f"{original_split}-{row_index:09d}"
            text = strict_text(record.get(text_field)) if text_field else ""
            speaker = strict_text(record.get(speaker_field)) if speaker_field else "unknown"
            speaker_id = f"{spec.name}:{speaker}" if speaker and speaker != "unknown" else "unknown"
            value = record.get(audio_field) if audio_field else None
            locator = audio_locator(spec, original_split, sample_id, value)
            split = source_split(spec, original_split, record, locator, speaker)
            split_policy = "official" if spec.name == "waxal_dag_asr" else "speaker_or_group"
            if "bible" in spec.name and split == "external_test":
                # The current 16-word Bible corpus exposes only audio/text/duration.
                # Keep adjacent aligned segments together in large contiguous blocks
                # when explicit book/chapter metadata is absent.
                group = f"contiguous-block-{row_index // 500:06d}"
                split = stable_split(f"{spec.name}:{group}")
                split_policy = "contiguous_500_row_fallback"
            license_name = spec.expected_license
            declared_duration = record.get("duration_s", record.get("duration", np.nan))
            try:
                declared_duration = float(declared_duration)
            except (TypeError, ValueError):
                declared_duration = np.nan
            base = {
                "sample_id": sample_id,
                "source": spec.name,
                "source_version": spec.config_name or "default",
                "audio_locator": locator,
                "transcript_raw": text,
                "transcript_eval": normalize_eval(text),
                # Speaker identifiers are source-local. Namespace them so numeric or
                # short IDs from unrelated corpora cannot create false leakage.
                "speaker_id": speaker_id,
                "speaker_id_raw": speaker or "unknown",
                "duration_s": declared_duration,
                "domain": spec.domain,
                "split": split,
                "origin_split": original_split,
                "split_policy": split_policy,
                "license": license_name,
                "audio_hash": "",
                "is_pseudo": False,
                "pseudo_confidence": np.nan,
            }
            is_unlabeled = original_split == "unlabeled"
            if license_name == "unknown":
                return None, base | {"reason": "license_not_verified"}, None
            if not audio_field:
                return None, base | {"reason": "audio_field_missing"}, None
            if not text and not is_unlabeled:
                return None, base | {"reason": "transcript_missing"}, None
            if unexpected_characters(text):
                base["unexpected_characters"] = "".join(unexpected_characters(text))
            if not decode_audio:
                return base, None, None
            try:
                array, sample_rate, _ = audio_payload(value)
                array = canonical_audio(array, sample_rate)
                duration = len(array) / 16_000
                base["duration_s"] = round(duration, 4)
                base["audio_hash"] = exact_audio_hash(array)
                near_hash = spectral_fingerprint(array)
                words_per_s = len(base["transcript_eval"].split()) / max(duration, 1e-6) if text else np.nan
                if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
                    return None, base | {"reason": "duration_out_of_range", "words_per_s": words_per_s}, near_hash
                if text and not (cfg.min_words_per_s <= words_per_s <= cfg.max_words_per_s):
                    return None, base | {"reason": "speech_rate_out_of_range", "words_per_s": words_per_s}, near_hash
                base["words_per_s"] = words_per_s
                return base, None, near_hash
            except Exception as exc:
                return None, base | {"reason": "audio_decode_failed", "error": f"{type(exc).__name__}: {exc}"}, None


        def scan_hf_source(spec: SourceSpec, report: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
            accepted, rejected, near = [], [], []
            if cfg.enabled_sources and spec.name not in cfg.enabled_sources:
                print(f"{spec.name}: skipped by enabled_sources")
                return accepted, rejected, near
            if report.get("status") != "available":
                return accepted, rejected, near
            config_name = report.get("resolved_config", spec.config_name)
            resolved_spec = dataclasses.replace(
                spec,
                config_name=config_name,
                expected_license=str(report.get("license") or spec.expected_license),
            )
            split_names = list((report.get("splits") or {}).keys()) or ["train"]
            for split_name in split_names:
                split_started = time.time()
                accepted_before, rejected_before = len(accepted), len(rejected)
                stream = load_dataset(spec.repo_id, config_name, split=split_name, streaming=True, token=cfg.hf_token)
                # Full mode decodes every supervised row, but deliberately samples the
                # very large unlabeled partition. Pseudo-labelling streams it later.
                limit = cfg.unlabeled_metadata_rows if split_name == "unlabeled" else cfg.metadata_limit
                audio_limit = cfg.unlabeled_audio_rows if split_name == "unlabeled" else cfg.audio_limit
                reported_rows = ((report.get("splits") or {}).get(split_name) or {}).get("num_examples")
                target_rows = min(limit, reported_rows) if limit is not None and reported_rows else (limit or reported_rows)
                for index, record in enumerate(stream):
                    if limit is not None and index >= limit:
                        break
                    decode = audio_limit is None or index < audio_limit
                    good, bad, fingerprint = canonical_record(resolved_spec, split_name, record, index, decode)
                    if good:
                        accepted.append(good)
                        if fingerprint:
                            near.append({"sample_id": good["sample_id"], "source": spec.name, "near_hash": fingerprint})
                    if bad:
                        rejected.append(bad)
                    processed = index + 1
                    if processed % 250 == 0:
                        elapsed = max(time.time() - split_started, 1e-6)
                        rows_per_s = processed / elapsed
                        eta_s = ((target_rows - processed) / rows_per_s) if target_rows and rows_per_s else None
                        eta_text = f"{eta_s / 60:.1f}m" if eta_s is not None else "unknown"
                        print(
                            f"{spec.name}/{split_name}: {processed:,}/{target_rows or '?'} rows "
                            f"({rows_per_s:.2f} rows/s, ETA {eta_text})"
                        )
                print(
                    f"{spec.name}/{split_name}: accepted={len(accepted) - accepted_before:,}, "
                    f"rejected={len(rejected) - rejected_before:,}, elapsed={(time.time() - split_started) / 60:.1f}m"
                )
            return accepted, rejected, near


        accepted_rows, rejected_rows, near_hash_rows = [], [], []
        report_by_name = {report["name"]: report for report in source_reports}
        for source_spec in HF_SOURCES:
            good, bad, near = scan_hf_source(source_spec, report_by_name[source_spec.name])
            accepted_rows.extend(good)
            rejected_rows.extend(bad)
            near_hash_rows.extend(near)
        print(f"HF scan complete: {len(accepted_rows):,} accepted metadata rows, {len(rejected_rows):,} rejected rows")
        """
    ),
    markdown("### Optional Common Voice, local-manifest, and Spell4Wiki adapters"),
    code(
        r"""
        def scan_common_voice(root: Path) -> tuple[list[dict], list[dict]]:
            accepted, rejected = [], []
            for split_name, filename in (("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv")):
                tsv = root / filename
                if not tsv.exists():
                    continue
                table = pd.read_csv(tsv, sep="\t")
                limit = cfg.metadata_limit or len(table)
                for index, row in table.head(limit).iterrows():
                    clip = root / "clips" / str(row["path"])
                    record = {
                        "id": str(row["path"]), "audio": str(clip), "sentence": row.get("sentence", ""),
                        "client_id": row.get("client_id", "unknown"),
                    }
                    spec = SourceSpec("common_voice_25_dag", "local", None, "scripted_read", "cc0-1.0")
                    good, bad, _ = canonical_record(spec, split_name, record, int(index), cfg.audit_mode == "full")
                    (accepted if good else rejected).append(good or bad)
            return accepted, rejected


        if cv_root and (cv_root / "validated.tsv").exists():
            good, bad = scan_common_voice(cv_root)
            accepted_rows.extend(good)
            rejected_rows.extend(bad)


        def scan_local_manifest(path: Path) -> tuple[list[dict], list[dict]]:
            table = pd.read_json(path, lines=True) if path.suffix.lower() in {".json", ".jsonl"} else pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
            required = {"audio", "transcript", "license", "source"}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            accepted, rejected = [], []
            limit = cfg.metadata_limit or len(table)
            for index, row in table.head(limit).iterrows():
                spec = SourceSpec(str(row["source"]), "local", None, str(row.get("domain", "user_supplied")), str(row["license"]), text_fields=("transcript",))
                record = row.to_dict() | {"id": row.get("sample_id", f"row-{index}"), "speaker_id": row.get("speaker_id", "unknown")}
                good, bad, _ = canonical_record(spec, str(row.get("split", "train")), record, int(index), cfg.audit_mode == "full")
                (accepted if good else rejected).append(good or bad)
            return accepted, rejected


        for local_path in map(Path, cfg.local_manifest_paths):
            if local_path.exists():
                good, bad = scan_local_manifest(local_path)
                accepted_rows.extend(good)
                rejected_rows.extend(bad)


        def spell4wiki_manifest(limit: int | None) -> tuple[list[dict], list[dict]]:
            endpoint = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query", "format": "json", "formatversion": 2,
                "generator": "categorymembers", "gcmtitle": "Category:Files uploaded by spell4wiki in dag",
                "gcmtype": "file", "gcmlimit": "max", "prop": "imageinfo",
                "iiprop": "url|sha1|mime|extmetadata",
            }
            accepted, rejected, continuation = [], [], {}
            while limit is None or len(accepted) + len(rejected) < limit:
                response = requests.get(endpoint, params=params | continuation, timeout=60)
                response.raise_for_status()
                payload = response.json()
                for page in payload.get("query", {}).get("pages", []):
                    title = page.get("title", "")
                    info = (page.get("imageinfo") or [{}])[0]
                    metadata = info.get("extmetadata") or {}
                    license_name = ((metadata.get("LicenseShortName") or {}).get("value") or "unknown").lower()
                    filename = re.sub(r"^File:", "", title, flags=re.I)
                    transcript = re.sub(r"\.(ogg|oga|wav|mp3)$", "", filename, flags=re.I)
                    transcript = re.sub(r"^dag[-_ ]", "", transcript, flags=re.I).replace("_", " ").strip()
                    admitted_for_audio = cfg.audit_mode == "full" and cfg.allow_network_audio_downloads
                    row = {
                        "sample_id": str(page.get("pageid", info.get("sha1", filename))),
                        "source": "spell4wiki_commons", "source_version": "wikimedia-api",
                        "audio_locator": info.get("url", ""), "transcript_raw": strict_text(transcript),
                        "transcript_eval": normalize_eval(transcript), "speaker_id": "unknown",
                        "duration_s": np.nan, "domain": "scripted_read",
                        "split": "external_test" if admitted_for_audio else "candidate_external",
                        "license": license_name, "audio_hash": info.get("sha1", ""),
                        "is_pseudo": False, "pseudo_confidence": np.nan,
                    }
                    if license_name == "unknown" or not row["audio_locator"]:
                        rejected.append(row | {"reason": "license_or_audio_url_missing"})
                    elif admitted_for_audio:
                        try:
                            array, sample_rate, _ = audio_payload(row["audio_locator"])
                            array = canonical_audio(array, sample_rate)
                            row["duration_s"] = len(array) / 16_000
                            row["audio_hash"] = exact_audio_hash(array)
                            if not (cfg.min_duration_s <= row["duration_s"] <= cfg.max_duration_s):
                                rejected.append(row | {"reason": "duration_out_of_range"})
                            else:
                                accepted.append(row)
                        except Exception as exc:
                            rejected.append(row | {"reason": "audio_decode_failed", "error": f"{type(exc).__name__}: {exc}"})
                    else:
                        accepted.append(row)
                    if limit is not None and len(accepted) + len(rejected) >= limit:
                        break
                continuation = payload.get("continue") or {}
                if not continuation:
                    break
            return accepted, rejected


        spell_limit = None if cfg.audit_mode == "full" else cfg.spell4wiki_quick_limit
        try:
            good, bad = spell4wiki_manifest(spell_limit)
            accepted_rows.extend(good)
            rejected_rows.extend(bad)
            source_reports.append({
                "name": "spell4wiki_commons", "status": "available" if good else "rejected",
                "domain": "scripted_read", "license": "per-file Commons license",
                "rows_scanned": len(good) + len(bad),
                "admission": "external_test" if cfg.audit_mode == "full" and cfg.allow_network_audio_downloads else "candidate_external",
            })
            print(f"Spell4Wiki metadata: accepted={len(good):,}, rejected={len(bad):,}")
        except Exception as exc:
            source_reports.append({"name": "spell4wiki_commons", "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
            print(f"Spell4Wiki unavailable: {exc}")
        """
    ),
    markdown("## 6. Duplicate, transcript, and split-leakage audit"),
    code(
        r"""
        accepted_df = pd.DataFrame(accepted_rows)
        rejected_df = pd.DataFrame(rejected_rows, columns=list(dict.fromkeys(CANONICAL_COLUMNS + ["reason", "error", "words_per_s"])))
        near_df = pd.DataFrame(near_hash_rows, columns=["sample_id", "source", "near_hash"])
        for column in CANONICAL_COLUMNS:
            if column not in accepted_df:
                accepted_df[column] = pd.Series(dtype="object")
        accepted_df = accepted_df[CANONICAL_COLUMNS + [column for column in accepted_df.columns if column not in CANONICAL_COLUMNS]]


        SPLIT_KEEP_PRIORITY = {"test": 4, "external_test": 3, "validation": 2, "train": 1}


        def quarantine_cross_split_audio_duplicates(
            table: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            '''Preserve held-out splits and quarantine lower-priority exact copies.'''
            supervised = table[table["split"].isin(SPLIT_KEEP_PRIORITY)]
            hashed = supervised[supervised["audio_hash"].fillna("").astype(str).ne("")]
            drop_indexes: list[int] = []
            decisions: list[dict[str, Any]] = []
            for audio_hash, group in hashed.groupby("audio_hash"):
                splits = set(group["split"].astype(str))
                if len(splits) <= 1:
                    continue
                keep_split = max(splits, key=lambda name: SPLIT_KEEP_PRIORITY.get(name, 0))
                kept_ids = sorted(group.loc[group["split"] == keep_split, "sample_id"].astype(str))
                for index, row in group[group["split"] != keep_split].iterrows():
                    drop_indexes.append(index)
                    decisions.append({
                        "audio_hash": audio_hash,
                        "dropped_sample_id": str(row["sample_id"]),
                        "dropped_split": str(row["split"]),
                        "kept_split": keep_split,
                        "kept_sample_ids": ",".join(kept_ids),
                        "reason": "cross_split_exact_audio_duplicate",
                    })
            if not drop_indexes:
                return table.copy(), pd.DataFrame(), pd.DataFrame(decisions)
            quarantined = table.loc[drop_indexes].copy()
            quarantined["reason"] = "cross_split_exact_audio_duplicate"
            quarantined["error"] = ""
            decision_by_id = {row["dropped_sample_id"]: row for row in decisions}
            quarantined["kept_split"] = quarantined["sample_id"].astype(str).map(
                lambda value: decision_by_id[value]["kept_split"]
            )
            quarantined["kept_sample_ids"] = quarantined["sample_id"].astype(str).map(
                lambda value: decision_by_id[value]["kept_sample_ids"]
            )
            repaired = table.drop(index=drop_indexes).reset_index(drop=True)
            return repaired, quarantined.reset_index(drop=True), pd.DataFrame(decisions)


        accepted_df, quarantined_duplicates_df, dedup_decisions_df = quarantine_cross_split_audio_duplicates(accepted_df)
        if not quarantined_duplicates_df.empty:
            rejected_df = pd.concat([rejected_df, quarantined_duplicates_df], ignore_index=True, sort=False)
            dropped_ids = set(quarantined_duplicates_df["sample_id"].astype(str))
            near_df = near_df[~near_df["sample_id"].astype(str).isin(dropped_ids)].reset_index(drop=True)
            print(f"Quarantined {len(quarantined_duplicates_df):,} lower-priority cross-split exact-audio copies.")


        def is_official_waxal_speaker_overlap(group: pd.DataFrame) -> bool:
            '''WAXAL's published ASR splits are topic-level and reuse speakers.'''
            official_splits = {"train", "validation", "test"}
            return bool(
                group["source"].eq("waxal_dag_asr").all()
                and set(group["split"].unique()).issubset(official_splits)
                and ("split_policy" not in group or group["split_policy"].eq("official").all())
            )


        def overlap_report(table: pd.DataFrame) -> pd.DataFrame:
            findings = []
            supervised = table[table["split"].isin(["train", "validation", "test", "external_test"])]
            for key in ("speaker_id", "audio_hash"):
                if key not in supervised or supervised.empty:
                    continue
                clean = supervised[supervised[key].fillna("").astype(str).ne("")]
                if key == "speaker_id":
                    clean = clean[clean[key] != "unknown"]
                for value, group in clean.groupby(key):
                    splits = sorted(group["split"].unique())
                    if len(splits) > 1:
                        official_waxal = key == "speaker_id" and is_official_waxal_speaker_overlap(group)
                        blocking = not official_waxal
                        findings.append({
                            "kind": "waxal_official_topic_split_speaker_overlap" if official_waxal else f"{key}_overlap",
                            "value": value,
                            "sources": ",".join(sorted(group["source"].astype(str).unique())),
                            "splits": ",".join(splits),
                            "rows": len(group),
                            "severity": "warning" if official_waxal else "error",
                            "blocking": blocking,
                            "note": (
                                "Published WAXAL topic-level protocol; retained for benchmark comparability."
                                if official_waxal else "Must be resolved before full training."
                            ),
                        })
            return pd.DataFrame(
                findings,
                columns=["kind", "value", "sources", "splits", "rows", "severity", "blocking", "note"],
            )


        leakage_df = overlap_report(accepted_df)
        blocking_leakage_df = leakage_df[leakage_df["blocking"].fillna(True)] if not leakage_df.empty else leakage_df
        waxal_protocol_overlap_df = (
            leakage_df[leakage_df["kind"] == "waxal_official_topic_split_speaker_overlap"]
            if not leakage_df.empty else leakage_df
        )
        exact_duplicates = accepted_df[accepted_df["audio_hash"].fillna("").astype(str).ne("")].groupby("audio_hash").filter(lambda group: len(group) > 1)
        transcript_duplicates = accepted_df[accepted_df["transcript_eval"].astype(str).ne("")].groupby("transcript_eval").filter(lambda group: len(group) > 1)
        near_duplicates = near_df.groupby("near_hash").filter(lambda group: len(group) > 1) if not near_df.empty else near_df

        inventory_rows = []
        if not accepted_df.empty:
            for (source, split), group in accepted_df.groupby(["source", "split"], dropna=False):
                durations = pd.to_numeric(group["duration_s"], errors="coerce")
                inventory_rows.append({
                    "source": source, "split": split, "rows_scanned": len(group),
                    "decoded_rows": int(durations.notna().sum()), "decoded_hours": float(durations.sum() / 3600),
                    "speakers": int(group["speaker_id"].replace("unknown", np.nan).nunique()),
                    "transcript_coverage": float(group["transcript_raw"].astype(bool).mean()),
                    "license_values": ",".join(sorted(group["license"].dropna().astype(str).unique())),
                })
        inventory_df = pd.DataFrame(inventory_rows)
        display(inventory_df)
        print({
            "accepted": len(accepted_df), "rejected": len(rejected_df),
            "exact_duplicate_rows": len(exact_duplicates), "near_duplicate_rows": len(near_duplicates),
            "repeated_transcript_rows": len(transcript_duplicates),
            "reported_split_overlaps": len(leakage_df),
            "blocking_split_leaks": len(blocking_leakage_df),
            "waxal_protocol_speaker_overlaps": len(waxal_protocol_overlap_df),
        })
        if not blocking_leakage_df.empty:
            print("Blocking cross-split overlaps (inspect before training):")
            display(blocking_leakage_df)
            blocking_audio_hashes = set(
                blocking_leakage_df.loc[
                    blocking_leakage_df["kind"] == "audio_hash_overlap", "value"
                ].astype(str)
            )
            if blocking_audio_hashes:
                display(
                    accepted_df[accepted_df["audio_hash"].astype(str).isin(blocking_audio_hashes)][
                        ["sample_id", "source", "split", "speaker_id", "duration_s", "audio_hash", "transcript_raw"]
                    ].sort_values(["audio_hash", "split", "sample_id"])
                )
        if not rejected_df.empty and "reason" in rejected_df:
            print("Rejected-row reasons:")
            print(rejected_df["reason"].value_counts(dropna=False).to_string())
        """
    ),
    markdown("## 7. Export immutable audit artifacts and readiness state"),
    code(
        r"""
        def write_table(table: pd.DataFrame, stem: str) -> None:
            if len(table.columns) == 0:
                table = pd.DataFrame({"_empty": pd.Series(dtype="boolean")})
            table.to_csv(ARTIFACT_DIR / f"{stem}.csv", index=False)
            table.to_parquet(ARTIFACT_DIR / f"{stem}.parquet", index=False)


        write_table(pd.DataFrame(source_reports), "source_status")
        write_table(inventory_df, "corpus_inventory")
        write_table(accepted_df, "accepted_manifest")
        write_table(rejected_df, "rejected_samples")
        write_table(dedup_decisions_df, "dedup_decisions")
        write_table(leakage_df, "split_leakage_report")
        write_table(exact_duplicates, "exact_audio_duplicates")
        write_table(near_duplicates, "near_audio_duplicates")
        write_table(transcript_duplicates, "repeated_transcripts")

        environment = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "audit_config": asdict(cfg) | {"hf_token": "<redacted>"},
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "accelerate", "huggingface_hub", "jiwer", "librosa", "soundfile")
                if importlib.util.find_spec(name)
            },
        }
        (ARTIFACT_DIR / "environment_manifest.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

        waxal_supervised = accepted_df[
            accepted_df["source"].eq("waxal_dag_asr")
            & accepted_df["split"].isin(["train", "validation", "test"])
        ]
        waxal_required_splits_present = {"train", "validation", "test"}.issubset(
            set(waxal_supervised["split"].unique())
        )

        readiness = {
            "audit_mode": cfg.audit_mode,
            "release_ready": bool(
                cfg.audit_mode == "full"
                and not accepted_df.empty
                and waxal_required_splits_present
                and blocking_leakage_df.empty
                and accepted_df[accepted_df["split"].isin(["train", "validation", "test", "external_test"])]["duration_s"].notna().all()
                and accepted_df[accepted_df["split"].isin(["train", "validation", "test", "external_test"])]["audio_hash"].astype(bool).all()
            ),
            "sealed_test_rows": int((accepted_df["split"] == "test").sum()) if not accepted_df.empty else 0,
            "waxal_required_splits_present": waxal_required_splits_present,
            "enabled_sources": list(cfg.enabled_sources),
            "warnings": [],
        }
        if cfg.audit_mode != "full":
            readiness["warnings"].append("Quick mode is diagnostic only; run full mode before full training.")
        if not waxal_required_splits_present:
            readiness["warnings"].append("WAXAL train/validation/test are required for supervised training.")
        if not waxal_protocol_overlap_df.empty:
            readiness["warnings"].append(
                "WAXAL official topic-level splits reuse speakers. Official metrics remain benchmark-comparable but are not speaker-independent."
            )
        if not blocking_leakage_df.empty:
            readiness["warnings"].append("Cross-split speaker/audio leakage must be resolved.")
        (ARTIFACT_DIR / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        print(f"Artifacts written to {ARTIFACT_DIR}")
        """
    ),
    markdown("## 8. Strict and minimally normalized ASR metrics"),
    code(
        r"""
        import jiwer


        def asr_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
            if len(references) != len(hypotheses):
                raise ValueError(f"Reference/hypothesis length mismatch: {len(references)} != {len(hypotheses)}")
            if not references:
                raise ValueError("At least one reference is required")
            strict_refs = [strict_text(value) for value in references]
            strict_hyps = [strict_text(value) for value in hypotheses]
            norm_refs = [normalize_eval(value) for value in references]
            norm_hyps = [normalize_eval(value) for value in hypotheses]
            return {
                "strict_wer": float(jiwer.wer(strict_refs, strict_hyps)),
                "strict_cer": float(jiwer.cer(strict_refs, strict_hyps)),
                "normalized_wer": float(jiwer.wer(norm_refs, norm_hyps)),
                "normalized_cer": float(jiwer.cer(norm_refs, norm_hyps)),
                "utterances": len(references),
            }


        metric_self_test = asr_metrics(["N nyɛla bɛ."], ["n nyɛla bɛ"])
        assert metric_self_test["normalized_wer"] == 0.0
        print("Metric self-test passed:", metric_self_test)
        """
    ),
    markdown(
        """
        ## 9. Optional comparable baselines

        Turn on a GPU, set `cfg.run_baselines = True`, and rerun this cell. Both models
        receive the same frozen, filtered WAXAL test rows. The base model is evaluated
        for context; `waxal-benchmarking/whisper-small-waxal-dag` is the 34.0% WER /
        11.9% CER public reference. No language token is forced because Whisper has no
        native Dagbani language token.
        """
    ),
    code(
        r"""
        def iter_waxal_benchmark(limit: int | None) -> Iterator[dict[str, Any]]:
            stream = load_dataset("google/WaxalNLP", "dag_asr", split="test", streaming=True, token=cfg.hf_token)
            yielded = 0
            for record in stream:
                array, sample_rate, _ = audio_payload(record["audio"])
                array = canonical_audio(array, sample_rate)
                reference = strict_text(record.get("transcription", ""))
                duration = len(array) / 16_000
                words_per_s = len(normalize_eval(reference).split()) / max(duration, 1e-6)
                if duration < cfg.waxal_benchmark_min_duration_s or words_per_s > cfg.waxal_benchmark_max_words_per_s:
                    continue
                yield {"id": str(record.get("id", yielded)), "audio": array, "reference": reference, "duration_s": duration}
                yielded += 1
                if limit is not None and yielded >= limit:
                    break


        def benchmark_model(model_id: str, rows: list[dict[str, Any]]) -> tuple[dict, pd.DataFrame]:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            if not torch.cuda.is_available():
                raise RuntimeError("Enable a Kaggle GPU before running baselines")
            major, minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            print({"gpu": torch.cuda.get_device_name(0), "device_arch": device_arch, "torch_arches": sorted(supported_arches)})
            if supported_arches and device_arch not in supported_arches and f"compute_{major}{minor}" not in supported_arches:
                raise RuntimeError(
                    f"Kaggle's current PyTorch build does not support {device_arch}. "
                    "Select a T4 GPU instead of P100 and restart the session."
                )
            processor = AutoProcessor.from_pretrained(model_id, token=cfg.hf_token)
            load_kwargs = {"token": cfg.hf_token, "low_cpu_mem_usage": True}
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.float16, **load_kwargs)
            except TypeError:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16, **load_kwargs)
            model.to("cuda").eval()
            model.generation_config.forced_decoder_ids = None
            predictions = []
            started = time.time()
            batch_size = 8
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                inputs = processor.feature_extractor(
                    [row["audio"] for row in batch], sampling_rate=16_000,
                    return_tensors="pt", padding="max_length", truncation=True,
                    max_length=30 * 16_000, return_attention_mask=True,
                )
                generate_kwargs = {
                    "input_features": inputs.input_features.to("cuda", dtype=torch.float16),
                    "task": "transcribe",
                }
                if "attention_mask" in inputs:
                    generate_kwargs["attention_mask"] = inputs.attention_mask.to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**generate_kwargs)
                predictions.extend(strict_text(text) for text in processor.batch_decode(generated, skip_special_tokens=True))
                completed = min(offset + len(batch), len(rows))
                if completed % 32 < batch_size or completed == len(rows):
                    print(f"{model_id}: {completed}/{len(rows)} rows, {time.time() - started:.1f}s")
            references = [row["reference"] for row in rows]
            metrics = asr_metrics(references, predictions) | {"model_id": model_id, "elapsed_s": time.time() - started}
            table = pd.DataFrame({"sample_id": [row["id"] for row in rows], "reference": references, "hypothesis": predictions})
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            return metrics, table


        baseline_metrics = []
        if cfg.run_baselines:
            eval_rows = list(iter_waxal_benchmark(cfg.baseline_limit))
            if not eval_rows:
                raise RuntimeError("No rows passed the WAXAL benchmark filter")
            for model_name in ("openai/whisper-small", "waxal-benchmarking/whisper-small-waxal-dag"):
                metrics, predictions = benchmark_model(model_name, eval_rows)
                baseline_metrics.append(metrics)
                predictions.to_parquet(ARTIFACT_DIR / f"baseline_{model_name.replace('/', '__')}.parquet", index=False)
                print(json.dumps(metrics, indent=2))
        else:
            print("Baselines skipped. This preserves GPU quota during data auditing.")
        (ARTIFACT_DIR / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
        """
    ),
    markdown("## 10. Publish the manifest and audit evidence to a private Hugging Face dataset"),
    code(
        r"""
        if cfg.publish_artifacts:
            from huggingface_hub import HfApi
            api = HfApi(token=cfg.hf_token)
            api.create_repo(cfg.manifest_repo_id, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(
                repo_id=cfg.manifest_repo_id,
                repo_type="dataset",
                folder_path=str(ARTIFACT_DIR),
                path_in_repo="phase1",
                commit_message=f"Dagbani ASR Phase 1 {cfg.audit_mode} audit",
            )
            print(f"Published privately to https://huggingface.co/datasets/{cfg.manifest_repo_id}")
        else:
            print("Publishing disabled. Inspect all artifacts before setting publish_artifacts=True.")
        """
    ),
]


BASELINE_CELLS = [
    markdown(
        r"""
        # Dagbani ASR Phase 1B — GPU Baseline Only

        This auxiliary Kaggle notebook evaluates `openai/whisper-small` and the
        public WAXAL Dagbani Whisper-small checkpoint without repeating the CPU data
        audit. It reads the release-ready private Phase 1 manifest, freezes the
        WAXAL cleaning filter (duration >= 1.5 seconds and speech rate <= 4
        words/second), batches GPU inference, and publishes predictions and metrics
        back to `phase1/baselines/` in the private dataset repository.

        Start with the 256-row pilot. The public checkpoint's 34.0% WER / 11.9% CER
        target applies to the full filtered evaluation set, not necessarily the pilot.
        The public model card currently prints `speech rate >= 4 WPS`; applied to the
        sealed manifest that retains only two rows. This notebook therefore records
        the empirically corrected `<= 4 WPS` cleaning rule as a separate v2 protocol.
        Do not call it an exact public-score reproduction unless the official cleaned
        evaluation manifest or filtering script confirms the same rows.
        """
    ),
    markdown("## 1. Install only missing runtime packages"),
    code(
        r"""
        import importlib.util
        import subprocess
        import sys

        REQUIRED = {
            "datasets": "datasets[audio]>=3.2,<5",
            "transformers": "transformers>=4.48,<6",
            "huggingface_hub": "huggingface_hub>=0.27,<2",
            "jiwer": "jiwer>=3.0,<5",
            "soundfile": "soundfile>=0.12,<1",
            "librosa": "librosa>=0.10,<1",
            "pyarrow": "pyarrow>=17,<25",
        }
        missing = [spec for module, spec in REQUIRED.items() if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("Baseline environment ready. PyTorch is intentionally not replaced.")
        """,
        "setup",
    ),
    markdown("## 2. Configuration — the pilot is opt-in"),
    code(
        r"""
        from __future__ import annotations

        import gc
        import hashlib
        import io
        import json
        import os
        import string
        import time
        import unicodedata
        from dataclasses import asdict, dataclass, field
        from pathlib import Path
        from typing import Any, Iterator

        import numpy as np
        import pandas as pd
        import soundfile as sf


        def kaggle_secret(name: str) -> str | None:
            value = os.getenv(name)
            if value:
                return value
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return None


        @dataclass
        class BaselineConfig:
            manifest_repo_id: str = "ats-tech/dagbani-asr-phase1"
            hf_token: str | None = field(default_factory=lambda: kaggle_secret("HF_TOKEN"))
            model_ids: tuple[str, ...] = (
                "waxal-benchmarking/whisper-small-waxal-dag",
                "openai/whisper-small",
            )
            run_baselines: bool = False
            max_samples: int | None = 256  # Set None only after the pilot succeeds.
            batch_size: int = 8
            upload_every: int = 100
            session_budget_minutes: float = 150.0
            work_dir: str = "/kaggle/working/dagbani_asr_baselines"
            min_duration_s: float = 1.5
            max_words_per_s: float = 4.0
            protocol_id: str = "waxal-clean-v2-max4wps"

            def validate(self) -> None:
                if not self.hf_token:
                    raise ValueError("Attach the Kaggle HF_TOKEN secret")
                if not self.manifest_repo_id:
                    raise ValueError("manifest_repo_id is required")
                if self.batch_size < 1:
                    raise ValueError("batch_size must be positive")
                if self.max_words_per_s <= 0:
                    raise ValueError("max_words_per_s must be positive")


        cfg = BaselineConfig()
        cfg.validate()
        WORK_DIR = Path(cfg.work_dir)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        print(json.dumps(asdict(cfg) | {"hf_token": "<set>"}, indent=2))
        """
    ),
    markdown("## 3. Load and freeze the release-ready evaluation manifest"),
    code(
        r"""
        from huggingface_hub import HfApi, hf_hub_download


        def artifact(filename: str) -> Path:
            return Path(hf_hub_download(
                repo_id=cfg.manifest_repo_id,
                filename=f"phase1/{filename}",
                repo_type="dataset",
                token=cfg.hf_token,
                cache_dir=str(WORK_DIR / "hf_cache"),
            ))


        readiness = json.loads(artifact("readiness.json").read_text(encoding="utf-8"))
        if not readiness.get("release_ready"):
            raise RuntimeError(f"Phase 1 is not release-ready: {readiness}")

        manifest = pd.read_parquet(artifact("accepted_manifest.parquet"))
        waxal_test = manifest[
            manifest["source"].eq("waxal_dag_asr") & manifest["split"].eq("test")
        ].copy()
        if "words_per_s" not in waxal_test:
            waxal_test["words_per_s"] = (
                waxal_test["transcript_eval"].fillna("").astype(str).str.split().str.len()
                / pd.to_numeric(waxal_test["duration_s"], errors="coerce")
            )
        frozen_eval = waxal_test[
            pd.to_numeric(waxal_test["duration_s"], errors="coerce").ge(cfg.min_duration_s)
            & pd.to_numeric(waxal_test["words_per_s"], errors="coerce").le(cfg.max_words_per_s)
        ].sort_values("sample_id").reset_index(drop=True)
        if cfg.max_samples is not None:
            frozen_eval = frozen_eval.head(cfg.max_samples).copy()
        if frozen_eval.empty:
            raise RuntimeError("No WAXAL test rows passed the WAXAL cleaning filter")

        repo_info = HfApi(token=cfg.hf_token).repo_info(cfg.manifest_repo_id, repo_type="dataset")
        phase1_repo_head = str(repo_info.sha)
        manifest_revision = hashlib.sha256(
            pd.util.hash_pandas_object(manifest, index=True).values.tobytes()
        ).hexdigest()
        eval_manifest_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                frozen_eval[["sample_id", "transcript_raw", "audio_hash"]], index=False
            ).values.tobytes()
        ).hexdigest()
        frozen_eval.to_parquet(WORK_DIR / "frozen_eval_manifest.parquet", index=False)
        print({
            "phase1_repo_head": phase1_repo_head,
            "phase1_manifest_hash": manifest_revision,
            "sealed_test_rows": len(waxal_test),
            "filtered_eval_rows": len(frozen_eval),
            "speech_rate_rule": f"words_per_s <= {cfg.max_words_per_s}",
            "eval_manifest_hash": eval_manifest_hash,
        })
        """
    ),
    markdown("## 4. Conservative metrics and audio decoding"),
    code(
        r"""
        import jiwer

        PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation + "“”‘’…–—"})


        def strict_text(value: Any) -> str:
            return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


        def normalize_eval(value: Any) -> str:
            return " ".join(strict_text(value).lower().translate(PUNCT_TRANSLATION).split())


        def asr_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
            if len(references) != len(hypotheses) or not references:
                raise ValueError("References and hypotheses must be non-empty and aligned")
            strict_refs = [strict_text(value) for value in references]
            strict_hyps = [strict_text(value) for value in hypotheses]
            norm_refs = [normalize_eval(value) for value in references]
            norm_hyps = [normalize_eval(value) for value in hypotheses]
            return {
                "strict_wer": float(jiwer.wer(strict_refs, strict_hyps)),
                "strict_cer": float(jiwer.cer(strict_refs, strict_hyps)),
                "normalized_wer": float(jiwer.wer(norm_refs, norm_hyps)),
                "normalized_cer": float(jiwer.cer(norm_refs, norm_hyps)),
                "utterances": len(references),
            }


        def audio_payload(value: Any) -> tuple[np.ndarray, int]:
            if hasattr(value, "get_all_samples"):
                samples = value.get_all_samples()
                array = samples.data.detach().cpu().numpy()
                sample_rate = int(samples.sample_rate)
            elif isinstance(value, dict) and value.get("array") is not None:
                array = np.asarray(value["array"], dtype=np.float32)
                sample_rate = int(value.get("sampling_rate") or 16_000)
            elif isinstance(value, dict) and value.get("bytes"):
                array, sample_rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
            else:
                raise ValueError("Unsupported WAXAL audio payload")
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:
                array = array.mean(axis=0 if array.shape[0] <= 8 else 1)
            return array.reshape(-1), sample_rate


        def canonical_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
            if sample_rate != 16_000:
                import librosa
                array = librosa.resample(array, orig_sr=sample_rate, target_sr=16_000)
            return np.clip(array, -1.0, 1.0).astype(np.float32)


        assert asr_metrics(["N nyɛla bɛ."], ["n nyɛla bɛ"])["normalized_wer"] == 0.0
        print("Metric and glyph self-test passed.")
        """
    ),
    markdown("## 5. Batched, resumable GPU inference"),
    code(
        r"""
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        api = HfApi(token=cfg.hf_token)
        eval_by_id = frozen_eval.set_index("sample_id")


        def slug(model_id: str) -> str:
            return model_id.replace("/", "__")


        def remote_predictions_path(model_id: str) -> str:
            mode = "full" if cfg.max_samples is None else f"pilot-{cfg.max_samples}"
            return f"phase1/baselines/{cfg.protocol_id}/{mode}/{slug(model_id)}-predictions.parquet"


        def load_existing_predictions(model_id: str) -> pd.DataFrame:
            try:
                path = hf_hub_download(
                    repo_id=cfg.manifest_repo_id,
                    filename=remote_predictions_path(model_id),
                    repo_type="dataset",
                    token=cfg.hf_token,
                    cache_dir=str(WORK_DIR / "resume_cache"),
                    force_download=True,
                )
            except Exception:
                return pd.DataFrame(columns=[
                    "sample_id", "reference", "hypothesis", "model_id",
                    "manifest_revision", "eval_manifest_hash",
                ])
            table = pd.read_parquet(path)
            if not table.empty and set(table["eval_manifest_hash"].astype(str)) != {eval_manifest_hash}:
                raise RuntimeError("Remote predictions belong to a different frozen evaluation manifest")
            return table


        def upload_predictions(model_id: str, table: pd.DataFrame) -> Path:
            path = WORK_DIR / f"{slug(model_id)}-predictions.parquet"
            table.to_parquet(path, index=False)
            api.upload_file(
                repo_id=cfg.manifest_repo_id,
                repo_type="dataset",
                path_or_fileobj=str(path),
                path_in_repo=remote_predictions_path(model_id),
                commit_message=f"Checkpoint Dagbani ASR baseline: {model_id} ({len(table)}/{len(frozen_eval)})",
            )
            return path


        def iter_pending_audio(pending_ids: set[str]) -> Iterator[dict[str, Any]]:
            found: set[str] = set()
            stream = load_dataset(
                "google/WaxalNLP", "dag_asr", split="test", streaming=True, token=cfg.hf_token
            )
            for record in stream:
                sample_id = str(record.get("id", ""))
                if sample_id not in pending_ids:
                    continue
                array, sample_rate = audio_payload(record["audio"])
                found.add(sample_id)
                yield {
                    "sample_id": sample_id,
                    "reference": strict_text(eval_by_id.at[sample_id, "transcript_raw"]),
                    "audio": canonical_audio(array, sample_rate),
                }
            missing = pending_ids - found
            if missing:
                raise RuntimeError(f"WAXAL stream did not yield {len(missing)} frozen IDs: {sorted(missing)[:10]}")


        def require_compatible_gpu() -> None:
            if not torch.cuda.is_available():
                raise RuntimeError("Enable a Kaggle GPU before running baselines")
            major, minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            gpu_info = {
                "gpu": torch.cuda.get_device_name(0),
                "device_arch": device_arch,
                "torch_supported_arches": sorted(supported_arches),
                "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            }
            print("GPU preflight:", gpu_info)
            if supported_arches and device_arch not in supported_arches and f"compute_{major}{minor}" not in supported_arches:
                raise RuntimeError(
                    f"Kaggle's current PyTorch build does not support {device_arch}. "
                    "Select T4 x2 instead of P100, restart the session, and rerun from the top."
                )


        def build_model(model_id: str):
            require_compatible_gpu()
            processor = AutoProcessor.from_pretrained(model_id, token=cfg.hf_token)
            load_kwargs = {"token": cfg.hf_token, "low_cpu_mem_usage": True}
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, dtype=torch.float16, **load_kwargs
                )
            except TypeError:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch.float16, **load_kwargs
                )
            model.to("cuda").eval()
            model.generation_config.forced_decoder_ids = None
            return processor, model


        def transcribe_batch(processor, model, batch: list[dict[str, Any]]) -> list[str]:
            inputs = processor.feature_extractor(
                [item["audio"] for item in batch],
                sampling_rate=16_000,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=30 * 16_000,
                return_attention_mask=True,
            )
            generate_kwargs = {
                "input_features": inputs.input_features.to("cuda", dtype=torch.float16),
                "task": "transcribe",
            }
            if "attention_mask" in inputs:
                generate_kwargs["attention_mask"] = inputs.attention_mask.to("cuda")
            with torch.inference_mode():
                generated_ids = model.generate(**generate_kwargs)
            return [strict_text(text) for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]


        def evaluate_model(model_id: str) -> dict[str, Any]:
            predictions = load_existing_predictions(model_id)
            predictions = predictions.drop_duplicates("sample_id", keep="last")
            expected_ids = set(frozen_eval["sample_id"].astype(str))
            predictions = predictions[predictions["sample_id"].astype(str).isin(expected_ids)]
            pending_ids = expected_ids - set(predictions["sample_id"].astype(str))
            started = time.time()
            processed_this_run = 0
            stopped_for_budget = False

            if pending_ids:
                processor, model = build_model(model_id)
                batch: list[dict[str, Any]] = []
                for row in iter_pending_audio(pending_ids):
                    batch.append(row)
                    if len(batch) < cfg.batch_size:
                        continue
                    hypotheses = transcribe_batch(processor, model, batch)
                    new_rows = [{
                        "sample_id": item["sample_id"],
                        "reference": item["reference"],
                        "hypothesis": hypothesis,
                        "model_id": model_id,
                        "manifest_revision": manifest_revision,
                        "eval_manifest_hash": eval_manifest_hash,
                    } for item, hypothesis in zip(batch, hypotheses)]
                    predictions = pd.concat([predictions, pd.DataFrame(new_rows)], ignore_index=True)
                    processed_this_run += len(new_rows)
                    batch = []
                    completed = predictions["sample_id"].nunique()
                    elapsed = max(time.time() - started, 1e-6)
                    rate = processed_this_run / elapsed
                    eta_min = (len(expected_ids) - completed) / max(rate, 1e-9) / 60
                    if completed % 25 < cfg.batch_size:
                        print(f"{model_id}: {completed}/{len(expected_ids)}; {rate:.2f} rows/s; ETA {eta_min:.1f}m")
                    if completed % cfg.upload_every < cfg.batch_size:
                        upload_predictions(model_id, predictions)
                    if elapsed / 60 >= cfg.session_budget_minutes:
                        stopped_for_budget = True
                        break

                if batch and not stopped_for_budget:
                    hypotheses = transcribe_batch(processor, model, batch)
                    predictions = pd.concat([predictions, pd.DataFrame([{
                        "sample_id": item["sample_id"], "reference": item["reference"],
                        "hypothesis": hypothesis, "model_id": model_id,
                        "manifest_revision": manifest_revision, "eval_manifest_hash": eval_manifest_hash,
                    } for item, hypothesis in zip(batch, hypotheses)])], ignore_index=True)
                upload_predictions(model_id, predictions)
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()

            completed_ids = set(predictions["sample_id"].astype(str))
            if completed_ids != expected_ids:
                return {
                    "model_id": model_id, "status": "partial",
                    "completed": len(completed_ids), "total": len(expected_ids),
                    "resume_safe": True,
                }

            ordered = frozen_eval[["sample_id", "transcript_raw"]].merge(
                predictions[["sample_id", "hypothesis"]], on="sample_id", how="left", validate="one_to_one"
            )
            metrics = asr_metrics(
                ordered["transcript_raw"].astype(str).tolist(),
                ordered["hypothesis"].astype(str).tolist(),
            ) | {
                "model_id": model_id,
                "status": "complete",
                "phase1_revision": manifest_revision,
                "eval_manifest_hash": eval_manifest_hash,
                "filter_min_duration_s": cfg.min_duration_s,
                "filter_max_words_per_s": cfg.max_words_per_s,
                "protocol_id": cfg.protocol_id,
                "pilot_limit": cfg.max_samples,
            }
            return metrics


        print("GPU runner loaded. It preserves Whisper suppression defaults and forces no foreign language token.")
        """
    ),
    markdown("## 6. Run and publish metrics"),
    code(
        r"""
        baseline_metrics = []
        if cfg.run_baselines:
            frozen_remote = WORK_DIR / "frozen_eval_manifest.parquet"
            api.upload_file(
                repo_id=cfg.manifest_repo_id,
                repo_type="dataset",
                path_or_fileobj=str(frozen_remote),
                path_in_repo=(
                    f"phase1/baselines/{cfg.protocol_id}/full/frozen_eval_manifest.parquet"
                    if cfg.max_samples is None
                    else f"phase1/baselines/{cfg.protocol_id}/pilot-{cfg.max_samples}/frozen_eval_manifest.parquet"
                ),
                commit_message="Freeze Dagbani WAXAL baseline evaluation manifest",
            )
            for model_id in cfg.model_ids:
                result = evaluate_model(model_id)
                baseline_metrics.append(result)
                print(json.dumps(result, indent=2))
                if result.get("status") != "complete":
                    print("Session budget reached. Rerun this notebook to resume safely.")
                    break

            mode = "full" if cfg.max_samples is None else f"pilot-{cfg.max_samples}"
            metrics_path = WORK_DIR / "baseline_metrics.json"
            metrics_path.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
            api.upload_file(
                repo_id=cfg.manifest_repo_id,
                repo_type="dataset",
                path_or_fileobj=str(metrics_path),
                path_in_repo=f"phase1/baselines/{cfg.protocol_id}/{mode}/baseline_metrics.json",
                commit_message=f"Publish Dagbani ASR {mode} baseline metrics",
            )
            print(
                "Published private baseline evidence to "
                f"{cfg.manifest_repo_id}/phase1/baselines/{cfg.protocol_id}/{mode}"
            )
        else:
            print("Baseline execution is OFF. Set cfg.run_baselines=True after enabling one Kaggle GPU.")
        """
    ),
]


PHASE2_CELLS = [
    markdown(
        r"""
        # Dagbani ASR Phase 2 — Budget-Aware Whisper Training

        This notebook trains an owned Dagbani ASR model from `openai/whisper-small`.
        It consumes the **full, release-ready Phase 1 manifest** and runs one stage at
        a time: `smoke`, `supervised`, `domain_adaptation`, or `pseudo_label`.

        The default is a 20-step smoke run. Full stages refuse to start when the Phase
        1 readiness file is missing, when the test split leaks into training, or when
        the audited manifest contains undecoded supervised audio.

        The notebook's validated Kaggle path uses one GPU. It also contains a guarded
        notebook-launched DDP experiment, but refuses to fork if the current kernel has
        already initialized CUDA. It never uses `device_map="auto"` as a substitute
        for data parallelism.
        """
    ),
    markdown("## 1. Install dependencies without replacing Kaggle's PyTorch/CUDA build"),
    code(
        r"""
        import importlib.util
        import subprocess
        import sys

        INSTALL_DEPS = True
        REQUIRED = {
            "datasets": "datasets[audio]>=3.2,<5",
            "transformers": "transformers>=4.48,<6",
            "accelerate": "accelerate>=1.2,<3",
            "huggingface_hub": "huggingface_hub>=0.27,<2",
            "peft": "peft>=0.14,<1",
            "bitsandbytes": "bitsandbytes>=0.45,<1",
            "jiwer": "jiwer>=3.0,<5",
            "soundfile": "soundfile>=0.12,<1",
            "librosa": "librosa>=0.10,<1",
            "pyarrow": "pyarrow>=17,<25",
        }
        missing = [spec for module, spec in REQUIRED.items() if importlib.util.find_spec(module) is None]
        if INSTALL_DEPS and missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("Dependencies ready. Restart the kernel only if pip explicitly requests it.")
        """,
        "setup",
    ),
    markdown("## 2. Stage configuration, secrets, and hard safety gates"),
    code(
        r"""
        from __future__ import annotations

        import dataclasses
        import gzip
        import hashlib
        import inspect
        import io
        import json
        import math
        import os
        import random
        import re
        import shutil
        import string
        import subprocess
        import textwrap
        import time
        import unicodedata
        from dataclasses import asdict, dataclass, field
        from pathlib import Path
        from typing import Any, Iterable, Iterator

        import numpy as np
        import pandas as pd


        def kaggle_secret(name: str) -> str | None:
            value = os.getenv(name)
            if value:
                return value
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return None


        @dataclass
        class TrainConfig:
            stage: str = "smoke"  # smoke | supervised | domain_adaptation | pseudo_label | medium_pilot | medium_full
            base_model_id: str = "openai/whisper-small"
            manifest_repo_id: str = os.getenv("DAGBANI_MANIFEST_REPO", "")
            model_repo_id: str = os.getenv("DAGBANI_MODEL_REPO", "")
            hf_token: str | None = field(default_factory=lambda: kaggle_secret("HF_TOKEN"))
            work_dir: str = "/kaggle/working/dagbani_asr_phase2"
            manifest_local_dir: str = os.getenv("DAGBANI_MANIFEST_DIR", "")
            seed: int = 42
            micro_batch_size: int = 4
            effective_batch_size: int = 32
            learning_rate: float = 1e-4
            warmup_steps: int = 250
            supervised_max_steps: int = 2_500
            domain_max_steps: int = 750
            pseudo_max_steps: int = 750
            smoke_steps: int = 20
            medium_pilot_steps: int = 100
            medium_full_max_steps: int = 2_500
            medium_gate_file: str = "/kaggle/working/dagbani_asr_phase2/evaluations/medium_gate.json"
            save_steps: int = 250
            eval_steps: int = 250
            major_checkpoint_steps: tuple[int, ...] = (1_000, 2_000, 2_500)
            eval_generation_limit: int | None = None
            max_session_minutes: int = 540
            stop_buffer_minutes: int = 20
            disk_stop_free_gib: float = 8.0
            stream_shuffle_buffer: int = 512
            dataloader_workers: int = 2
            use_ddp: str = "one"  # one is the smoke/default path; two requires the DDP gate
            training_mode: str = "full"  # full | lora
            waxal_batch_share: float = 0.60
            pseudo_batch_share: float = 0.25
            pseudo_initial_hours: float = 20.0
            pseudo_calibration_rows: int = 128
            pseudo_max_new_tokens: int = 225
            resume: bool = True
            upload_full_state: bool = True
            run_training: bool = False
            run_full_generation_eval: bool = False
            run_external_generation_eval: bool = False
            external_eval_limit: int | None = None
            run_pseudo_labelling: bool = False
            allow_remote_audio_downloads: bool = False

            def validate(self) -> None:
                allowed = {"smoke", "supervised", "domain_adaptation", "pseudo_label", "medium_pilot", "medium_full"}
                if self.stage not in allowed:
                    raise ValueError(f"stage must be one of {sorted(allowed)}")
                if self.training_mode not in {"full", "lora"}:
                    raise ValueError("training_mode must be full or lora")
                if self.use_ddp not in {"auto", "one", "two"}:
                    raise ValueError("use_ddp must be auto, one, or two")
                if self.stage in {"supervised", "domain_adaptation", "pseudo_label", "medium_pilot", "medium_full"} and not self.manifest_repo_id and not self.manifest_local_dir:
                    raise ValueError("Full stages require DAGBANI_MANIFEST_REPO or DAGBANI_MANIFEST_DIR")
                if self.run_training and self.run_pseudo_labelling:
                    raise ValueError("Generate pseudo labels and train in separate runs; enable only one action")
                if (self.run_training or self.run_pseudo_labelling) and not self.hf_token:
                    raise ValueError("HF_TOKEN is required for datasets and private checkpoint persistence")
                if self.run_training and not self.model_repo_id:
                    raise ValueError("Set DAGBANI_MODEL_REPO before training")
                if not (0.60 <= self.waxal_batch_share <= 1.0):
                    raise ValueError("waxal_batch_share must preserve at least 60% WAXAL batches")
                if self.stage == "medium_full":
                    gate_path = Path(self.medium_gate_file)
                    if not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("allow_medium_full_run"):
                        raise ValueError("medium_full is locked until a passing medium_gate.json is created")

            def max_steps(self) -> int:
                return {
                    "smoke": self.smoke_steps,
                    "supervised": self.supervised_max_steps,
                    "domain_adaptation": self.domain_max_steps,
                    "pseudo_label": self.pseudo_max_steps,
                    "medium_pilot": self.medium_pilot_steps,
                    "medium_full": self.medium_full_max_steps,
                }[self.stage]


        cfg = TrainConfig()
        cfg.validate()
        WORK_DIR = Path(cfg.work_dir)
        OUTPUT_DIR = WORK_DIR / f"{cfg.stage}_output"
        CACHE_DIR = WORK_DIR / "hf_cache"
        EVAL_DIR = WORK_DIR / "evaluations"
        for directory in (WORK_DIR, OUTPUT_DIR, CACHE_DIR, EVAL_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        startup_free_gib = shutil.disk_usage(WORK_DIR).free / 2**30
        if cfg.run_training and startup_free_gib < cfg.disk_stop_free_gib:
            raise RuntimeError(
                f"Only {startup_free_gib:.1f} GiB disk is free; start a fresh Kaggle session "
                f"with at least {cfg.disk_stop_free_gib:.1f} GiB free before training."
            )
        os.environ["HF_HOME"] = str(CACHE_DIR)
        os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "models")
        if cfg.hf_token:
            os.environ["HF_TOKEN"] = cfg.hf_token
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Ask Accelerate to reject a poisoned notebook fork explicitly instead of
        # allowing child processes to hang silently during model construction.
        os.environ["ACCELERATE_DEBUG_MODE"] = "yes"
        if cfg.use_ddp == "one":
            # Trainer otherwise sees both T4s and silently uses DataParallel,
            # invalidating the effective-batch calculation for the one-GPU path.
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(json.dumps(asdict(cfg) | {"hf_token": "<set>" if cfg.hf_token else "<missing>"}, indent=2))
        print(
            "Training action: ENABLED after preflight checks."
            if cfg.run_training
            else "Training action: OFF; this execution is a dry preflight."
        )
        """
    ),
    markdown("## 3. Load and validate the Phase 1 evidence contract"),
    code(
        r"""
        CANONICAL_COLUMNS = [
            "sample_id", "source", "source_version", "audio_locator",
            "transcript_raw", "transcript_eval", "speaker_id", "duration_s",
            "domain", "split", "license", "audio_hash", "is_pseudo",
            "pseudo_confidence",
        ]
        PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation + "“”‘’…–—"})


        def normalize_eval(text: Any) -> str:
            value = unicodedata.normalize("NFC", str(text or "")).lower().translate(PUNCT_TRANSLATION)
            return " ".join(value.split())


        def artifact_path(filename: str) -> Path:
            if cfg.manifest_local_dir:
                candidate = Path(cfg.manifest_local_dir) / filename
                if candidate.exists():
                    return candidate
                nested = Path(cfg.manifest_local_dir) / "phase1" / filename
                if nested.exists():
                    return nested
            if not cfg.manifest_repo_id:
                raise FileNotFoundError(filename)
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(
                cfg.manifest_repo_id, filename=f"phase1/{filename}", repo_type="dataset",
                token=cfg.hf_token, cache_dir=str(CACHE_DIR),
            ))


        manifest_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        readiness = {"audit_mode": "missing", "release_ready": False, "warnings": ["No Phase 1 manifest loaded"]}
        if cfg.manifest_repo_id or cfg.manifest_local_dir:
            manifest_df = pd.read_parquet(artifact_path("accepted_manifest.parquet"))
            readiness = json.loads(artifact_path("readiness.json").read_text(encoding="utf-8"))
            missing_columns = set(CANONICAL_COLUMNS) - set(manifest_df.columns)
            if missing_columns:
                raise ValueError(f"Manifest contract is missing: {sorted(missing_columns)}")


        def assert_manifest_safe(table: pd.DataFrame, require_release_ready: bool) -> None:
            if require_release_ready and not readiness.get("release_ready"):
                raise RuntimeError(f"Phase 1 is not release-ready: {readiness}")
            supervised = table[table["split"].isin(["train", "validation", "test", "external_test"])]
            if require_release_ready and supervised.empty:
                raise RuntimeError("No supervised rows were admitted")
            if not supervised.empty:
                if supervised["transcript_raw"].fillna("").eq("").any():
                    raise RuntimeError("A supervised manifest row has no transcript")
                if require_release_ready and pd.to_numeric(supervised["duration_s"], errors="coerce").isna().any():
                    raise RuntimeError("A supervised manifest row was not audio-audited")
                audio_rows = supervised[supervised["audio_hash"].fillna("").astype(str).ne("")]
                if (audio_rows.groupby("audio_hash")["split"].nunique() > 1).any():
                    raise RuntimeError("Cross-split audio_hash leakage remains")

                speaker_rows = supervised[
                    supervised["speaker_id"].fillna("").astype(str).ne("")
                    & supervised["speaker_id"].ne("unknown")
                ]
                for speaker_id, group in speaker_rows.groupby("speaker_id"):
                    if group["split"].nunique() <= 1:
                        continue
                    official_waxal_overlap = bool(
                        group["source"].eq("waxal_dag_asr").all()
                        and set(group["split"].unique()).issubset({"train", "validation", "test"})
                        and ("split_policy" not in group or group["split_policy"].eq("official").all())
                    )
                    if not official_waxal_overlap:
                        raise RuntimeError(f"Cross-split speaker_id leakage remains for {speaker_id}")
            held_out = table[table["split"].isin(["test", "external_test"])]
            if not held_out.empty and held_out["is_pseudo"].fillna(False).any():
                raise RuntimeError("Held-out rows may never be pseudo-labelled")


        assert_manifest_safe(manifest_df, require_release_ready=cfg.stage != "smoke")
        print(f"Manifest rows: {len(manifest_df):,}; readiness={readiness}")
        """
    ),
    markdown("## 4. Dataset loaders preserve official splits and audited IDs"),
    code(
        r"""
        import requests
        from datasets import Audio, Dataset, DatasetDict, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset

        HF_SOURCE_REGISTRY = {
            "waxal_dag_asr": ("google/WaxalNLP", "dag_asr"),
            "dagbani_bible": ("ghananlpcommunity/dagbani-bible-audio-text-tts", None),
            "ghana_speech_dag": ("ghananlpcommunity/ghana-speech", "Dagbani_dag"),
            "navigation_dagbani": ("ghananlpcommunity/navigation-corpus-dagbani-speech", None),
            "health_unicef_dagbani": ("ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani", None),
            "youth_conversations_dagbani": ("ghananlpcommunity/youth-conversations-dag", None),
        }


        def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
            available = set(columns)
            return next((name for name in candidates if name in available), None)


        def standardize_dataset(
            dataset: Dataset | IterableDataset,
            source: str,
            text_override: dict[str, str] | None = None,
            accepted_ids: set[str] | None = None,
        ) -> Dataset | IterableDataset:
            audio_col = pick_column(dataset.column_names, ("audio", "speech"))
            text_col = pick_column(dataset.column_names, ("transcription", "sentence", "text", "transcript"))
            id_col = pick_column(dataset.column_names, ("id", "sample_id", "path", "file", "filename"))
            if not audio_col or not id_col:
                raise ValueError(f"{source} does not expose auditable audio and id columns: {dataset.column_names}")
            if text_override is None and not text_col:
                raise ValueError(f"{source} has no transcription column")
            if audio_col != "audio":
                dataset = dataset.rename_column(audio_col, "audio")
            if id_col != "sample_id":
                dataset = dataset.rename_column(id_col, "sample_id")
            if accepted_ids is not None:
                if not accepted_ids:
                    return dataset.take(0) if isinstance(dataset, IterableDataset) else dataset.select([])
                dataset = dataset.filter(
                    lambda sample_id: str(sample_id) in accepted_ids,
                    input_columns=["sample_id"],
                )
            if text_override is not None:
                dataset = dataset.map(
                    lambda sample_id: {"sentence": text_override[str(sample_id)]},
                    input_columns=["sample_id"],
                )
            elif text_col != "sentence":
                dataset = dataset.rename_column(text_col, "sentence")
            dataset = dataset.map(
                lambda sample_id: {"sample_id": str(sample_id), "source": source},
                input_columns=["sample_id"],
            )
            # Some datasets 4.x streaming builders expose column names but leave
            # info.features unset. cast_column then crashes while trying to edit
            # a None schema. Streaming audio is decoded explicitly in the
            # collator; map-style datasets retain the convenient Audio feature.
            if not isinstance(dataset, IterableDataset):
                dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
            return dataset.select_columns(["sample_id", "source", "sentence", "audio"])


        def allowed_ids(source: str, split: str) -> set[str]:
            if manifest_df.empty:
                return set()
            rows = manifest_df[(manifest_df["source"] == source) & (manifest_df["split"] == split)]
            return set(rows["sample_id"].astype(str))


        def filter_to_ids(dataset: Dataset | IterableDataset, ids: set[str]) -> Dataset | IterableDataset:
            if not ids:
                return dataset.take(0) if isinstance(dataset, IterableDataset) else dataset.select([])
            return dataset.filter(
                lambda sample_id: str(sample_id) in ids,
                input_columns=["sample_id"],
            )


        SMOKE_DATASETS: tuple[Dataset, Dataset] | None = None


        def canonical_training_audio(array: Any, sample_rate: int) -> tuple[np.ndarray, int]:
            '''Return one finite, contiguous 16 kHz mono waveform.'''
            waveform = np.asarray(array, dtype=np.float32)
            waveform = np.squeeze(waveform)
            if waveform.ndim == 2:
                if waveform.shape[0] <= 8:
                    waveform = waveform.mean(axis=0)
                elif waveform.shape[1] <= 8:
                    waveform = waveform.mean(axis=1)
                else:
                    raise ValueError(f"Ambiguous multi-channel audio shape: {waveform.shape}")
            if waveform.ndim != 1 or waveform.size == 0:
                raise ValueError(f"Expected non-empty mono audio, received shape {waveform.shape}")
            if not np.isfinite(waveform).all():
                raise ValueError("Audio contains NaN or infinite samples")
            if int(sample_rate) != 16_000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=int(sample_rate), target_sr=16_000)
            return np.ascontiguousarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32), 16_000


        def decode_training_audio(audio: Any) -> tuple[np.ndarray, int]:
            '''Decode HF Audio objects, decoded mappings, raw Parquet bytes, or local paths.'''
            if hasattr(audio, "get_all_samples"):
                samples = audio.get_all_samples()
                array = samples.data.detach().cpu().numpy()
                sample_rate = int(samples.sample_rate)
            elif isinstance(audio, dict) and audio.get("array") is not None:
                array = audio["array"]
                sample_rate = int(audio["sampling_rate"])
            else:
                import soundfile as sf
                if isinstance(audio, dict) and audio.get("bytes") is not None:
                    source = io.BytesIO(audio["bytes"])
                elif isinstance(audio, dict) and audio.get("path"):
                    source = audio["path"]
                elif isinstance(audio, (str, Path)):
                    source = str(audio)
                else:
                    raise TypeError(f"Unsupported audio payload: {type(audio).__name__}")
                array, sample_rate = sf.read(source, dtype="float32", always_2d=False)
            return canonical_training_audio(array, sample_rate)


        def load_waxal(split: str) -> IterableDataset:
            # WAXAL audio is larger than a Kaggle session disk. Streaming keeps
            # shards remote and decodes only examples consumed by a batch.
            dataset = load_dataset(
                "google/WaxalNLP", "dag_asr", split=split, streaming=True,
                token=cfg.hf_token, cache_dir=str(CACHE_DIR),
            )
            ids = allowed_ids("waxal_dag_asr", split)
            if not ids and cfg.stage != "smoke":
                raise RuntimeError(f"No audited WAXAL IDs for {split}")
            dataset = standardize_dataset(dataset, "waxal_dag_asr", accepted_ids=ids or None)
            if split == "train":
                dataset = dataset.shuffle(seed=cfg.seed, buffer_size=cfg.stream_shuffle_buffer)
            return dataset


        def load_waxal_smoke(split: str, rows: int) -> Dataset:
            '''Materialize only a tiny streamed shard so smoke mode does not download 90h.'''
            stream = load_dataset("google/WaxalNLP", "dag_asr", split=split, streaming=True, token=cfg.hf_token)
            materialized = []
            for index, record in enumerate(stream):
                if index >= rows:
                    break
                array, sample_rate = decode_training_audio(record["audio"])
                materialized.append({
                    "sample_id": str(record.get("id", f"{split}-{index}")),
                    "source": "waxal_dag_asr", "sentence": str(record.get("transcription", "")),
                    # A plain variable-length sequence avoids a datasets v4 Audio
                    # recast of heterogeneous nested NumPy shapes. The collator
                    # converts this list back to one float32 waveform per example.
                    "audio": {"array": array.tolist(), "sampling_rate": sample_rate},
                })
            if not materialized:
                raise RuntimeError(f"Unable to stream WAXAL {split} smoke rows")
            dataset = Dataset.from_list(materialized)
            if len(dataset) != rows:
                raise RuntimeError(f"Expected {rows} WAXAL {split} smoke rows, received {len(dataset)}")
            return dataset


        def prepare_smoke_data(config: TrainConfig) -> tuple[Dataset, Dataset] | None:
            '''Materialize and validate smoke data once per process.'''
            global SMOKE_DATASETS
            if config.stage != "smoke":
                return None
            if SMOKE_DATASETS is None:
                SMOKE_DATASETS = (
                    load_waxal_smoke("train", 64),
                    load_waxal_smoke("validation", 32),
                )
            return SMOKE_DATASETS


        def load_domain_source(source: str, split: str) -> Dataset | None:
            repo_id, config_name = HF_SOURCE_REGISTRY[source]
            rows = manifest_df[(manifest_df["source"] == source) & (manifest_df["split"] == split)].copy()
            if rows.empty:
                return None
            try:
                parts = []
                origin_values = rows["origin_split"].fillna("train").astype(str) if "origin_split" in rows else pd.Series(["train"] * len(rows))
                rows = rows.assign(_origin_split=origin_values.values)
                for origin_split, origin_rows in rows.groupby("_origin_split"):
                    dataset = load_dataset(
                        repo_id, config_name, split=origin_split,
                        token=cfg.hf_token, cache_dir=str(CACHE_DIR),
                    )
                    ids = set(origin_rows["sample_id"].astype(str))
                    part = standardize_dataset(dataset, source, accepted_ids=ids)
                    if len(part):
                        parts.append(part)
                return concatenate_datasets(parts) if parts else None
            except Exception as exc:
                print(f"Skipping {source}: {type(exc).__name__}: {exc}")
                return None


        def load_manifest_locator_source(source: str, split: str) -> Dataset | None:
            rows = manifest_df[(manifest_df["source"] == source) & (manifest_df["split"] == split)].copy()
            if rows.empty:
                return None
            cache = WORK_DIR / "domain_audio" / source
            cache.mkdir(parents=True, exist_ok=True)
            usable = []
            for row in rows.to_dict("records"):
                locator = str(row["audio_locator"])
                resolved = Path(locator)
                if re.match(r"https?://", locator):
                    if not cfg.allow_remote_audio_downloads:
                        continue
                    suffix = Path(locator.split("?", 1)[0]).suffix or ".audio"
                    resolved = cache / f"{hashlib.sha256(locator.encode()).hexdigest()}{suffix}"
                    if not resolved.exists():
                        response = requests.get(locator, timeout=120)
                        response.raise_for_status()
                        resolved.write_bytes(response.content)
                if locator.startswith("hf://") or not resolved.exists():
                    continue
                usable.append({
                    "sample_id": str(row["sample_id"]), "source": source,
                    "sentence": str(row["transcript_raw"]), "audio": str(resolved),
                })
            if not usable:
                print(f"Skipping {source}: audited rows exist but no local/allowed audio locator is resolvable")
                return None
            return Dataset.from_list(usable).cast_column("audio", Audio(sampling_rate=16_000))


        def make_training_data(config: TrainConfig) -> tuple[Any, Any]:
            if config.stage == "smoke":
                prepared = prepare_smoke_data(config)
                assert prepared is not None
                return prepared
            train = load_waxal("train")
            validation = load_waxal("validation")
            if config.stage in {"supervised", "medium_pilot", "medium_full"}:
                return train, validation
            if config.stage == "domain_adaptation":
                extras = []
                audited_sources = sorted(set(manifest_df.loc[manifest_df["split"] == "train", "source"].astype(str)))
                for source in audited_sources:
                    if source == "waxal_dag_asr":
                        continue
                    candidate = (
                        load_domain_source(source, "train")
                        if source in HF_SOURCE_REGISTRY
                        else load_manifest_locator_source(source, "train")
                    )
                    if candidate is not None and len(candidate):
                        extras.append(candidate)
                if not extras:
                    raise RuntimeError("No audited domain-adaptation source is loadable")
                extra = concatenate_datasets(extras).shuffle(seed=config.seed)
                if isinstance(train, IterableDataset):
                    extra = extra.to_iterable_dataset(num_shards=max(1, min(16, len(extra))))
                mixed = interleave_datasets(
                    [train.shuffle(seed=config.seed), extra],
                    probabilities=[config.waxal_batch_share, 1.0 - config.waxal_batch_share],
                    seed=config.seed, stopping_strategy="all_exhausted",
                )
                return mixed, validation
            if config.stage == "pseudo_label":
                pseudo_path = artifact_path("pseudo_manifest.parquet")
                pseudo = pd.read_parquet(pseudo_path)
                if pseudo.empty or (pseudo["split"] == "test").any():
                    raise RuntimeError("Pseudo manifest is empty or contains test rows")
                pseudo_text = dict(zip(pseudo["sample_id"].astype(str), pseudo["transcript_raw"].astype(str)))
                raw = load_dataset(
                    "google/WaxalNLP", "dag_asr", split="unlabeled", streaming=True,
                    token=config.hf_token, cache_dir=str(CACHE_DIR),
                )
                raw = standardize_dataset(
                    raw, "waxal_dag_asr_pseudo", text_override=pseudo_text,
                    accepted_ids=set(pseudo_text),
                )
                raw = raw.shuffle(seed=config.seed, buffer_size=config.stream_shuffle_buffer)
                mixed = interleave_datasets(
                    [train.shuffle(seed=config.seed), raw.shuffle(seed=config.seed)],
                    probabilities=[1.0 - config.pseudo_batch_share, config.pseudo_batch_share],
                    seed=config.seed, stopping_strategy="all_exhausted",
                )
                return mixed, validation
            raise AssertionError(config.stage)


        print("Dataset loaders defined. They do not access the test split during training.")
        """
    ),
    markdown("## 5. Whisper processor and exact Dagbani glyph round-trip"),
    code(
        r"""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.base_model_id, task="transcribe", cache_dir=str(CACHE_DIR), token=cfg.hf_token)
        glyph_probe = "n nyɛla bɛ paɣaŋa mini ɔ ka ʒɛm shɛli"
        # Whisper aliases unk/eos/pad to the same <|endoftext|> ID. With the
        # tokenizer's default special tokens, testing for unk therefore mistakes
        # the expected end token for an unknown Dagbani byte. Test only the raw
        # byte-level encoding and then require an exact Unicode round-trip.
        token_ids = processor.tokenizer(glyph_probe, add_special_tokens=False).input_ids
        decoded_probe = processor.tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if unicodedata.normalize("NFC", decoded_probe) != unicodedata.normalize("NFC", glyph_probe):
            raise RuntimeError(f"Whisper tokenizer damaged Dagbani glyphs: {decoded_probe!r}")
        unk_token_id = getattr(processor.tokenizer, "unk_token_id", None)
        if unk_token_id is not None and unk_token_id in token_ids:
            raise RuntimeError("Unexpected <unk> token in Dagbani glyph probe")
        print("Whisper byte-level tokenizer round-trip passed:", decoded_probe, f"({len(token_ids)} tokens)")
        """
    ),
    markdown("## 6. Dynamic feature collation — no full-corpus mel cache"),
    code(
        r"""
        import torch
        from dataclasses import dataclass


        @dataclass
        class SpeechSeq2SeqCollator:
            processor: Any
            decoder_start_token_id: int

            def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
                input_features = []
                label_features = []
                for feature in features:
                    array, sample_rate = decode_training_audio(feature["audio"])
                    inputs = self.processor.feature_extractor(array, sampling_rate=sample_rate)
                    input_features.append({"input_features": inputs.input_features[0]})
                    label_features.append({"input_ids": self.processor.tokenizer(str(feature["sentence"])).input_ids})
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
                label_ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
                if (label_ids[:, 0] == self.decoder_start_token_id).all().item():
                    label_ids = label_ids[:, 1:]
                batch["labels"] = label_ids
                return batch


        decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if decoder_start_token_id is None or decoder_start_token_id < 0:
            raise RuntimeError("Whisper decoder-start token is unavailable")
        collator = SpeechSeq2SeqCollator(processor, decoder_start_token_id)
        if cfg.stage == "smoke" and cfg.use_ddp != "two":
            smoke_train, smoke_validation = prepare_smoke_data(cfg)
            probe_batch = collator([smoke_train[0], smoke_train[1]])
            if probe_batch["input_features"].ndim != 3 or probe_batch["labels"].ndim != 2:
                raise RuntimeError({key: tuple(value.shape) for key, value in probe_batch.items()})
            if (probe_batch["labels"][:, 0] == decoder_start_token_id).any().item():
                raise RuntimeError("Collator retained a duplicated Whisper decoder-start label")
            print({
                "smoke_train_rows": len(smoke_train),
                "smoke_validation_rows": len(smoke_validation),
                "collator_input_shape": tuple(probe_batch["input_features"].shape),
                "collator_label_shape": tuple(probe_batch["labels"].shape),
                "decoder_start_token_id": decoder_start_token_id,
                "audio_contract": "contiguous mono float32 at 16 kHz",
            })
        elif cfg.stage == "smoke":
            # Do not create PyTorch tensors in the notebook parent before
            # notebook_launcher forks. Each DDP child runs this exact contract
            # immediately after loading its tiny dataset.
            print("DDP smoke collator gate deferred to each launched rank.")
        print("Dynamic collator ready; features are computed only for the current batch.")
        """
    ),
    markdown("## 7. Telemetry, wall-clock stopping, and private resume checkpoints"),
    code(
        r"""
        from transformers import TrainerCallback
        from huggingface_hub import HfApi


        class BudgetTelemetryCallback(TrainerCallback):
            def __init__(self, config: TrainConfig, output_dir: Path):
                self.config = config
                self.output_dir = output_dir
                self.started = None
                self.log_path = output_dir / "telemetry.jsonl"

            def on_train_begin(self, args, state, control, **kwargs):
                self.started = time.monotonic()

            def on_step_end(self, args, state, control, **kwargs):
                if not state.is_world_process_zero or not self.started or state.global_step < 1:
                    return control
                elapsed_min = (time.monotonic() - self.started) / 60
                step_min = elapsed_min / state.global_step
                eta_min = step_min * max(0, state.max_steps - state.global_step)
                payload = {
                    "step": state.global_step, "max_steps": state.max_steps,
                    "elapsed_min": elapsed_min, "minutes_per_step": step_min,
                    "eta_min": eta_min, "projected_total_min": elapsed_min + eta_min,
                    "disk_free_gib": shutil.disk_usage(self.output_dir).free / 2**30,
                }
                try:
                    query = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        text=True, timeout=10,
                    ).strip()
                    payload["gpu"] = query
                except Exception:
                    payload["gpu"] = "unavailable"
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
                if state.global_step == 20:
                    print("20-step projection:", json.dumps(payload, indent=2))
                    if payload["projected_total_min"] > self.config.max_session_minutes:
                        print("Projected run exceeds the session budget; saving and stopping safely.")
                        control.should_save = True
                        control.should_training_stop = True
                if elapsed_min >= self.config.max_session_minutes - self.config.stop_buffer_minutes:
                    print("Wall-clock safety buffer reached; saving and stopping safely.")
                    control.should_save = True
                    control.should_training_stop = True
                if payload["disk_free_gib"] < self.config.disk_stop_free_gib:
                    print("Free-disk safety threshold reached; saving and stopping safely.")
                    control.should_save = True
                    control.should_training_stop = True
                return control


        class PrivateHubCheckpointCallback(TrainerCallback):
            '''Keep remote latest, indexed best two, and named major checkpoints.'''
            def __init__(self, config: TrainConfig, output_dir: Path):
                self.config = config
                self.output_dir = output_dir
                self.api = HfApi(token=config.hf_token) if config.hf_token and config.model_repo_id else None
                self.eval_losses: dict[int, float] = {}
                self.futures = []

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics and "eval_loss" in metrics:
                    self.eval_losses[int(state.global_step)] = float(metrics["eval_loss"])

            def _upload(self, folder: Path, remote_path: str, message: str) -> None:
                if not self.api or not folder.exists():
                    return
                future = self.api.upload_folder(
                    repo_id=self.config.model_repo_id, repo_type="model", folder_path=str(folder),
                    path_in_repo=remote_path, commit_message=message, run_as_future=True,
                )
                self.futures.append(future)

            def on_save(self, args, state, control, **kwargs):
                if not state.is_world_process_zero:
                    return control
                step = int(state.global_step)
                folder = Path(args.output_dir) / f"checkpoint-{step}"
                if self.config.upload_full_state:
                    self._upload(folder, f"resume/{self.config.stage}/latest", f"Update {self.config.stage} resumable checkpoint at step {step}")
                ranked = sorted(self.eval_losses.items(), key=lambda item: item[1])[:2]
                if step in {item[0] for item in ranked}:
                    self._upload(folder, f"resume/{self.config.stage}/best-step-{step}", f"Preserve {self.config.stage} top-two checkpoint step {step}")
                if step in self.config.major_checkpoint_steps:
                    self._upload(folder, f"major/{self.config.stage}/checkpoint-{step}", f"Preserve {self.config.stage} major checkpoint step {step}")
                index = {
                    "latest_step": step,
                    "best_two": [{"step": key, "eval_loss": value} for key, value in ranked],
                    "major_steps": list(self.config.major_checkpoint_steps),
                }
                index_path = self.output_dir / "checkpoint_index.json"
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
                if self.api:
                    future = self.api.upload_file(
                        repo_id=self.config.model_repo_id, repo_type="model", path_or_fileobj=str(index_path),
                        path_in_repo=f"resume/{self.config.stage}/checkpoint_index.json", commit_message=f"{self.config.stage} checkpoint index step {step}",
                        run_as_future=True,
                    )
                    self.futures.append(future)
                return control

            def wait(self) -> None:
                for future in self.futures:
                    future.result()


        print("Budget telemetry and resumable private checkpoint callbacks ready.")
        """
    ),
    markdown("## 8. Trainer factory: full fine-tuning by default, LoRA only as fallback"),
    code(
        r"""
        from transformers import AutoModelForSpeechSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed


        def gpu_count_without_cuda_init() -> int:
            try:
                output = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
                return len([line for line in output.splitlines() if line.strip()])
            except Exception:
                return 0


        def requested_world_size(config: TrainConfig) -> int:
            available = gpu_count_without_cuda_init()
            if config.use_ddp == "one":
                return 1
            if config.use_ddp == "two":
                if available < 2:
                    raise RuntimeError("Two-GPU DDP requested but fewer than two GPUs are visible")
                return 2
            return 2 if available >= 2 else 1


        def model_starting_point(config: TrainConfig) -> str:
            if config.stage in {"smoke", "supervised", "medium_pilot", "medium_full"}:
                return "openai/whisper-medium" if config.stage in {"medium_pilot", "medium_full"} else config.base_model_id
            return config.model_repo_id


        def training_args(config: TrainConfig, world_size: int) -> Seq2SeqTrainingArguments:
            accumulation = max(1, math.ceil(config.effective_batch_size / (config.micro_batch_size * world_size)))
            kwargs = dict(
                output_dir=str(OUTPUT_DIR), per_device_train_batch_size=config.micro_batch_size,
                per_device_eval_batch_size=max(1, config.micro_batch_size),
                gradient_accumulation_steps=accumulation, learning_rate=config.learning_rate,
                warmup_steps=min(config.warmup_steps, max(1, config.max_steps() // 10)),
                max_steps=config.max_steps(), lr_scheduler_type="linear", fp16=True,
                gradient_checkpointing=True, remove_unused_columns=False,
                dataloader_num_workers=config.dataloader_workers, dataloader_pin_memory=True,
                save_strategy="steps", save_steps=min(config.save_steps, config.max_steps()),
                eval_steps=min(config.eval_steps, config.max_steps()), logging_steps=10,
                save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss",
                greater_is_better=False, predict_with_generate=False, report_to="none",
                disable_tqdm=True,
                seed=config.seed, data_seed=config.seed, ddp_find_unused_parameters=False,
                # Exact sample skipping is prohibitively expensive for a remote
                # streaming corpus. Optimizer/scheduler/RNG state still resume;
                # the stream restarts from its deterministic epoch shuffle.
                ignore_data_skip=True,
                optim="adamw_bnb_8bit" if importlib.util.find_spec("bitsandbytes") else "adamw_torch_fused",
                push_to_hub=bool(config.model_repo_id), hub_model_id=config.model_repo_id or None,
                hub_private_repo=True, hub_token=config.hf_token, hub_strategy="end",
            )
            signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
            kwargs["eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"] = "steps"
            return Seq2SeqTrainingArguments(**kwargs)


        def build_model(config: TrainConfig):
            start = os.getenv("DAGBANI_MODEL_SNAPSHOT") or model_starting_point(config)
            # Full AMP fine-tuning keeps trainable/master weights in FP32. Trainer
            # autocast performs T4 compute in FP16; loading trainable weights as
            # FP16 makes GradScaler reject their already-FP16 gradients.
            load_dtype = torch.float32
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "cache_dir": str(CACHE_DIR),
                "token": config.hf_token,
                "local_files_only": bool(os.getenv("DAGBANI_MODEL_SNAPSHOT")),
            }
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    start, dtype=load_dtype, **load_kwargs,
                )
            except TypeError:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    start, torch_dtype=load_dtype, **load_kwargs,
                )
            model.config.forced_decoder_ids = None
            model.generation_config.forced_decoder_ids = None
            model.generation_config.task = "transcribe"
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            if config.training_mode == "lora":
                from peft import LoraConfig, TaskType, get_peft_model
                lora = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                    target_modules=["q_proj", "v_proj", "out_proj"], task_type=TaskType.SEQ_2_SEQ_LM,
                )
                model = get_peft_model(model, lora)
                model.print_trainable_parameters()
            trainable_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.requires_grad}
            if trainable_dtypes != {torch.float32}:
                raise RuntimeError(f"AMP requires FP32 trainable weights, received {trainable_dtypes}")
            print(f"Trainable-weight dtype gate passed: {trainable_dtypes}", flush=True)
            return model


        def latest_local_checkpoint(output_dir: Path) -> str | None:
            checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
            return str(checkpoints[-1]) if checkpoints else None


        def resumable_checkpoint(config: TrainConfig, output_dir: Path) -> str | None:
            local = latest_local_checkpoint(output_dir)
            if local or not config.resume or not config.model_repo_id:
                return local
            from huggingface_hub import snapshot_download
            remote_root = WORK_DIR / f"remote_resume_{config.stage}"
            try:
                snapshot_download(
                    repo_id=config.model_repo_id, repo_type="model", token=config.hf_token,
                    allow_patterns=[f"resume/{config.stage}/latest/**"], local_dir=str(remote_root),
                )
                candidate = remote_root / "resume" / config.stage / "latest"
                if (candidate / "trainer_state.json").exists():
                    print(f"Resuming from private Hub state: {candidate}")
                    return str(candidate)
            except Exception as exc:
                print(f"No compatible remote resume state: {type(exc).__name__}: {exc}")
            return None


        def train_process(config_dict: dict[str, Any]) -> dict[str, Any]:
            config = TrainConfig(**config_dict)
            set_seed(config.seed)
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            print(f"[rank {rank}/{world_size}] preparing training data", flush=True)
            train_data, validation_data = make_training_data(config)
            def size_label(dataset: Any) -> str:
                try:
                    return f"{len(dataset):,}"
                except TypeError:
                    return "streaming"
            print(
                f"[rank {rank}/{world_size}] data ready: "
                f"train={size_label(train_data)}, validation={size_label(validation_data)}; loading model",
                flush=True,
            )
            if config.stage == "smoke" and world_size > 1:
                probe_batch = collator([train_data[0], train_data[1]])
                if probe_batch["input_features"].ndim != 3 or probe_batch["labels"].ndim != 2:
                    raise RuntimeError({key: tuple(value.shape) for key, value in probe_batch.items()})
                if (probe_batch["labels"][:, 0] == decoder_start_token_id).any().item():
                    raise RuntimeError("Collator retained a duplicated Whisper decoder-start label")
                print(
                    f"[rank {rank}/{world_size}] deferred collator gate passed: "
                    f"inputs={tuple(probe_batch['input_features'].shape)}, "
                    f"labels={tuple(probe_batch['labels'].shape)}",
                    flush=True,
                )
            model = build_model(config)
            print(f"[rank {rank}/{world_size}] model ready; building Trainer", flush=True)
            telemetry = BudgetTelemetryCallback(config, OUTPUT_DIR)
            hub_checkpoints = PrivateHubCheckpointCallback(config, OUTPUT_DIR)
            args = training_args(config, world_size)
            trainer = Seq2SeqTrainer(
                model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
                data_collator=SpeechSeq2SeqCollator(processor, decoder_start_token_id),
                processing_class=processor,
                callbacks=[telemetry, hub_checkpoints],
            )
            resume_checkpoint = resumable_checkpoint(config, OUTPUT_DIR)
            print(f"[rank {rank}/{world_size}] Trainer ready; starting optimizer steps", flush=True)
            result = trainer.train(resume_from_checkpoint=resume_checkpoint)
            # Synchronize both ranks before and after final persistence. Only rank
            # zero may write/upload the shared final_model directory; otherwise
            # two Kaggle workers can race while serializing processor files.
            trainer.accelerator.wait_for_everyone()
            if trainer.is_world_process_zero():
                trainer.save_model(str(OUTPUT_DIR / "final_model"))
                processor.save_pretrained(str(OUTPUT_DIR / "final_model"))
                if config.model_repo_id:
                    trainer.push_to_hub(commit_message=f"Dagbani ASR {config.stage} completed at step {trainer.state.global_step}")
                    hub_checkpoints.wait()
            trainer.accelerator.wait_for_everyone()
            return {"metrics": result.metrics, "global_step": trainer.state.global_step, "world_size": world_size}


        world_size = requested_world_size(cfg)
        effective_accumulation = math.ceil(cfg.effective_batch_size / (cfg.micro_batch_size * max(1, world_size)))
        print({
            "visible_gpus": gpu_count_without_cuda_init(), "requested_world_size": world_size,
            "micro_batch": cfg.micro_batch_size, "gradient_accumulation": effective_accumulation,
            "effective_batch": cfg.micro_batch_size * world_size * effective_accumulation,
            "max_steps": cfg.max_steps(), "starting_model": model_starting_point(cfg),
        })
        """
    ),
    markdown(
        """
        ## 9. Launch exactly one training stage

        For two GPUs, this uses `accelerate.notebook_launcher`, which starts one process
        per GPU. Do not run unrelated CUDA cells before this launcher. Set
        `cfg.run_training=True` only after the preflight report is correct.
        """
    ),
    code(
        r"""
        training_result = None
        if cfg.run_training:
            from accelerate import notebook_launcher
            from huggingface_hub import HfApi, snapshot_download
            api = HfApi(token=cfg.hf_token)
            api.create_repo(cfg.model_repo_id, repo_type="model", private=True, exist_ok=True)
            if world_size > 1:
                if torch.cuda.is_initialized():
                    raise RuntimeError(
                        "Two-GPU notebook DDP is unsafe because this Kaggle kernel has already "
                        "initialized CUDA. Use use_ddp='one'. A spawn-based standalone launcher "
                        "is required for two GPUs in this environment."
                    )
                start_id = model_starting_point(cfg)
                repo_files = api.list_repo_files(start_id, repo_type="model")
                root_files = [name for name in repo_files if "/" not in name]
                weight_files = [name for name in root_files if name.endswith(".safetensors")]
                if not weight_files:
                    weight_files = [name for name in root_files if name.endswith(".bin")]
                metadata_files = [
                    name for name in root_files
                    if name.endswith((".json", ".txt", ".model", ".tiktoken"))
                ]
                if not weight_files:
                    raise RuntimeError(f"No root model weights found in {start_id}")
                model_snapshot = snapshot_download(
                    repo_id=start_id, repo_type="model", token=cfg.hf_token,
                    cache_dir=str(CACHE_DIR), allow_patterns=weight_files + metadata_files,
                )
                os.environ["DAGBANI_MODEL_SNAPSHOT"] = model_snapshot
                print(f"DDP model snapshot prepared before fork: {model_snapshot}")
                notebook_launcher(train_process, args=(asdict(cfg),), num_processes=world_size, mixed_precision="fp16")
                training_result = {"status": "DDP child processes completed", "world_size": world_size}
            else:
                training_result = train_process(asdict(cfg))
            print(json.dumps(training_result, indent=2, default=str))
        else:
            print("Dry preflight complete. Training was not launched.")
        """
    ),
    markdown("## 10. Full generation evaluation for final and preserved major checkpoints"),
    code(
        r"""
        import jiwer


        def strict_text(text: Any) -> str:
            return unicodedata.normalize("NFC", str(text or "")).strip()


        def asr_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
            if len(references) != len(hypotheses):
                raise ValueError("Reference and hypothesis counts differ")
            return {
                "strict_wer": float(jiwer.wer([strict_text(x) for x in references], [strict_text(x) for x in hypotheses])),
                "strict_cer": float(jiwer.cer([strict_text(x) for x in references], [strict_text(x) for x in hypotheses])),
                "normalized_wer": float(jiwer.wer([normalize_eval(x) for x in references], [normalize_eval(x) for x in hypotheses])),
                "normalized_cer": float(jiwer.cer([normalize_eval(x) for x in references], [normalize_eval(x) for x in hypotheses])),
                "utterances": len(references),
            }


        def evaluation_dataset(source: str, split: str) -> Dataset | IterableDataset:
            if source == "waxal_dag_asr":
                if cfg.stage == "smoke" and split == "validation":
                    prepared = prepare_smoke_data(cfg)
                    assert prepared is not None
                    return prepared[1]
                return load_waxal(split)
            candidate = (
                load_domain_source(source, split)
                if source in HF_SOURCE_REGISTRY
                else load_manifest_locator_source(source, split)
            )
            if candidate is None or not len(candidate):
                raise RuntimeError(f"No resolvable audited rows for {source}/{split}")
            return candidate


        def evaluate_generation(
            model_id_or_path: str,
            split: str = "validation",
            limit: int | None = None,
            source: str = "waxal_dag_asr",
        ) -> tuple[dict, pd.DataFrame]:
            from transformers import AutoModelForSpeechSeq2Seq
            inference_kwargs = {
                "low_cpu_mem_usage": True,
                "token": cfg.hf_token,
                "cache_dir": str(CACHE_DIR),
            }
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id_or_path, dtype=torch.float16, **inference_kwargs,
                ).to("cuda")
            except TypeError:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id_or_path, torch_dtype=torch.float16, **inference_kwargs,
                ).to("cuda")
            model.eval()
            model.generation_config.forced_decoder_ids = None
            model.generation_config.task = "transcribe"
            dataset = evaluation_dataset(source, split)
            if limit is not None:
                dataset = (
                    dataset.take(limit)
                    if isinstance(dataset, IterableDataset)
                    else dataset.select(range(min(limit, len(dataset))))
                )
            references, hypotheses, sample_ids = [], [], []
            started = time.time()
            for index, row in enumerate(dataset, 1):
                array, sample_rate = decode_training_audio(row["audio"])
                inputs = processor(
                    array, sampling_rate=sample_rate, return_tensors="pt",
                    return_attention_mask=True,
                )
                features = inputs.input_features.to(device=model.device, dtype=model.dtype)
                attention_mask = inputs.attention_mask.to(model.device)
                with torch.inference_mode():
                    tokens = model.generate(
                        input_features=features,
                        attention_mask=attention_mask,
                        max_length=225,
                    )
                hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                sample_ids.append(str(row["sample_id"]))
                references.append(strict_text(row["sentence"]))
                hypotheses.append(strict_text(hypothesis))
                if index % 50 == 0:
                    print(f"Generated {index} utterances in {(time.time() - started) / 60:.1f} min")
            metrics = asr_metrics(references, hypotheses) | {
                "model": model_id_or_path, "source": source, "split": split,
                "elapsed_s": time.time() - started,
            }
            predictions = pd.DataFrame({"sample_id": sample_ids, "reference": references, "hypothesis": hypotheses})
            del model
            torch.cuda.empty_cache()
            return metrics, predictions


        def resolve_major_checkpoint(step: int) -> str | None:
            local = OUTPUT_DIR / f"checkpoint-{step}"
            if local.exists():
                return str(local)
            if not cfg.model_repo_id:
                return None
            from huggingface_hub import snapshot_download
            download_root = EVAL_DIR / f"remote_major_{step}"
            try:
                snapshot_download(
                    repo_id=cfg.model_repo_id, repo_type="model", token=cfg.hf_token,
                    allow_patterns=[f"major/{cfg.stage}/checkpoint-{step}/**"], local_dir=str(download_root),
                )
                candidate = download_root / "major" / cfg.stage / f"checkpoint-{step}"
                return str(candidate) if (candidate / "config.json").exists() else None
            except Exception as exc:
                print(f"Major checkpoint {step} unavailable: {type(exc).__name__}: {exc}")
                return None


        generation_metrics = []
        auto_smoke_generation = cfg.stage == "smoke" and training_result is not None
        if cfg.run_full_generation_eval or auto_smoke_generation:
            final_local = OUTPUT_DIR / "final_model"
            targets = [str(final_local) if final_local.exists() else cfg.model_repo_id]
            if not auto_smoke_generation:
                targets.extend(target for step in cfg.major_checkpoint_steps if (target := resolve_major_checkpoint(step)))
            for target in targets:
                generation_limit = 8 if auto_smoke_generation and cfg.eval_generation_limit is None else cfg.eval_generation_limit
                metrics, predictions = evaluate_generation(target, limit=generation_limit)
                generation_metrics.append(metrics)
                stem = Path(target).name
                predictions.to_parquet(EVAL_DIR / f"{stem}_validation_predictions.parquet", index=False)
                print(json.dumps(metrics, indent=2))
            (EVAL_DIR / "generation_metrics.json").write_text(json.dumps(generation_metrics, indent=2), encoding="utf-8")
        else:
            print("Generation evaluation disabled. Enable it for selected/major checkpoints, not every 250-step save.")

        external_metrics = []
        if cfg.run_external_generation_eval:
            final_target = str(OUTPUT_DIR / "final_model") if (OUTPUT_DIR / "final_model").exists() else cfg.model_repo_id
            external_sources = sorted(set(manifest_df[manifest_df["source"] != "waxal_dag_asr"]["source"].astype(str)))
            for source in external_sources:
                source_rows = manifest_df[manifest_df["source"] == source]
                split = next((name for name in ("external_test", "test", "validation") if (source_rows["split"] == name).any()), None)
                if not split:
                    continue
                try:
                    metrics, predictions = evaluate_generation(
                        final_target, split=split, limit=cfg.external_eval_limit, source=source,
                    )
                    external_metrics.append(metrics)
                    predictions.to_parquet(EVAL_DIR / f"external_{source}_{split}.parquet", index=False)
                    print(json.dumps(metrics, indent=2))
                except Exception as exc:
                    external_metrics.append({"source": source, "split": split, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
            (EVAL_DIR / "external_metrics.json").write_text(json.dumps(external_metrics, indent=2), encoding="utf-8")
        else:
            print("External-domain evaluation disabled. Enable it before a release decision.")
        """
    ),
    markdown("## 11. Confidence-calibrated pseudo-labelling of at most 20 initial hours"),
    code(
        r"""
        def compression_ratio(text: str) -> float:
            raw = text.encode("utf-8")
            return len(raw) / max(1, len(gzip.compress(raw)))


        def repetition_ratio(text: str, n: int = 3) -> float:
            tokens = normalize_eval(text).split()
            if len(tokens) < n:
                return 0.0
            grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            return 1.0 - len(set(grams)) / max(1, len(grams))


        def utterance_wer(reference: str, hypothesis: str) -> float:
            return float(jiwer.wer(normalize_eval(reference), normalize_eval(hypothesis)))


        def decode_with_confidence(model, array: np.ndarray, sample_rate: int, num_beams: int) -> dict[str, Any]:
            array, sample_rate = canonical_training_audio(array, sample_rate)
            inputs = processor(
                array, sampling_rate=sample_rate, return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device=model.device, dtype=model.dtype)
            attention_mask = inputs.attention_mask.to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    input_features=features, attention_mask=attention_mask,
                    max_length=cfg.pseudo_max_new_tokens,
                    num_beams=num_beams, return_dict_in_generate=True, output_scores=True,
                )
            beam_indices = getattr(output, "beam_indices", None)
            transition = model.compute_transition_scores(output.sequences, output.scores, beam_indices, normalize_logits=True)
            valid_scores = transition[transition < 0]
            average_logprob = float(valid_scores.mean().item()) if valid_scores.numel() else -99.0
            no_speech_id = processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")
            first_logits = output.scores[0][0]
            no_speech_probability = float(torch.softmax(first_logits.float(), dim=-1)[no_speech_id].item())
            text = strict_text(processor.batch_decode(output.sequences, skip_special_tokens=True)[0])
            return {"text": text, "average_logprob": average_logprob, "no_speech_probability": no_speech_probability}


        def calibrate_pseudo_thresholds(
            model, validation: Dataset | IterableDataset, rows: int,
        ) -> dict[str, float]:
            scores = []
            calibration = (
                validation.take(rows)
                if isinstance(validation, IterableDataset)
                else validation.select(range(min(rows, len(validation))))
            )
            for row in calibration:
                array, sample_rate = decode_training_audio(row["audio"])
                greedy = decode_with_confidence(model, array, sample_rate, 1)
                beam = decode_with_confidence(model, array, sample_rate, 3)
                scores.append(greedy | {
                    "agreement_wer": utterance_wer(greedy["text"], beam["text"]),
                    "reference_wer": utterance_wer(row["sentence"], greedy["text"]),
                })
            good = pd.DataFrame(scores)
            good = good[good["reference_wer"] <= 0.25]
            if len(good) < 10:
                return {"average_logprob": -0.8, "no_speech_probability": 0.20, "agreement_wer": 0.20}
            return {
                "average_logprob": float(good["average_logprob"].quantile(0.10)),
                "no_speech_probability": float(min(0.30, good["no_speech_probability"].quantile(0.95))),
                "agreement_wer": float(min(0.25, good["agreement_wer"].quantile(0.95))),
            }


        def pseudo_label_unlabeled(model_id: str) -> pd.DataFrame:
            from transformers import AutoModelForSpeechSeq2Seq
            pseudo_kwargs = {
                "low_cpu_mem_usage": True,
                "token": cfg.hf_token,
                "cache_dir": str(CACHE_DIR),
            }
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, dtype=torch.float16, **pseudo_kwargs,
                ).to("cuda").eval()
            except TypeError:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch.float16, **pseudo_kwargs,
                ).to("cuda").eval()
            model.generation_config.forced_decoder_ids = None
            model.generation_config.task = "transcribe"
            thresholds = calibrate_pseudo_thresholds(model, load_waxal("validation"), cfg.pseudo_calibration_rows)
            print("Calibrated pseudo-label thresholds:", thresholds)
            stream = load_dataset("google/WaxalNLP", "dag_asr", split="unlabeled", streaming=True, token=cfg.hf_token)
            accepted, accepted_seconds = [], 0.0
            for index, row in enumerate(stream, 1):
                array, sample_rate = decode_training_audio(row["audio"])
                duration = len(array) / max(1, sample_rate)
                greedy = decode_with_confidence(model, array, sample_rate, 1)
                if greedy["average_logprob"] < thresholds["average_logprob"] or greedy["no_speech_probability"] > thresholds["no_speech_probability"]:
                    continue
                beam = decode_with_confidence(model, array, sample_rate, 3)
                agreement = utterance_wer(greedy["text"], beam["text"])
                wps = len(normalize_eval(greedy["text"]).split()) / max(duration, 1e-6)
                if not (
                    agreement <= thresholds["agreement_wer"]
                    and 0.5 <= wps <= 5.0
                    and compression_ratio(greedy["text"]) < 2.4
                    and repetition_ratio(greedy["text"]) < 0.35
                ):
                    continue
                sample_id = str(row.get("id", f"unlabeled-{index}"))
                confidence = float(np.exp(greedy["average_logprob"]) * (1.0 - greedy["no_speech_probability"]) * (1.0 - min(1.0, agreement)))
                accepted.append({
                    "sample_id": sample_id, "source": "waxal_dag_asr", "source_version": "dag_asr",
                    "audio_locator": f"hf://google/WaxalNLP/dag_asr/unlabeled/{sample_id}",
                    "transcript_raw": greedy["text"], "transcript_eval": normalize_eval(greedy["text"]),
                    "speaker_id": (
                        f"waxal_dag_asr:{row.get('speaker_id')}"
                        if str(row.get("speaker_id", "unknown")) != "unknown" else "unknown"
                    ),
                    "speaker_id_raw": str(row.get("speaker_id", "unknown")), "duration_s": duration,
                    "domain": "spontaneous_pseudo", "split": "unlabeled", "license": "cc-by-4.0",
                    "audio_hash": "", "is_pseudo": True, "pseudo_confidence": confidence,
                    "average_logprob": greedy["average_logprob"], "no_speech_probability": greedy["no_speech_probability"],
                    "greedy_beam_wer": agreement,
                })
                accepted_seconds += duration
                if len(accepted) % 50 == 0:
                    print(f"Accepted {len(accepted):,} pseudo labels / {accepted_seconds / 3600:.2f}h after scanning {index:,} rows")
                if accepted_seconds >= cfg.pseudo_initial_hours * 3600:
                    break
            del model
            torch.cuda.empty_cache()
            return pd.DataFrame(accepted)


        if cfg.run_pseudo_labelling:
            if cfg.stage != "pseudo_label":
                raise RuntimeError("Set stage='pseudo_label' before generating pseudo labels")
            pseudo_df = pseudo_label_unlabeled(cfg.model_repo_id)
            if pseudo_df.empty:
                raise RuntimeError("No unlabeled rows passed calibrated confidence filters")
            pseudo_file = EVAL_DIR / "pseudo_manifest.parquet"
            pseudo_df.to_parquet(pseudo_file, index=False)
            from huggingface_hub import HfApi
            HfApi(token=cfg.hf_token).upload_file(
                repo_id=cfg.manifest_repo_id, repo_type="dataset", path_or_fileobj=str(pseudo_file),
                path_in_repo="phase1/pseudo_manifest.parquet", commit_message="Add confidence-filtered Dagbani pseudo labels",
            )
            print(f"Published {len(pseudo_df):,} pseudo labels covering {pseudo_df.duration_s.sum() / 3600:.2f}h")
        else:
            print("Pseudo-labelling disabled. It is allowed only after the supervised model passes validation.")
        """
    ),
    markdown("## 12. Release, regression, and Medium-scaling gates"),
    code(
        r"""
        def bootstrap_wer_difference(
            references: list[str], candidate: list[str], baseline: list[str],
            samples: int = 1_000, seed: int = 42,
        ) -> dict[str, float]:
            if not (len(references) == len(candidate) == len(baseline)):
                raise ValueError("Bootstrap inputs must have equal length")
            rng = np.random.default_rng(seed)
            differences = []
            for _ in range(samples):
                indices = rng.integers(0, len(references), size=len(references))
                refs = [normalize_eval(references[i]) for i in indices]
                cand = [normalize_eval(candidate[i]) for i in indices]
                base = [normalize_eval(baseline[i]) for i in indices]
                differences.append(float(jiwer.wer(refs, cand) - jiwer.wer(refs, base)))
            low, high = np.quantile(differences, [0.025, 0.975])
            return {"mean_difference": float(np.mean(differences)), "ci95_low": float(low), "ci95_high": float(high)}


        def release_decision(candidate_metrics: dict, domain_regressions: dict[str, float], bootstrap: dict | None = None) -> dict:
            candidate_wer = float(candidate_metrics["normalized_wer"])
            beats_by_one_point = candidate_wer <= 0.33
            significant = bool(bootstrap and bootstrap["ci95_high"] < 0.0)
            domain_safe = all(regression <= 0.02 for regression in domain_regressions.values())
            return {
                "accepted": bool((beats_by_one_point or significant) and domain_safe),
                "candidate_wer": candidate_wer,
                "beats_34pct_by_one_point": beats_by_one_point,
                "bootstrap_supports_improvement": significant,
                "domain_regressions": domain_regressions,
                "research_only": True,
                "reason": "Native-speaker listening review is still required before production use.",
            }


        def medium_gate(
            small_waxal_wer: float, small_pilot_wer: float, medium_pilot_wer: float,
            remaining_quota_hours: float, medium_projected_hours: float, medium_oom: bool,
        ) -> dict:
            checks = {
                "small_reproducible": small_waxal_wer <= 0.35,
                "matched_pilot_gain": medium_pilot_wer <= small_pilot_wer - 0.01,
                "fits_memory": not medium_oom,
                "fits_remaining_quota": medium_projected_hours + 1.0 <= remaining_quota_hours,
            }
            return {"allow_medium_full_run": all(checks.values()), "checks": checks}


        def save_medium_gate(decision: dict, evidence: dict) -> Path:
            if not decision.get("allow_medium_full_run"):
                raise ValueError(f"Medium gate did not pass: {decision}")
            path = Path(cfg.medium_gate_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(decision | {"evidence": evidence}, indent=2), encoding="utf-8")
            print(f"Saved passing Medium authorization to {path}")
            return path


        print("Acceptance gates loaded. Test is evaluated once, only after validation selects the final checkpoint.")
        print("Reproduction target: WER <= 0.35 and CER <= 0.13. Release target: WER <= 0.33 or significant gain.")
        """
    ),
    markdown("## 13. Generate the research model card and run manifest"),
    code(
        r"""
        def write_run_evidence(metrics: list[dict] | None = None) -> None:
            run_manifest = {
                "created_utc": pd.Timestamp.utcnow().isoformat(),
                "config": asdict(cfg) | {"hf_token": "<redacted>"},
                "phase1_readiness": readiness,
                "manifest_sha256": hashlib.sha256(pd.util.hash_pandas_object(manifest_df, index=True).values.tobytes()).hexdigest() if not manifest_df.empty else None,
                "metrics": metrics or [],
                "data_access": {
                    "waxal_supervised": "streaming",
                    "shuffle_buffer": cfg.stream_shuffle_buffer,
                    "resume_sample_skip": False,
                },
                "research_only": True,
            }
            (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            card = textwrap.dedent(f'''\
            ---
            language: dag
            license: other
            license_name: research-only pending corpus license review
            library_name: transformers
            pipeline_tag: automatic-speech-recognition
            ---

            # Dagbani Whisper ASR — Research Model

            Fine-tuned from `{cfg.base_model_id}` using the audited Dagbani manifest revision
            `{run_manifest['manifest_sha256']}`. The model is intended for research on
            conversational Dagbani ASR.

            ## Important limitation

            No native-speaker transcript/listening review was available for this run. The
            model must not be represented as production-ready or used for high-stakes
            decisions until native-speaker validation is completed.

            ## Transcript policy

            Ground truth was preserved. Evaluation normalization used only Unicode NFC,
            lowercase, whitespace cleanup, and punctuation removal; it did not infer tone
            or replace ambiguous Dagbani letters.
            ''').strip() + "\n"
            (OUTPUT_DIR / "README.md").write_text(card, encoding="utf-8")
            print(f"Run evidence written to {OUTPUT_DIR}")


        write_run_evidence(generation_metrics + external_metrics)
        """
    ),
]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_PATH.write_text(
        json.dumps(notebook(PHASE1_CELLS, "Dagbani ASR Phase 1 — Audit and Baselines"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    BASELINE_PATH.write_text(
        json.dumps(notebook(BASELINE_CELLS, "Dagbani ASR Phase 1B — Baseline Only", accelerator="gpu"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    # Phase 2 is appended below to keep the notebooks generated from one source.
    PHASE2_PATH.write_text(
        json.dumps(notebook(PHASE2_CELLS, "Dagbani ASR Phase 2 — Budget-Aware Training", accelerator="gpu"), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {PHASE1_PATH}")
    print(f"Wrote {BASELINE_PATH}")
    print(f"Wrote {PHASE2_PATH}")


if __name__ == "__main__":
    main()
