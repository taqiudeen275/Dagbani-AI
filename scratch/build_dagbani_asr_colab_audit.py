"""Build the standalone, resumable Colab full-audit notebook.

The notebook deliberately reuses the already release-ready WAXAL audit and performs
a fresh, fully decoded Bible audit.  This removes redundant WAXAL network/audio work
without weakening exact-hash or split-leakage checks in the combined manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01c_Dagbani_ASR_Full_Audit_Colab_Optimized.ipynb"


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


CELLS = [
    markdown(
        r"""
        # Dagbani ASR Phase 1C — Optimized Colab Full Audit

        This is a **standalone CPU notebook** for the combined WAXAL + Dagbani Bible
        audit. It is faster than rerunning the original full audit because it:

        - imports the immutable, release-ready WAXAL audit from
          `ats-tech/dagbani-asr-phase1`;
        - fully decodes and audits only the Bible corpus;
        - processes audio with a bounded thread pool;
        - writes resumable compressed shards to Google Drive;
        - publishes only if every blocking acceptance check passes.

        Navigation is intentionally quarantined because native listening indicated that
        its samples do not sound like Dagbani. No missing source is replaced with
        synthetic audio.

        **Colab setting:** Runtime → Change runtime type → Hardware accelerator →
        **None**. A GPU does not accelerate this audit.
        """
    ),
    markdown("## 1. Install only missing dependencies"),
    code(
        r"""
        import importlib.util
        import subprocess
        import sys

        REQUIRED = {
            "datasets": "datasets[audio]>=3.2,<5",
            "huggingface_hub": "huggingface_hub[hf_xet]>=0.27,<2",
            "soundfile": "soundfile>=0.12,<1",
            "librosa": "librosa>=0.10,<1",
            "pyarrow": "pyarrow>=17,<25",
            "pandas": "pandas>=2.1,<4",
        }
        missing = [spec for module, spec in REQUIRED.items() if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("Environment ready. This notebook does not install or use a GPU runtime.")
        """,
        "setup",
    ),
    markdown("## 2. Configuration, secrets, and resumable storage"),
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
        import math
        import os
        import platform
        import re
        import shutil
        import string
        import sys
        import time
        import unicodedata
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import asdict, dataclass, field
        from pathlib import Path
        from typing import Any, Iterable

        import numpy as np
        import pandas as pd
        import requests
        import soundfile as sf
        from IPython.display import Audio as NotebookAudio, display


        def colab_secret(name: str) -> str | None:
            value = os.getenv(name)
            if value:
                return value
            try:
                from google.colab import userdata
                return userdata.get(name)
            except Exception:
                return None


        @dataclass
        class AuditConfig:
            audit_mode: str = "full"
            work_dir: str = "/content/dagbani_asr_phase1_colab"
            drive_dir: str = "/content/drive/MyDrive/dagbani_asr_phase1_colab"
            mount_drive: bool = True
            hf_token: str | None = field(default_factory=lambda: colab_secret("HF_TOKEN"))
            manifest_repo_id: str = "ats-tech/dagbani-asr-phase1-v2"
            reuse_waxal_repo_id: str = "ats-tech/dagbani-asr-phase1"
            reuse_waxal_revision: str | None = None
            publish_artifacts: bool = True
            bible_review_passed: bool = False
            bible_review_rows: int = 20
            seed: int = 42
            decode_workers: int = min(4, max(1, os.cpu_count() or 1))
            shard_rows: int = 500
            min_duration_s: float = 0.5
            max_duration_s: float = 30.0
            min_words_per_s: float = 0.25
            max_words_per_s: float = 8.0
            enabled_sources: tuple[str, ...] = ("waxal_dag_asr", "dagbani_bible")
            allow_network_audio_downloads: bool = False

            def validate(self) -> None:
                if self.audit_mode != "full":
                    raise ValueError("This optimized notebook is intentionally full-audit only")
                if self.enabled_sources != ("waxal_dag_asr", "dagbani_bible"):
                    raise ValueError("Only audited WAXAL + Bible are allowed in this notebook")
                if not self.hf_token:
                    raise ValueError("Add a Colab secret named HF_TOKEN with read/write access")
                if not 1 <= self.decode_workers <= 8:
                    raise ValueError("decode_workers must stay between 1 and 8")
                if self.shard_rows < 100:
                    raise ValueError("shard_rows must be at least 100")


        cfg = AuditConfig()
        cfg.validate()

        if cfg.mount_drive:
            from google.colab import drive
            drive.mount("/content/drive")

        WORK_DIR = Path(cfg.work_dir)
        ARTIFACT_DIR = WORK_DIR / "artifacts"
        CACHE_DIR = WORK_DIR / "hf_cache"
        LOCAL_PART_DIR = WORK_DIR / "resume_parts"
        PERSIST_PART_DIR = Path(cfg.drive_dir) / "resume_parts" if cfg.mount_drive else LOCAL_PART_DIR
        for directory in (WORK_DIR, ARTIFACT_DIR, CACHE_DIR, LOCAL_PART_DIR, PERSIST_PART_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        # Keep heavy cache and decoding on the Colab VM. Drive receives only small,
        # completed resume shards; reading audio directly from Drive is slower.
        os.environ["HF_HOME"] = str(CACHE_DIR)
        os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

        try:
            import torch
            if torch.cuda.is_available():
                print("WARNING: A GPU runtime is active but will remain unused. Switch accelerator to None.")
        except Exception:
            pass

        public_config = asdict(cfg) | {"hf_token": "<set>"}
        print(json.dumps(public_config, indent=2))
        print({"cpu_count": os.cpu_count(), "decode_workers": cfg.decode_workers})
        """
    ),
    markdown("## 3. Conservative canonical helpers"),
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
            value = unicodedata.normalize("NFC", str(text or "")).lower()
            return " ".join(value.translate(PUNCT_TRANSLATION).split())


        def strict_text(text: Any) -> str:
            return unicodedata.normalize("NFC", str(text or "")).strip()


        def unexpected_characters(text: str) -> list[str]:
            allowed = {"Ll", "Lu", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd", "Zs", "Po", "Pd"}
            return sorted({ch for ch in text if unicodedata.category(ch) not in allowed and ch not in "\n\t"})


        def stable_split(group_id: str, train: int = 80, validation: int = 10) -> str:
            bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            return "train" if bucket < train else "validation" if bucket < train + validation else "test"


        def bible_group(record: dict[str, Any], locator: str) -> str | None:
            for key in ("book_chapter", "chapter", "book", "group_id"):
                value = record.get(key)
                if value not in (None, ""):
                    return f"{record.get('book', 'book')}:{value}"
            match = re.search(
                r"(?i)(genesis|exodus|leviticus|numbers|deuteronomy|[1-3]?\s*[a-z]+)[_\-/ ]*(\d{1,3})",
                locator,
            )
            return f"{match.group(1).lower()}:{match.group(2)}" if match else None


        def audio_payload(value: Any) -> tuple[np.ndarray, int]:
            if hasattr(value, "get_all_samples"):
                samples = value.get_all_samples()
                array = samples.data.detach().cpu().numpy()
                sample_rate = int(samples.sample_rate)
            elif isinstance(value, dict) and value.get("array") is not None:
                array = np.asarray(value["array"], dtype=np.float32)
                sample_rate = int(value.get("sampling_rate") or 16_000)
            else:
                raw_bytes = value.get("bytes") if isinstance(value, dict) else None
                path = value.get("path") if isinstance(value, dict) else value if isinstance(value, (str, os.PathLike)) else None
                if raw_bytes:
                    source = io.BytesIO(raw_bytes)
                elif path and re.match(r"https?://", str(path)):
                    if not cfg.allow_network_audio_downloads:
                        raise PermissionError("Direct network audio download is disabled")
                    response = requests.get(str(path), timeout=60)
                    response.raise_for_status()
                    source = io.BytesIO(response.content)
                elif path:
                    source = str(path)
                else:
                    raise ValueError("Unsupported or empty audio payload")
                array, sample_rate = sf.read(source, dtype="float32", always_2d=False)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim == 2:
                if array.shape[0] <= 8:
                    array = array.mean(axis=0)
                elif array.shape[1] <= 8:
                    array = array.mean(axis=1)
                else:
                    raise ValueError(f"Ambiguous multichannel shape: {array.shape}")
            if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError("Audio must be finite, non-empty, and mono")
            return np.ascontiguousarray(array), int(sample_rate)


        def canonical_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
            import librosa
            if sample_rate != 16_000:
                array = librosa.resample(array, orig_sr=sample_rate, target_sr=16_000)
            peak = float(np.max(np.abs(array))) if len(array) else 0.0
            if peak > 1.0:
                array = array / peak
            return np.ascontiguousarray(np.clip(array, -1.0, 1.0), dtype=np.float32)


        def exact_audio_hash(array: np.ndarray) -> str:
            pcm16 = np.round(np.clip(array, -1, 1) * 32767).astype("<i2")
            return hashlib.sha256(pcm16.tobytes()).hexdigest()


        def spectral_fingerprint(array: np.ndarray) -> str:
            import librosa
            if len(array) < 800:
                return ""
            mel = librosa.feature.melspectrogram(
                y=array, sr=16_000, n_fft=400, hop_length=320, n_mels=24
            )
            log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
            old_x = np.linspace(0.0, 1.0, log_mel.shape[1])
            new_x = np.linspace(0.0, 1.0, 32)
            resized = np.vstack([np.interp(new_x, old_x, row) for row in log_mel])
            quantized = np.clip(np.round((resized + 80.0) * 3.0), 0, 255).astype(np.uint8)
            return hashlib.sha256(quantized.tobytes()).hexdigest()


        assert normalize_eval("  N NYƐLA—BƐ!  ") == "n nyɛla bɛ"
        print("Canonical helpers passed. transcript_raw will not be corrected or tone-inferred.")
        """
    ),
    markdown("## 4. Discover the two admitted sources and record navigation quarantine"),
    code(
        r"""
        from datasets import Audio, get_dataset_config_info, get_dataset_config_names, load_dataset
        from huggingface_hub import HfApi, hf_hub_download


        @dataclass(frozen=True)
        class SourceSpec:
            name: str
            repo_id: str
            config_name: str | None
            domain: str
            expected_license: str
            text_fields: tuple[str, ...] = ("transcription", "sentence", "text", "transcript")
            audio_fields: tuple[str, ...] = ("audio", "speech")
            speaker_fields: tuple[str, ...] = ("speaker_id", "client_id", "speaker", "user_id")
            id_fields: tuple[str, ...] = ("id", "sample_id", "path", "file", "filename")


        WAXAL = SourceSpec("waxal_dag_asr", "google/WaxalNLP", "dag_asr", "spontaneous", "cc-by-4.0")
        BIBLE = SourceSpec(
            "dagbani_bible", "ghananlpcommunity/dagbani-bible-audio-text-tts",
            None, "bible_read", "cc-by-nc-4.0",
        )


        def split_info_value(value: Any, field_name: str) -> int | None:
            raw = value.get(field_name) if isinstance(value, dict) else getattr(value, field_name, None)
            return int(raw) if raw is not None else None


        def discover(spec: SourceSpec) -> dict[str, Any]:
            report = asdict(spec) | {"status": "unavailable", "splits": {}}
            try:
                configs = get_dataset_config_names(spec.repo_id, token=cfg.hf_token)
                resolved = spec.config_name or (configs[0] if len(configs) == 1 else None)
                info = get_dataset_config_info(spec.repo_id, resolved, token=cfg.hf_token)
                hub_info = HfApi(token=cfg.hf_token).dataset_info(spec.repo_id)
                report.update({
                    "status": "available",
                    "resolved_config": resolved,
                    "revision": hub_info.sha,
                    "license": str(getattr(info, "license", "") or spec.expected_license),
                    "splits": {
                        name: {
                            "num_examples": split_info_value(value, "num_examples"),
                            "num_bytes": split_info_value(value, "num_bytes"),
                        }
                        for name, value in (getattr(info, "splits", {}) or {}).items()
                    },
                })
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
            return report


        waxal_report = discover(WAXAL)
        bible_report = discover(BIBLE)
        source_reports = [
            waxal_report | {"admission": "reuse_verified_full_audit"},
            bible_report | {"admission": "fresh_full_audit_pending_listening_review"},
            {
                "name": "navigation_dagbani",
                "status": "rejected_pending_language_verification",
                "domain": "navigation",
                "license": "cc-by-nc-4.0",
                "admission": "excluded",
                "reason": "Native listening indicated that samples do not sound like Dagbani.",
            },
        ]
        display(pd.DataFrame(source_reports)[["name", "status", "license", "admission"]])
        if bible_report["status"] != "available":
            raise RuntimeError(f"Bible source is not available: {bible_report.get('error')}")
        """
    ),
    markdown(
        r"""
        ## 5. Required Bible listening gate

        Run this cell, listen to all 20 clips, and compare each clip with its transcript.
        Sample multiple locations. If the speech is Dagbani and alignment is consistently
        correct, change `BIBLE_REVIEW_PASSED` to `True` and rerun this cell. Do not pass
        the gate merely because the dataset name says Dagbani.
        """
    ),
    code(
        r"""
        BIBLE_REVIEW_PASSED = False  # CHANGE TO True ONLY AFTER LISTENING
        SHOW_REVIEW_AUDIO = True

        if SHOW_REVIEW_AUDIO:
            review_stream = load_dataset(
                BIBLE.repo_id,
                bible_report.get("resolved_config"),
                split="train",
                streaming=True,
                token=cfg.hf_token,
                revision=bible_report.get("revision"),
            )
            audio_field = next((name for name in BIBLE.audio_fields if name in review_stream.column_names), None)
            text_field = next((name for name in BIBLE.text_fields if name in review_stream.column_names), None)
            if not audio_field or not text_field:
                raise RuntimeError(f"Bible fields changed: {review_stream.column_names}")
            review_stream = review_stream.cast_column(audio_field, Audio(decode=False))
            review_rows = list(
                review_stream.shuffle(seed=cfg.seed, buffer_size=1_000).take(cfg.bible_review_rows)
            )
            for index, row in enumerate(review_rows, 1):
                waveform, sample_rate = audio_payload(row[audio_field])
                print(f"\n[{index}/{len(review_rows)}] {row.get(text_field, '')}")
                display(NotebookAudio(waveform, rate=sample_rate))

        cfg.bible_review_passed = bool(BIBLE_REVIEW_PASSED)
        review_evidence = {
            "review_passed": cfg.bible_review_passed,
            "rows_listened": cfg.bible_review_rows if cfg.bible_review_passed else 0,
            "acknowledged_utc": pd.Timestamp.utcnow().isoformat(),
            "policy": "Self-reported language and alignment listening gate; not a formal multi-reviewer evaluation.",
        }
        (ARTIFACT_DIR / "bible_listening_review.json").write_text(
            json.dumps(review_evidence, indent=2), encoding="utf-8"
        )
        print(review_evidence)
        """
    ),
    markdown("## 6. Reuse the sealed WAXAL full audit at an immutable revision"),
    code(
        r"""
        api = HfApi(token=cfg.hf_token)
        waxal_audit_info = api.dataset_info(cfg.reuse_waxal_repo_id, revision=cfg.reuse_waxal_revision)
        waxal_audit_revision = waxal_audit_info.sha


        def download_waxal_artifact(filename: str) -> Path:
            return Path(hf_hub_download(
                repo_id=cfg.reuse_waxal_repo_id,
                repo_type="dataset",
                filename=f"phase1/{filename}",
                revision=waxal_audit_revision,
                token=cfg.hf_token,
                cache_dir=str(CACHE_DIR),
            ))


        old_readiness = json.loads(download_waxal_artifact("readiness.json").read_text(encoding="utf-8"))
        if old_readiness.get("audit_mode") != "full" or not old_readiness.get("release_ready"):
            raise RuntimeError("The reused WAXAL audit is not full and release-ready")
        if int(old_readiness.get("sealed_test_rows", 0)) != 1_838:
            raise RuntimeError("The reused WAXAL sealed test row count changed")

        waxal_df = pd.read_parquet(download_waxal_artifact("accepted_manifest.parquet"))
        waxal_df = waxal_df[waxal_df["source"].eq("waxal_dag_asr")].copy()
        if waxal_df.empty or not {"train", "validation", "test"}.issubset(set(waxal_df["split"])):
            raise RuntimeError("Reused manifest does not contain all official WAXAL splits")
        accepted_rows = waxal_df.to_dict("records")
        rejected_rows: list[dict[str, Any]] = []
        near_hash_rows: list[dict[str, Any]] = []
        try:
            old_near = pd.read_parquet(download_waxal_artifact("near_audio_duplicates.parquet"))
            if {"sample_id", "source", "near_hash"}.issubset(old_near.columns):
                near_hash_rows.extend(old_near[["sample_id", "source", "near_hash"]].to_dict("records"))
        except Exception as exc:
            print(f"Prior WAXAL near-duplicate report unavailable: {exc}")

        reuse_evidence = {
            "repo_id": cfg.reuse_waxal_repo_id,
            "revision": waxal_audit_revision,
            "rows": len(waxal_df),
            "sealed_test_rows": int(waxal_df["split"].eq("test").sum()),
            "readiness": old_readiness,
        }
        (ARTIFACT_DIR / "waxal_reuse_evidence.json").write_text(
            json.dumps(reuse_evidence, indent=2), encoding="utf-8"
        )
        print({key: value for key, value in reuse_evidence.items() if key != "readiness"})
        """
    ),
    markdown("## 7. Fully decode the Bible with bounded parallelism and resumable shards"),
    code(
        r"""
        if not cfg.bible_review_passed:
            raise RuntimeError(
                "Bible listening gate has not passed. Set BIBLE_REVIEW_PASSED=True only after listening."
            )


        def first_field(record: dict[str, Any], candidates: Iterable[str]) -> str | None:
            return next((name for name in candidates if name in record), None)


        def bible_locator(split: str, sample_id: str, value: Any) -> str:
            if isinstance(value, dict) and value.get("path"):
                return str(value["path"])
            if isinstance(value, (str, os.PathLike)):
                return str(value)
            return f"hf://{BIBLE.repo_id}/{bible_report.get('resolved_config') or 'default'}/{split}/{sample_id}"


        def canonical_bible_record(
            original_split: str, record: dict[str, Any], row_index: int
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
            text_field = first_field(record, BIBLE.text_fields)
            audio_field = first_field(record, BIBLE.audio_fields)
            id_field = first_field(record, BIBLE.id_fields)
            speaker_field = first_field(record, BIBLE.speaker_fields)
            sample_id = strict_text(record.get(id_field)) if id_field else f"{original_split}-{row_index:09d}"
            # Preserve the source string exactly. Only transcript_eval is normalized.
            raw_value = record.get(text_field) if text_field else ""
            transcript_raw = "" if raw_value is None else str(raw_value)
            speaker_raw = strict_text(record.get(speaker_field)) if speaker_field else "unknown"
            speaker_id = f"dagbani_bible:{speaker_raw}" if speaker_raw != "unknown" else "unknown"
            value = record.get(audio_field) if audio_field else None
            locator = bible_locator(original_split, sample_id, value)
            group = bible_group(record, locator)
            split_policy = "book_chapter_group"
            if not group:
                group = f"contiguous-block-{row_index // 500:06d}"
                split_policy = "contiguous_500_row_fallback"
            split = stable_split(f"dagbani_bible:{group}")
            base = {
                "sample_id": sample_id,
                "source": "dagbani_bible",
                "source_version": bible_report.get("revision") or "default",
                "audio_locator": locator,
                "transcript_raw": transcript_raw,
                "transcript_eval": normalize_eval(transcript_raw),
                "speaker_id": speaker_id,
                "speaker_id_raw": speaker_raw,
                "duration_s": np.nan,
                "domain": "bible_read",
                "split": split,
                "origin_split": original_split,
                "split_policy": split_policy,
                "license": "cc-by-nc-4.0",
                "audio_hash": "",
                "is_pseudo": False,
                "pseudo_confidence": np.nan,
            }
            if not audio_field:
                return None, base | {"reason": "audio_field_missing", "error": ""}, None
            if not transcript_raw.strip():
                return None, base | {"reason": "transcript_missing", "error": ""}, None
            unexpected = unexpected_characters(transcript_raw)
            if unexpected:
                base["unexpected_characters"] = "".join(unexpected)
            try:
                array, sample_rate = audio_payload(value)
                array = canonical_audio(array, sample_rate)
                duration = len(array) / 16_000
                words_per_s = len(base["transcript_eval"].split()) / max(duration, 1e-6)
                base.update({
                    "duration_s": round(duration, 4),
                    "audio_hash": exact_audio_hash(array),
                    "words_per_s": words_per_s,
                })
                near_hash = spectral_fingerprint(array)
                if duration < cfg.min_duration_s or duration > cfg.max_duration_s:
                    return None, base | {"reason": "duration_out_of_range", "error": ""}, near_hash
                if not (cfg.min_words_per_s <= words_per_s <= cfg.max_words_per_s):
                    return None, base | {"reason": "speech_rate_out_of_range", "error": ""}, near_hash
                return base, None, near_hash
            except Exception as exc:
                return None, base | {
                    "reason": "audio_decode_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }, None


        def restore_resume_parts() -> None:
            if PERSIST_PART_DIR.resolve() == LOCAL_PART_DIR.resolve():
                return
            for source in PERSIST_PART_DIR.glob("bible-*.json.gz"):
                destination = LOCAL_PART_DIR / source.name
                if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                    shutil.copy2(source, destination)


        def read_resume_parts() -> tuple[int, list[dict], list[dict], list[dict]]:
            next_index = 0
            good_rows: list[dict] = []
            bad_rows: list[dict] = []
            fingerprints: list[dict] = []
            for path in sorted(LOCAL_PART_DIR.glob("bible-*.json.gz")):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if int(payload["start"]) != next_index:
                    print(f"Ignoring non-contiguous resume shard {path.name}")
                    break
                if payload.get("source_revision") != bible_report.get("revision"):
                    raise RuntimeError("Bible dataset revision changed; clear resume_parts before continuing")
                good_rows.extend(payload["accepted"])
                bad_rows.extend(payload["rejected"])
                fingerprints.extend(payload["near"])
                next_index = int(payload["end"])
            return next_index, good_rows, bad_rows, fingerprints


        def write_resume_part(start: int, end: int, good: list[dict], bad: list[dict], near: list[dict]) -> None:
            name = f"bible-{start:09d}-{end:09d}.json.gz"
            final_path = LOCAL_PART_DIR / name
            temporary = LOCAL_PART_DIR / f"{name}.tmp"
            payload = {
                "start": start,
                "end": end,
                "source_revision": bible_report.get("revision"),
                "accepted": good,
                "rejected": bad,
                "near": near,
            }
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
            temporary.replace(final_path)
            if PERSIST_PART_DIR.resolve() != LOCAL_PART_DIR.resolve():
                shutil.copy2(final_path, PERSIST_PART_DIR / name)


        restore_resume_parts()
        resume_at, bible_good, bible_bad, bible_near = read_resume_parts()
        print(f"Bible resume position: {resume_at:,} rows")

        bible_stream = load_dataset(
            BIBLE.repo_id,
            bible_report.get("resolved_config"),
            split="train",
            streaming=True,
            token=cfg.hf_token,
            revision=bible_report.get("revision"),
        )
        bible_audio_field = next((name for name in BIBLE.audio_fields if name in bible_stream.column_names), None)
        if not bible_audio_field:
            raise RuntimeError(f"Bible audio field missing: {bible_stream.column_names}")
        bible_stream = bible_stream.cast_column(bible_audio_field, Audio(decode=False))
        if resume_at:
            bible_stream = bible_stream.skip(resume_at)

        reported_total = ((bible_report.get("splits") or {}).get("train") or {}).get("num_examples")
        scan_started = time.time()
        buffer: list[tuple[int, dict[str, Any]]] = []


        def process_buffer(rows: list[tuple[int, dict[str, Any]]]) -> tuple[list[dict], list[dict], list[dict]]:
            def process(item: tuple[int, dict[str, Any]]):
                index, record = item
                return canonical_bible_record("train", record, index)

            good: list[dict] = []
            bad: list[dict] = []
            near: list[dict] = []
            with ThreadPoolExecutor(max_workers=cfg.decode_workers) as executor:
                for (index, _), (accepted, rejected, fingerprint) in zip(rows, executor.map(process, rows)):
                    if accepted:
                        good.append(accepted)
                    if rejected:
                        bad.append(rejected)
                    if fingerprint:
                        near.append({
                            "sample_id": (accepted or rejected)["sample_id"],
                            "source": "dagbani_bible",
                            "near_hash": fingerprint,
                        })
            return good, bad, near


        def flush_buffer(rows: list[tuple[int, dict[str, Any]]]) -> None:
            if not rows:
                return
            start = rows[0][0]
            end = rows[-1][0] + 1
            good, bad, near = process_buffer(rows)
            write_resume_part(start, end, good, bad, near)
            bible_good.extend(good)
            bible_bad.extend(bad)
            bible_near.extend(near)
            completed = end
            elapsed = max(time.time() - scan_started, 1e-6)
            fresh_rows = completed - resume_at
            rate = fresh_rows / elapsed
            eta = (reported_total - completed) / rate / 60 if reported_total and rate > 0 else None
            print(
                f"Bible: {completed:,}/{reported_total or '?'} rows; accepted={len(bible_good):,}; "
                f"rejected={len(bible_bad):,}; {rate:.2f} fresh rows/s; ETA={eta:.1f}m" if eta is not None
                else f"Bible: {completed:,}/? rows; accepted={len(bible_good):,}; rejected={len(bible_bad):,}"
            )
            gc.collect()


        for index, record in enumerate(bible_stream, start=resume_at):
            buffer.append((index, record))
            if len(buffer) >= cfg.shard_rows:
                flush_buffer(buffer)
                buffer = []
        flush_buffer(buffer)

        accepted_rows.extend(bible_good)
        rejected_rows.extend(bible_bad)
        near_hash_rows.extend(bible_near)
        print({
            "waxal_reused": len(waxal_df),
            "bible_accepted": len(bible_good),
            "bible_rejected": len(bible_bad),
            "combined_before_dedup": len(accepted_rows),
        })
        """
    ),
    markdown("## 8. Combined duplicate, transcript, and split-leakage audit"),
    code(
        r"""
        accepted_df = pd.DataFrame(accepted_rows)
        rejected_df = pd.DataFrame(
            rejected_rows,
            columns=list(dict.fromkeys(CANONICAL_COLUMNS + ["reason", "error", "words_per_s"])),
        )
        near_df = pd.DataFrame(near_hash_rows, columns=["sample_id", "source", "near_hash"])
        for column in CANONICAL_COLUMNS:
            if column not in accepted_df:
                accepted_df[column] = pd.Series(dtype="object")
        accepted_df = accepted_df[
            CANONICAL_COLUMNS + [column for column in accepted_df.columns if column not in CANONICAL_COLUMNS]
        ]

        SPLIT_KEEP_PRIORITY = {"test": 4, "external_test": 3, "validation": 2, "train": 1}


        def quarantine_cross_split_audio_duplicates(table: pd.DataFrame):
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
                for row_index, row in group[group["split"] != keep_split].iterrows():
                    drop_indexes.append(row_index)
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
            repaired = table.drop(index=drop_indexes).reset_index(drop=True)
            return repaired, quarantined.reset_index(drop=True), pd.DataFrame(decisions)


        accepted_df, quarantined_df, dedup_decisions_df = quarantine_cross_split_audio_duplicates(accepted_df)
        if not quarantined_df.empty:
            rejected_df = pd.concat([rejected_df, quarantined_df], ignore_index=True, sort=False)
            dropped_ids = set(quarantined_df["sample_id"].astype(str))
            near_df = near_df[~near_df["sample_id"].astype(str).isin(dropped_ids)].reset_index(drop=True)
            print(f"Quarantined {len(quarantined_df):,} lower-priority cross-split exact copies")


        def official_waxal_speaker_overlap(group: pd.DataFrame) -> bool:
            return bool(
                group["source"].eq("waxal_dag_asr").all()
                and set(group["split"].unique()).issubset({"train", "validation", "test"})
                and ("split_policy" not in group or group["split_policy"].eq("official").all())
            )


        findings: list[dict[str, Any]] = []
        supervised = accepted_df[accepted_df["split"].isin(["train", "validation", "test", "external_test"])]
        for key in ("speaker_id", "audio_hash"):
            clean = supervised[supervised[key].fillna("").astype(str).ne("")]
            if key == "speaker_id":
                clean = clean[clean[key] != "unknown"]
            for value, group in clean.groupby(key):
                splits = sorted(group["split"].unique())
                if len(splits) > 1:
                    official = key == "speaker_id" and official_waxal_speaker_overlap(group)
                    findings.append({
                        "kind": "waxal_official_topic_split_speaker_overlap" if official else f"{key}_overlap",
                        "value": value,
                        "sources": ",".join(sorted(group["source"].astype(str).unique())),
                        "splits": ",".join(splits),
                        "rows": len(group),
                        "severity": "warning" if official else "error",
                        "blocking": not official,
                        "note": "Published WAXAL topic split" if official else "Resolve before training",
                    })
        leakage_df = pd.DataFrame(
            findings,
            columns=["kind", "value", "sources", "splits", "rows", "severity", "blocking", "note"],
        )
        blocking_leakage_df = leakage_df[leakage_df["blocking"].fillna(True)] if not leakage_df.empty else leakage_df
        waxal_protocol_overlap_df = (
            leakage_df[leakage_df["kind"] == "waxal_official_topic_split_speaker_overlap"]
            if not leakage_df.empty else leakage_df
        )
        exact_duplicates = accepted_df[
            accepted_df["audio_hash"].fillna("").astype(str).ne("")
        ].groupby("audio_hash").filter(lambda group: len(group) > 1)
        transcript_duplicates = accepted_df[
            accepted_df["transcript_eval"].astype(str).ne("")
        ].groupby("transcript_eval").filter(lambda group: len(group) > 1)
        near_duplicates = (
            near_df.groupby("near_hash").filter(lambda group: len(group) > 1)
            if not near_df.empty else near_df
        )

        inventory_rows = []
        for (source, split), group in accepted_df.groupby(["source", "split"], dropna=False):
            durations = pd.to_numeric(group["duration_s"], errors="coerce")
            inventory_rows.append({
                "source": source,
                "split": split,
                "rows_scanned": len(group),
                "decoded_rows": int(durations.notna().sum()),
                "decoded_hours": float(durations.sum() / 3600),
                "speakers": int(group["speaker_id"].mask(group["speaker_id"].eq("unknown")).nunique()),
                "transcript_coverage": float(group["transcript_raw"].astype(bool).mean()),
                "license_values": ",".join(sorted(group["license"].dropna().astype(str).unique())),
            })
        inventory_df = pd.DataFrame(inventory_rows)
        display(inventory_df)
        print({
            "accepted": len(accepted_df),
            "rejected": len(rejected_df),
            "exact_duplicate_rows": len(exact_duplicates),
            "near_duplicate_rows": len(near_duplicates),
            "repeated_transcript_rows": len(transcript_duplicates),
            "blocking_split_leaks": len(blocking_leakage_df),
            "waxal_protocol_speaker_overlaps": len(waxal_protocol_overlap_df),
        })
        if not rejected_df.empty and "reason" in rejected_df:
            print(rejected_df["reason"].value_counts(dropna=False).to_string())
        """
    ),
    markdown("## 9. Export, verify, and publish the private combined manifest"),
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

        supervised = accepted_df[accepted_df["split"].isin(["train", "validation", "test", "external_test"])]
        waxal_supervised = supervised[supervised["source"].eq("waxal_dag_asr")]
        waxal_required_splits_present = {"train", "validation", "test"}.issubset(set(waxal_supervised["split"]))
        bible_rows_present = accepted_df["source"].eq("dagbani_bible").any()
        release_ready = bool(
            cfg.audit_mode == "full"
            and cfg.bible_review_passed
            and waxal_required_splits_present
            and bible_rows_present
            and blocking_leakage_df.empty
            and supervised["duration_s"].notna().all()
            and supervised["audio_hash"].astype(bool).all()
        )
        readiness = {
            "audit_mode": cfg.audit_mode,
            "release_ready": release_ready,
            "sealed_test_rows": int(
                (accepted_df["source"].eq("waxal_dag_asr") & accepted_df["split"].eq("test")).sum()
            ),
            "waxal_required_splits_present": waxal_required_splits_present,
            "bible_rows_present": bool(bible_rows_present),
            "bible_review_passed": cfg.bible_review_passed,
            "enabled_sources": list(cfg.enabled_sources),
            "navigation_status": "rejected_pending_language_verification",
            "waxal_reuse_revision": waxal_audit_revision,
            "bible_revision": bible_report.get("revision"),
            "warnings": [
                "WAXAL official topic-level splits reuse speakers; metrics are not speaker-independent.",
                "Bible is CC-BY-NC-4.0; keep this mixed branch non-commercial.",
                "Cross-source near-duplicate coverage is limited to retained prior WAXAL findings plus the fresh Bible scan; exact audio hashes are checked across every decoded supervised row.",
            ],
        }
        (ARTIFACT_DIR / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        environment = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "audit_config": asdict(cfg) | {"hf_token": "<redacted>"},
            "source_revisions": {
                "waxal_audit": waxal_audit_revision,
                "bible_dataset": bible_report.get("revision"),
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("datasets", "huggingface_hub", "librosa", "soundfile", "pandas", "pyarrow")
                if importlib.util.find_spec(name)
            },
        }
        (ARTIFACT_DIR / "environment_manifest.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        print(json.dumps(readiness, indent=2))

        if cfg.publish_artifacts:
            if not release_ready:
                raise RuntimeError("Audit is not release-ready; artifacts were not published")
            api.create_repo(cfg.manifest_repo_id, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(
                repo_id=cfg.manifest_repo_id,
                repo_type="dataset",
                folder_path=str(ARTIFACT_DIR),
                path_in_repo="phase1",
                commit_message=(
                    f"Full WAXAL+Bible audit; WAXAL base {waxal_audit_revision[:12]}; "
                    f"Bible {str(bible_report.get('revision'))[:12]}"
                ),
            )
            print(f"Published privately to https://huggingface.co/datasets/{cfg.manifest_repo_id}")
        else:
            print(f"Publishing disabled; artifacts remain in {ARTIFACT_DIR}")

        archive_path = shutil.make_archive(str(WORK_DIR / "dagbani_asr_phase1_colab_artifacts"), "zip", ARTIFACT_DIR)
        print(f"Downloadable archive: {archive_path}")
        """
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "cells": CELLS,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "none",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
