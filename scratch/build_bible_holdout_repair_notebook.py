"""Build the CPU-only Bible holdout repair notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01d_Dagbani_ASR_Bible_Holdout_Repair_Kaggle.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
        # Dagbani ASR Phase 1D — Bible Holdout Repair

        This CPU-only notebook derives a new immutable manifest repository from the
        release-ready Phase 1 v2 artifacts. It fixes the discovered fact that all
        53,410 Bible rows were marked `train` by creating contiguous 80/10/10
        train/validation/external-test spans. WAXAL rows and its sealed test remain
        unchanged. No audio or model weights are downloaded.

        The earlier domain pilot trained on all Bible rows and is therefore not
        eligible for Bible model selection. Use the new v3 manifest for a fresh
        pilot. Bible `validation` is for selection; `external_test` remains sealed.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import hashlib
        import json
        import re

        import numpy as np
        import pandas as pd
        from huggingface_hub import HfApi, snapshot_download


        def kaggle_secret(name: str) -> str | None:
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return None


        SOURCE_REPO = "ats-tech/dagbani-asr-phase1-v2"
        TARGET_REPO = "ats-tech/dagbani-asr-phase1-v3"
        HF_TOKEN = kaggle_secret("HF_TOKEN")
        WORK_DIR = Path("/kaggle/working/dagbani_bible_holdout_repair")
        DOWNLOAD_DIR = WORK_DIR / "source"
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        if not HF_TOKEN:
            raise RuntimeError("Add a writable HF_TOKEN Kaggle secret")
        if SOURCE_REPO == TARGET_REPO:
            raise RuntimeError("The repaired manifest must use a new repository")

        api = HfApi(token=HF_TOKEN)
        source_info = api.repo_info(SOURCE_REPO, repo_type="dataset")
        snapshot_root = Path(snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            revision=source_info.sha,
            allow_patterns=["phase1/*"],
            local_dir=str(DOWNLOAD_DIR),
        ))
        ARTIFACT_DIR = snapshot_root / "phase1"
        print({"source_repo": SOURCE_REPO, "source_revision": source_info.sha, "artifact_dir": str(ARTIFACT_DIR)})
        """
    ),
    code(
        r"""
        def read_table(stem: str) -> pd.DataFrame:
            path = ARTIFACT_DIR / f"{stem}.parquet"
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()


        manifest = read_table("accepted_manifest")
        rejected = read_table("rejected_samples")
        existing_decisions = read_table("dedup_decisions")
        near_duplicates = read_table("near_audio_duplicates")
        readiness = json.loads((ARTIFACT_DIR / "readiness.json").read_text(encoding="utf-8"))

        if not readiness.get("release_ready") or readiness.get("audit_mode") != "full":
            raise RuntimeError(f"Source manifest is not a release-ready full audit: {readiness}")

        waxal_before = manifest.loc[
            manifest["source"].eq("waxal_dag_asr"),
            ["sample_id", "split", "audio_hash"],
        ].sort_values("sample_id").reset_index(drop=True)

        bible_mask = manifest["source"].eq("dagbani_bible")
        bible = manifest.loc[bible_mask].copy()
        if len(bible) != 53_410:
            raise RuntimeError(f"Expected 53,410 Bible rows from v2, received {len(bible):,}")
        if set(bible["split"].astype(str)) != {"train"}:
            raise RuntimeError(f"Bible is not the expected all-train v2 manifest: {bible['split'].value_counts().to_dict()}")

        known_speakers = bible[
            bible["speaker_id"].fillna("").astype(str).ne("")
            & bible["speaker_id"].fillna("").astype(str).ne("unknown")
        ]
        if not known_speakers.empty:
            raise RuntimeError(
                "Bible exposes non-empty speaker IDs; stop and construct a speaker-disjoint split instead: "
                f"{known_speakers['speaker_id'].value_counts().head(10).to_dict()}"
            )

        row_numbers = bible["sample_id"].astype(str).str.extract(r"(\d+)$", expand=False)
        if row_numbers.isna().any():
            raise RuntimeError("A Bible sample ID does not end in the Phase 1 deterministic row index")
        bible["_row_number"] = row_numbers.astype("int64")
        if bible["_row_number"].duplicated().any():
            raise RuntimeError("Bible deterministic row indices are not unique")
        ordered_indexes = bible.sort_values("_row_number").index.to_numpy()

        train_end = int(len(ordered_indexes) * 0.80)
        validation_end = train_end + int(len(ordered_indexes) * 0.10)
        train_indexes = ordered_indexes[:train_end]
        validation_indexes = ordered_indexes[train_end:validation_end]
        external_test_indexes = ordered_indexes[validation_end:]

        manifest.loc[train_indexes, "split"] = "train"
        manifest.loc[validation_indexes, "split"] = "validation"
        manifest.loc[external_test_indexes, "split"] = "external_test"
        manifest.loc[bible_mask, "split_policy"] = "contiguous_span_80_10_10_no_speaker_metadata"

        assignments = manifest.loc[bible_mask, ["sample_id", "source", "split", "split_policy"]].copy()
        assignments["row_number"] = row_numbers.astype("int64").values
        assignments = assignments.sort_values("row_number").reset_index(drop=True)

        split_counts_before_dedup = assignments["split"].value_counts().to_dict()
        expected_counts = {"train": 42_728, "validation": 5_341, "external_test": 5_341}
        if split_counts_before_dedup != expected_counts:
            raise RuntimeError(f"Unexpected Bible split counts: {split_counts_before_dedup}")

        # Preserve held-out rows and quarantine any exact-audio copy that the new
        # split happens to place in a lower-priority split.
        priority = {"external_test": 4, "test": 4, "validation": 3, "train": 2}
        supervised_mask = manifest["split"].isin(priority)
        hashed = manifest[
            supervised_mask & manifest["audio_hash"].fillna("").astype(str).ne("")
        ]
        drop_indexes = []
        new_decisions = []
        for audio_hash, group in hashed.groupby("audio_hash"):
            splits = set(group["split"].astype(str))
            if len(splits) <= 1:
                continue
            keep_split = max(splits, key=lambda value: priority[value])
            kept_ids = sorted(group.loc[group["split"] == keep_split, "sample_id"].astype(str))
            for index, row in group[group["split"] != keep_split].iterrows():
                drop_indexes.append(index)
                new_decisions.append({
                    "audio_hash": str(audio_hash),
                    "dropped_sample_id": str(row["sample_id"]),
                    "dropped_split": str(row["split"]),
                    "kept_split": keep_split,
                    "kept_sample_ids": ",".join(kept_ids),
                    "reason": "cross_split_exact_audio_duplicate_after_bible_holdout",
                })

        decisions_df = pd.DataFrame(new_decisions)
        dropped = manifest.loc[drop_indexes].copy() if drop_indexes else manifest.iloc[0:0].copy()
        if not dropped.empty:
            dropped["reason"] = "cross_split_exact_audio_duplicate_after_bible_holdout"
            dropped["error"] = ""
            if list(rejected.columns) == ["_empty"]:
                rejected = pd.DataFrame()
            rejected = pd.concat([rejected, dropped], ignore_index=True, sort=False)
            dropped_ids = set(dropped["sample_id"].astype(str))
            manifest = manifest.drop(index=drop_indexes).reset_index(drop=True)
            if not near_duplicates.empty and "sample_id" in near_duplicates:
                near_duplicates = near_duplicates[
                    ~near_duplicates["sample_id"].astype(str).isin(dropped_ids)
                ]

        if list(existing_decisions.columns) == ["_empty"]:
            existing_decisions = pd.DataFrame()
        all_decisions = pd.concat([existing_decisions, decisions_df], ignore_index=True, sort=False)

        def official_waxal_speaker_overlap(group: pd.DataFrame) -> bool:
            return bool(
                group["source"].eq("waxal_dag_asr").all()
                and set(group["split"].astype(str)).issubset({"train", "validation", "test"})
                and ("split_policy" not in group or group["split_policy"].eq("official").all())
            )


        findings = []
        supervised = manifest[manifest["split"].isin(priority)]
        for key in ("speaker_id", "audio_hash"):
            clean = supervised[supervised[key].fillna("").astype(str).ne("")]
            if key == "speaker_id":
                clean = clean[clean[key].astype(str).ne("unknown")]
            for value, group in clean.groupby(key):
                splits = sorted(group["split"].astype(str).unique())
                if len(splits) <= 1:
                    continue
                official = key == "speaker_id" and official_waxal_speaker_overlap(group)
                findings.append({
                    "kind": "waxal_official_topic_split_speaker_overlap" if official else f"{key}_overlap",
                    "value": str(value),
                    "sources": ",".join(sorted(group["source"].astype(str).unique())),
                    "splits": ",".join(splits),
                    "rows": len(group),
                    "severity": "warning" if official else "error",
                    "blocking": not official,
                    "note": (
                        "Published WAXAL topic-level protocol; retained for benchmark comparability."
                        if official else "Must be resolved before training."
                    ),
                })

        leakage = pd.DataFrame(
            findings,
            columns=["kind", "value", "sources", "splits", "rows", "severity", "blocking", "note"],
        )
        blockers = leakage[leakage["blocking"].fillna(True)] if not leakage.empty else leakage
        if not blockers.empty:
            raise RuntimeError(f"Blocking leakage remains:\n{blockers.to_string(index=False)}")

        waxal_after = manifest.loc[
            manifest["source"].eq("waxal_dag_asr"),
            ["sample_id", "split", "audio_hash"],
        ].sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(waxal_before, waxal_after)

        supervised = manifest[manifest["split"].isin(priority)]
        if pd.to_numeric(supervised["duration_s"], errors="coerce").isna().any():
            raise RuntimeError("A supervised row is missing audited duration")
        if supervised["audio_hash"].fillna("").astype(str).eq("").any():
            raise RuntimeError("A supervised row is missing audited audio hash")
        if supervised["transcript_raw"].fillna("").astype(str).eq("").any():
            raise RuntimeError("A supervised row is missing transcript")

        inventory_rows = []
        for (source, split), group in manifest.groupby(["source", "split"], dropna=False):
            durations = pd.to_numeric(group["duration_s"], errors="coerce")
            speakers = group["speaker_id"].fillna("").astype(str)
            speakers = speakers[~speakers.isin(["", "unknown"])]
            inventory_rows.append({
                "source": source,
                "split": split,
                "rows_scanned": len(group),
                "decoded_rows": int(durations.notna().sum()),
                "decoded_hours": float(durations.sum() / 3600),
                "speakers": int(speakers.nunique()),
                "transcript_coverage": float(group["transcript_raw"].fillna("").astype(str).ne("").mean()),
                "license_values": ",".join(sorted(group["license"].dropna().astype(str).unique())),
            })
        inventory = pd.DataFrame(inventory_rows)

        hashed_all = manifest[manifest["audio_hash"].fillna("").astype(str).ne("")]
        exact_duplicates = hashed_all.groupby("audio_hash").filter(lambda group: len(group) > 1)
        text_rows = manifest[manifest["transcript_eval"].fillna("").astype(str).ne("")]
        repeated_transcripts = text_rows.groupby("transcript_eval").filter(lambda group: len(group) > 1)
        if not near_duplicates.empty and "near_hash" in near_duplicates:
            near_duplicates = near_duplicates.groupby("near_hash").filter(lambda group: len(group) > 1)

        readiness["release_ready"] = True
        readiness["sealed_test_rows"] = int(
            (manifest["source"].eq("waxal_dag_asr") & manifest["split"].eq("test")).sum()
        )
        readiness["warnings"] = [
            "WAXAL official topic-level splits reuse speakers. Official metrics remain benchmark-comparable but are not speaker-independent.",
            "Bible holdouts are contiguous content spans because the source exposes no usable speaker/book/chapter identifiers; they are not speaker-independent.",
        ]
        readiness["bible_split_repair"] = {
            "policy": "contiguous_span_80_10_10_no_speaker_metadata",
            "source_rows": 53_410,
            "counts_before_duplicate_quarantine": expected_counts,
            "cross_split_duplicates_quarantined": len(dropped),
            "validation_use": "model selection",
            "external_test_use": "sealed final evaluation only",
            "source_manifest_revision": source_info.sha,
        }

        tables = {
            "accepted_manifest": manifest,
            "rejected_samples": rejected,
            "dedup_decisions": all_decisions,
            "split_leakage_report": leakage,
            "exact_audio_duplicates": exact_duplicates,
            "near_audio_duplicates": near_duplicates,
            "repeated_transcripts": repeated_transcripts,
            "corpus_inventory": inventory,
            "bible_split_assignments": assignments,
        }
        for stem, table in tables.items():
            table.to_parquet(ARTIFACT_DIR / f"{stem}.parquet", index=False)
            table.to_csv(ARTIFACT_DIR / f"{stem}.csv", index=False)

        (ARTIFACT_DIR / "readiness.json").write_text(
            json.dumps(readiness, indent=2), encoding="utf-8"
        )
        metadata = {
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "source_repo": SOURCE_REPO,
            "source_revision": source_info.sha,
            "target_repo": TARGET_REPO,
            "bible_counts_after_quarantine": manifest.loc[
                manifest["source"].eq("dagbani_bible"), "split"
            ].value_counts().to_dict(),
            "accepted_rows": len(manifest),
            "blocking_split_leaks": len(blockers),
            "sealed_waxal_test_rows": readiness["sealed_test_rows"],
            "accepted_manifest_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(manifest, index=True).values.tobytes()
            ).hexdigest(),
        }
        (ARTIFACT_DIR / "bible_holdout_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(readiness, indent=2))
        print(json.dumps(metadata, indent=2))
        """
    ),
    code(
        r"""
        api.create_repo(TARGET_REPO, repo_type="dataset", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=TARGET_REPO,
            repo_type="dataset",
            folder_path=str(ARTIFACT_DIR),
            path_in_repo="phase1",
            commit_message="Create Bible validation and sealed external-test holdouts",
        )
        target_info = api.repo_info(TARGET_REPO, repo_type="dataset")
        print({
            "status": "published",
            "target_repo": TARGET_REPO,
            "target_revision": target_info.sha,
            "commit_url": commit.commit_url,
            "next_manifest_repo_id": TARGET_REPO,
        })
        """
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "none", "title": "Dagbani ASR Bible Holdout Repair"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
