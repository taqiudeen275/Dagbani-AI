"""Contract tests for the generated Dagbani ASR Kaggle notebooks."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scratch" / "build_dagbani_asr_recovery_notebooks.py"
PHASE1 = ROOT / "notebooks" / "01_Dagbani_ASR_Data_Audit_and_Baselines_Kaggle.ipynb"
PHASE2 = ROOT / "notebooks" / "02_Dagbani_ASR_Whisper_Small_Training_Kaggle.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_source(document: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )


class NotebookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_notebook(PHASE1)
        cls.phase2 = load_notebook(PHASE2)
        cls.phase1_code = code_source(cls.phase1)
        cls.phase2_code = code_source(cls.phase2)

    def test_notebooks_are_valid_unexecuted_v4_documents(self) -> None:
        for document in (self.phase1, self.phase2):
            self.assertEqual(document["nbformat"], 4)
            self.assertGreater(len(document["cells"]), 10)
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
        self.assertEqual(self.phase1["metadata"]["kaggle"]["accelerator"], "none")
        self.assertEqual(self.phase2["metadata"]["kaggle"]["accelerator"], "gpu")

    def test_every_code_cell_compiles_independently(self) -> None:
        for path, document in ((PHASE1, self.phase1), (PHASE2, self.phase2)):
            for index, cell in enumerate(document["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")

    def test_phase1_has_all_sources_and_canonical_contract(self) -> None:
        for token in (
            "google/WaxalNLP",
            "ghananlpcommunity/dagbani-bible-audio-text-tts",
            "common_voice_25_dag",
            "spell4wiki_commons",
            "audio_hash",
            "pseudo_confidence",
            "split_leakage_report",
        ):
            self.assertIn(token, self.phase1_code)
        self.assertIn("def split_info_value", self.phase1_code)
        self.assertIn("isinstance(value, dict)", self.phase1_code)
        self.assertNotIn("value.num_examples", self.phase1_code)
        self.assertNotIn("value.num_bytes", self.phase1_code)
        self.assertIn("waxal_official_topic_split_speaker_overlap", self.phase1_code)
        self.assertIn("blocking_leakage_df", self.phase1_code)
        self.assertIn('f"{spec.name}:{speaker}"', self.phase1_code)
        self.assertIn("enabled_sources", self.phase1_code)
        self.assertIn("rows/s, ETA", self.phase1_code)
        self.assertIn("waxal_required_splits_present", self.phase1_code)

    def test_phase1_has_no_fake_data_or_random_split(self) -> None:
        self.assertNotIn("gen_synth", self.phase1_code)
        self.assertNotIn("np.sin(", self.phase1_code)
        self.assertNotIn("train_test_split", self.phase1_code)

    def test_phase2_defaults_to_safe_smoke_from_whisper_small(self) -> None:
        self.assertIn('stage: str = "smoke"', self.phase2_code)
        self.assertIn('base_model_id: str = "openai/whisper-small"', self.phase2_code)
        self.assertIn("supervised_max_steps: int = 2_500", self.phase2_code)
        self.assertIn("run_training: bool = False", self.phase2_code)
        self.assertIn("waxal_batch_share: float = 0.60", self.phase2_code)

    def test_phase2_preserves_whisper_tokenizer_and_real_ddp(self) -> None:
        self.assertNotIn('device_map="auto"', self.phase2_code)
        self.assertNotIn("suppress_tokens = []", self.phase2_code)
        self.assertNotIn("from tokenizers", self.phase2_code)
        self.assertIn("notebook_launcher", self.phase2_code)
        self.assertIn("forced_decoder_ids = None", self.phase2_code)
        self.assertIn("official_waxal_overlap", self.phase2_code)
        self.assertIn("Cross-split audio_hash leakage remains", self.phase2_code)

    def test_phase2_contains_budget_resume_and_pseudo_label_gates(self) -> None:
        for token in (
            "BudgetTelemetryCallback",
            "PrivateHubCheckpointCallback",
            "projected_total_min",
            "pseudo_initial_hours: float = 20.0",
            "calibrate_pseudo_thresholds",
            "bootstrap_wer_difference",
            "medium_gate",
        ):
            self.assertIn(token, self.phase2_code)

    def test_generated_files_match_generator(self) -> None:
        spec = importlib.util.spec_from_file_location("dagbani_notebook_builder", GENERATOR)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        expected1 = module.notebook(module.PHASE1_CELLS, "Dagbani ASR Phase 1 — Audit and Baselines")
        expected2 = module.notebook(module.PHASE2_CELLS, "Dagbani ASR Phase 2 — Budget-Aware Training", accelerator="gpu")
        self.assertEqual(self.phase1, expected1)
        self.assertEqual(self.phase2, expected2)


if __name__ == "__main__":
    unittest.main()
