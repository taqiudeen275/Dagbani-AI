"""Contract tests for the generated Dagbani ASR Kaggle notebooks."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scratch" / "build_dagbani_asr_recovery_notebooks.py"
PHASE1 = ROOT / "notebooks" / "01_Dagbani_ASR_Data_Audit_and_Baselines_Kaggle.ipynb"
BASELINE = ROOT / "notebooks" / "01b_Dagbani_ASR_Baseline_Only_Kaggle.ipynb"
PHASE2 = ROOT / "notebooks" / "02_Dagbani_ASR_Whisper_Small_Training_Kaggle.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_source(document: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    )


def without_cell_ids(document: dict) -> dict:
    """Jupyter frontends may add valid, non-semantic nbformat 4.5 cell IDs."""
    normalized = json.loads(json.dumps(document))
    for cell in normalized["cells"]:
        cell.pop("id", None)
    return normalized


class NotebookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_notebook(PHASE1)
        cls.baseline = load_notebook(BASELINE)
        cls.phase2 = load_notebook(PHASE2)
        cls.phase1_code = code_source(cls.phase1)
        cls.baseline_code = code_source(cls.baseline)
        cls.phase2_code = code_source(cls.phase2)

    def test_notebooks_are_valid_unexecuted_v4_documents(self) -> None:
        for document in (self.phase1, self.baseline, self.phase2):
            self.assertEqual(document["nbformat"], 4)
            self.assertGreater(len(document["cells"]), 10)
            for cell in document["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
        self.assertEqual(self.phase1["metadata"]["kaggle"]["accelerator"], "none")
        self.assertEqual(self.baseline["metadata"]["kaggle"]["accelerator"], "gpu")
        self.assertEqual(self.phase2["metadata"]["kaggle"]["accelerator"], "gpu")

    def test_every_code_cell_compiles_independently(self) -> None:
        for path, document in ((PHASE1, self.phase1), (BASELINE, self.baseline), (PHASE2, self.phase2)):
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
        self.assertIn("Blocking cross-split overlaps", self.phase1_code)
        self.assertIn("Rejected-row reasons", self.phase1_code)
        self.assertIn("quarantine_cross_split_audio_duplicates", self.phase1_code)
        self.assertIn("cross_split_exact_audio_duplicate", self.phase1_code)
        self.assertIn('"test": 4', self.phase1_code)

    def test_phase1_has_no_fake_data_or_random_split(self) -> None:
        self.assertNotIn("gen_synth", self.phase1_code)
        self.assertNotIn("np.sin(", self.phase1_code)
        self.assertNotIn("train_test_split", self.phase1_code)

    def test_baseline_only_is_frozen_batched_and_resumable(self) -> None:
        for token in (
            'run_baselines: bool = False',
            'max_samples: int | None = 256',
            'waxal-benchmarking/whisper-small-waxal-dag',
            'openai/whisper-small',
            'batch_size: int = 8',
            'max_words_per_s: float = 4.0',
            'protocol_id: str = "waxal-clean-v2-max4wps"',
            'eval_manifest_hash',
            'resume_safe',
            'forced_decoder_ids = None',
            'AutoModelForSpeechSeq2Seq',
            'AutoProcessor',
            '"task": "transcribe"',
            'torch.cuda.get_arch_list()',
            'Select T4 x2 instead of P100',
            '.le(cfg.max_words_per_s)',
        ):
            self.assertIn(token, self.baseline_code)
        self.assertNotIn('from transformers import pipeline', self.baseline_code)
        self.assertNotIn('.ge(cfg.min_words_per_s)', self.baseline_code)
        self.assertNotIn('device_map="auto"', self.baseline_code)
        self.assertNotIn('suppress_tokens = []', self.baseline_code)

    def test_phase2_defaults_to_safe_smoke_from_whisper_small(self) -> None:
        self.assertIn('stage: str = "smoke"', self.phase2_code)
        self.assertIn('base_model_id: str = "openai/whisper-small"', self.phase2_code)
        self.assertIn('use_ddp: str = "one"', self.phase2_code)
        self.assertIn("supervised_max_steps: int = 2_500", self.phase2_code)
        self.assertIn("run_training: bool = False", self.phase2_code)
        self.assertIn("waxal_batch_share: float = 0.60", self.phase2_code)

    def test_phase2_preserves_whisper_tokenizer_and_real_ddp(self) -> None:
        self.assertNotIn('device_map="auto"', self.phase2_code)
        self.assertNotIn("suppress_tokens = []", self.phase2_code)
        self.assertNotIn("from tokenizers", self.phase2_code)
        self.assertIn("notebook_launcher", self.phase2_code)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = "0"', self.phase2_code)
        self.assertIn('os.environ["DAGBANI_MODEL_SNAPSHOT"]', self.phase2_code)
        self.assertIn('os.environ["ACCELERATE_DEBUG_MODE"] = "yes"', self.phase2_code)
        self.assertIn("Two-GPU notebook DDP is unsafe", self.phase2_code)
        self.assertIn("torch.cuda.is_initialized()", self.phase2_code)
        self.assertIn("DDP smoke collator gate deferred to each launched rank.", self.phase2_code)
        self.assertIn("deferred collator gate passed", self.phase2_code)
        self.assertIn("def load_waxal(split: str) -> IterableDataset", self.phase2_code)
        self.assertIn('split=split, streaming=True', self.phase2_code)
        self.assertIn("stream_shuffle_buffer: int = 512", self.phase2_code)
        self.assertIn("disk_stop_free_gib: float = 8.0", self.phase2_code)
        self.assertIn("ignore_data_skip=True", self.phase2_code)
        self.assertIn('"waxal_supervised": "streaming"', self.phase2_code)
        self.assertIn("load_dtype = torch.float32", self.phase2_code)
        self.assertIn("AMP requires FP32 trainable weights", self.phase2_code)
        self.assertIn("auto_smoke_generation", self.phase2_code)
        self.assertIn("generation_limit = 8", self.phase2_code)
        self.assertIn("return_attention_mask=True", self.phase2_code)
        self.assertIn("attention_mask=attention_mask", self.phase2_code)
        self.assertIn("trainer.accelerator.wait_for_everyone()", self.phase2_code)
        self.assertIn("if trainer.is_world_process_zero():", self.phase2_code)
        self.assertIn("max_length=225", self.phase2_code)
        self.assertIn("forced_decoder_ids = None", self.phase2_code)
        self.assertIn("official_waxal_overlap", self.phase2_code)
        self.assertIn("Cross-split audio_hash leakage remains", self.phase2_code)
        self.assertIn("add_special_tokens=False", self.phase2_code)
        self.assertIn("clean_up_tokenization_spaces=False", self.phase2_code)
        self.assertIn("def canonical_training_audio", self.phase2_code)
        self.assertIn("prepare_smoke_data", self.phase2_code)
        self.assertIn('"audio_contract": "contiguous mono float32 at 16 kHz"', self.phase2_code)
        self.assertIn('convert_tokens_to_ids("<|startoftranscript|>")', self.phase2_code)
        self.assertIn(
            "data_collator=SpeechSeq2SeqCollator(processor, decoder_start_token_id)",
            self.phase2_code,
        )
        self.assertNotIn("data_collator=SpeechSeq2SeqCollator(processor),", self.phase2_code)
        self.assertNotIn("self.processor.tokenizer.bos_token_id", self.phase2_code)
        self.assertNotIn('Dataset.from_list(materialized).cast_column("audio"', self.phase2_code)

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
        expected_baseline = module.notebook(module.BASELINE_CELLS, "Dagbani ASR Phase 1B — Baseline Only", accelerator="gpu")
        expected2 = module.notebook(module.PHASE2_CELLS, "Dagbani ASR Phase 2 — Budget-Aware Training", accelerator="gpu")
        self.assertEqual(without_cell_ids(self.phase1), without_cell_ids(expected1))
        self.assertEqual(without_cell_ids(self.baseline), without_cell_ids(expected_baseline))
        self.assertEqual(without_cell_ids(self.phase2), without_cell_ids(expected2))


if __name__ == "__main__":
    unittest.main()
