"""Contract tests for the optimized standalone Colab audit notebook."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01c_Dagbani_ASR_Full_Audit_Colab_Optimized.ipynb"


class ColabAuditNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n\n".join(
            "".join(cell["source"])
            for cell in cls.document["cells"]
            if cell["cell_type"] == "code"
        )

    def test_valid_unexecuted_notebook(self) -> None:
        self.assertEqual(self.document["nbformat"], 4)
        self.assertEqual(self.document["metadata"]["accelerator"], "none")
        for cell in self.document["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_every_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.document["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_scope_and_manual_gate_are_explicit(self) -> None:
        self.assertIn(
            'enabled_sources: tuple[str, ...] = ("waxal_dag_asr", "dagbani_bible")',
            self.code,
        )
        self.assertIn("BIBLE_REVIEW_PASSED = False", self.code)
        self.assertIn("rejected_pending_language_verification", self.code)
        self.assertNotIn(
            'enabled_sources=("waxal_dag_asr", "dagbani_bible", "navigation_dagbani")',
            self.code,
        )

    def test_speedups_preserve_resumability_and_audit_guards(self) -> None:
        self.assertIn(
            'reuse_waxal_repo_id: str = "ats-tech/dagbani-asr-phase1"',
            self.code,
        )
        self.assertIn("ThreadPoolExecutor", self.code)
        self.assertIn("write_resume_part", self.code)
        self.assertIn("waxal_audit_revision", self.code)
        self.assertIn("quarantine_cross_split_audio_duplicates", self.code)
        self.assertIn("if not release_ready:", self.code)


if __name__ == "__main__":
    unittest.main()
