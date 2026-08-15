# Dagbani ASR recovery notebooks

Use the two numbered notebooks instead of the older combined/Colab notebooks.

## Kaggle setup

1. Add a private Kaggle secret named `HF_TOKEN` with write access to your private
   Hugging Face repositories.
2. Set `DAGBANI_MANIFEST_REPO` to a private dataset repository such as
   `your-name/dagbani-asr-audit`.
3. Set `DAGBANI_MODEL_REPO` to a separate private model repository such as
   `your-name/dagbani-whisper-small`.
4. Accept any gated dataset terms yourself. For Common Voice, attach the extracted
   Dagbani package as a Kaggle input and set `COMMON_VOICE_DAG_ROOT` to its directory.

## Phase 1 — audit before using a GPU

Open `01_Dagbani_ASR_Data_Audit_and_Baselines_Kaggle.ipynb` with the accelerator
off. Keep `audit_mode="quick"` for the first pass. Review `source_status`, rejected
samples, corpus counts, licenses, and split leakage.

After resolving failures, run the first full audit on WAXAL only. This is enough
for the supervised milestone and avoids decoding the much larger Bible corpus
before it is needed for domain adaptation:

```python
cfg = AuditConfig(
    audit_mode="full",
    enabled_sources=("waxal_dag_asr",),
    manifest_repo_id="YOUR_HF_USERNAME/dagbani-asr-phase1",
    publish_artifacts=True,
)
cfg.validate()
```

The full pass prints progress and ETA every 250 rows and decodes/hashes admitted
supervised audio. Phase 2 will refuse a full training stage unless
`readiness.json` says `release_ready: true`. Audit Bible and other labeled sources
in a later Phase 1 run before domain-adaptation training.

The baseline cell is optional and disabled. Enable a GPU only for that cell, then
turn it off again after predictions are saved.

## Phase 2 — one stage at a time

Open `02_Dagbani_ASR_Whisper_Small_Training_Kaggle.ipynb` with a GPU. Run stages in
this order:

1. `smoke`: 20 optimizer steps on 64 clips.
2. `supervised`: up to 2,500 steps from `openai/whisper-small` on audited WAXAL.
3. `domain_adaptation`: continue from the private supervised repository and mix
   admitted labeled domains while retaining at least 60% WAXAL batches.
4. `pseudo_label`: first generate at most 20 confidence-filtered hours from WAXAL's
   unlabeled split, then train with a maximum 25% pseudo-labelled batch share.

`run_training`, `run_full_generation_eval`, and `run_pseudo_labelling` are all false
by default. Generate pseudo labels and pseudo-label training are separate runs; the
notebook rejects enabling `run_training` and `run_pseudo_labelling` together. Turn on
only the action you intend to run. Training logs a 20-step ETA,
stops before the wall-clock budget, and uploads resumable state every 250 steps.

Whisper Medium remains locked behind the `medium_gate` checks. Run the matched
`medium_pilot`, call `medium_gate(...)` with the measured evidence, and persist a
passing result with `save_medium_gate(...)`; only then will `medium_full` validate.
A model that passes
the numeric checks is still research-only until native speakers review transcripts
and listen to representative outputs.

## Regeneration and validation

The notebooks are generated from `scratch/build_dagbani_asr_recovery_notebooks.py`.
After modifying it, regenerate and run:

```powershell
python scratch/build_dagbani_asr_recovery_notebooks.py
python -m unittest tests/test_dagbani_asr_notebooks.py
```
