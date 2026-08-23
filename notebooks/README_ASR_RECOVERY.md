# Dagbani ASR recovery notebooks

Use the numbered recovery notebooks instead of the older combined notebook. The
`01b` notebook is a small GPU-only baseline utility, while `01c` is a standalone
CPU-only Colab utility for the later combined WAXAL + Bible audit.

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

### Optimized Colab combined audit

For the WAXAL + Bible full audit, upload
`01c_Dagbani_ASR_Full_Audit_Colab_Optimized.ipynb` to Colab and select a standard
runtime with **Hardware accelerator: None**. Add `HF_TOKEN` through Colab's Secrets
panel. The notebook pins and reuses the release-ready WAXAL audit from
`ats-tech/dagbani-asr-phase1`, then fully decodes only the Bible corpus. It uses a
bounded thread pool and stores small resume shards under
`MyDrive/dagbani_asr_phase1_colab/`, so a runtime interruption does not require
re-decoding completed Bible shards.

The listening gate intentionally defaults to false. Listen to the 20 displayed
Bible samples, compare their transcripts, then change
`BIBLE_REVIEW_PASSED = True` only if the language and alignment are credible.
Navigation remains excluded pending independent native-speaker verification.
Successful artifacts are published to the separate private repository
`ats-tech/dagbani-asr-phase1-v2`. The mixed repository is non-commercial because
the Bible source is recorded as CC-BY-NC-4.0.

If the v2 inventory reports all 53,410 Bible rows as `train`, run the standalone
CPU notebook `01d_Dagbani_ASR_Bible_Holdout_Repair_Kaggle.ipynb`. It derives the
new private `ats-tech/dagbani-asr-phase1-v3` repository without downloading audio,
keeps WAXAL unchanged, and assigns contiguous Bible spans to 42,728 train, 5,341
validation, and 5,341 sealed `external_test` rows. Any pilot trained against the
all-train v2 Bible manifest is contaminated for Bible evaluation and must not be
promoted; rerun it from the frozen champion using v3.

After Phase 1 is release-ready, open
`01b_Dagbani_ASR_Baseline_Only_Kaggle.ipynb` as a standalone Kaggle notebook.
Select **GPU T4 x2**, not P100: Kaggle's current PyTorch build no longer includes
the P100's `sm_60` CUDA kernels. The baseline runner uses the first T4 and checks
GPU compatibility before downloading a model. Set
`run_baselines=True` for the default 256-row pilot. It reads the private manifest,
batches inference, checkpoints predictions to `phase1/baselines/`, and resumes
from those predictions after interruption. Use `max_samples=None` only after the
pilot succeeds to reproduce the public model on the full filtered test set. The
WAXAL cleaning rule retains clips at least 1.5 seconds long and at most 4 words per
second; the corrected protocol has its own repository path so the earlier inverted
two-row pilot cannot be resumed. Because the public model card currently states the
opposite speech-rate inequality, treat this as a transparent corrected comparison,
not an exact reproduction, until WAXAL publishes or confirms the cleaned row list.

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

Continuation stages use two deliberately separate model settings. Set
`starting_model_id` to the frozen parent (for example
`ats-tech/dagbani-whisper-small`) and `model_repo_id` to a new private output
repository. The notebook refuses to run when they are identical. For domain
adaptation, first run a fresh 100-step pilot into
`ats-tech/dagbani-whisper-small-domain-pilot`; do not resume that pilot into the
full schedule. If matched WAXAL generation evaluation does not regress, start a
fresh session from the same frozen parent and train the 750-step candidate into
`ats-tech/dagbani-whisper-small-domain-v1`.
When external generation evaluation is enabled for a continuation stage, the
notebook evaluates both the frozen parent and candidate on the same held-out rows
from every admitted external domain. A domain candidate is not accepted merely
because its training loss falls; it must show a measured domain benefit without a
material WAXAL conversational regression.
`external_eval_split="validation"` is the continuation default and is used for
pilot/model selection. Keep the domain `external_test` split sealed; change the
setting to `external_test` only once, after validation has selected a final
candidate.

The supervised WAXAL train and validation splits are remote streaming datasets.
They are never materialized into Kaggle's limited session disk. Training uses a
bounded shuffle buffer and positive `max_steps`; checkpoint resume restores the
model, optimizer, scheduler, and RNG state without replaying the remote stream.
The notebook also stops safely when free disk approaches 8 GiB.

Audited Hugging Face domain sources are streamed as well. When a source such as
the Bible corpus has no native sample-ID column, Phase 2 recreates Phase 1's
deterministic `<origin-split>-<nine-digit-row-index>` IDs before filtering to the
accepted manifest. A decoded one-row probe must pass before the source is admitted
to a domain run; the loader never materializes all 37 Bible Parquet shards.
WAXAL and admitted domain streams are combined by a PyTorch iterable with an exact,
shuffled 100-example schedule (60 WAXAL and 40 domain examples by default). This
avoids forcing lazy torchcodec `AudioDecoder` objects through PyArrow feature
inference and cycles finite streams without changing the configured source share.
Continuation stages deliberately use `dataloader_workers=0`, disable pinned host
memory, and use a 128-row per-source shuffle buffer. Kaggle worker processes can
otherwise duplicate remote buffers and torchcodec state until the operating system
kills a worker. Single-process loading can be slower but leaves the source schedule
and optimization results unchanged.

The smoke run defaults to `use_ddp="one"` and hides the second T4 so Trainer cannot
silently substitute DataParallel. This validates model loading, collation,
backpropagation, evaluation, and private persistence without notebook forking. In
the current Kaggle image, notebook-launched two-GPU DDP can fail because Accelerate
uses `fork` after the kernel has initialized CUDA. The notebook now detects that
state and fails before downloading or loading the model. Use `use_ddp="one"` for
Kaggle training; two GPUs require a future standalone spawn-based launcher.

Whisper Medium remains locked behind the `medium_gate` checks. Run the matched
`medium_pilot`, call `medium_gate(...)` with the measured evidence, and persist a
passing result with `save_medium_gate(...)`; only then will `medium_full` validate.
A model that passes
the numeric checks is still research-only until native speakers review transcripts
and listen to representative outputs.

## Regeneration and validation

The Kaggle notebooks are generated from
`scratch/build_dagbani_asr_recovery_notebooks.py`; the optimized Colab audit is
generated from `scratch/build_dagbani_asr_colab_audit.py`. After modifying them,
regenerate and run:

```powershell
python scratch/build_dagbani_asr_recovery_notebooks.py
python scratch/build_dagbani_asr_colab_audit.py
python -m unittest tests/test_dagbani_asr_notebooks.py
python -m unittest tests/test_dagbani_asr_colab_audit.py
```
