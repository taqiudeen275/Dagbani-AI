# Dagbani TTS Prototype Roadmap

## Objective

Build a research-grade, single-voice Dagbani text-to-speech model that is understandable and stable enough for an investor demonstration and for an ASR -> response -> TTS prototype.

This milestone is not a production or commercial release. The currently available Bible speech is licensed CC BY-NC 4.0, the narrator identity and consent status must be recorded, and native-speaker listening evaluation is still required.

## Fixed technical decisions

- Use the Dagbani Bible audio-text Parquet corpus as the first paired-speech source.
- Exclude the navigation corpus because native listening found that it does not sound like acceptable Dagbani.
- Treat `ghananlpcommunity/ghana-speech` Dagbani as a probable repackaging or overlap source until audio hashes prove otherwise.
- Do not use unlabeled WAXAL ASR audio for the first TTS model. TTS requires clean paired text and speech.
- Train a single-voice model first. The audit must determine whether the Bible corpus contains one consistent narrator before training is unlocked.
- Use the original 24 kHz audio when it passes quality checks.
- Begin with grapheme/character input. Preserve the characters `ɛ`, `ɔ`, `ŋ`, `ɣ`, and `ʒ` and ordinary punctuation.
- Do not insert written tone, guess pronunciations, or use the existing automatic Dagbani phonemizer as ground truth.
- Preserve the source transcription exactly in `text_raw`. `text_train` may use Unicode NFC, whitespace cleanup, and explicitly documented punctuation handling only.
- Use one Kaggle GPU for initial training. Two-GPU DDP is allowed only after a one-GPU smoke test and an explicit DDP test succeed.
- Never replace missing audio with synthetic waveforms or silently skip a required source.

## Source position

| Source | Initial use | Decision |
|---|---|---|
| `ghananlpcommunity/dagbani-bible-audio-text-tts` | Audit, training, validation, sealed external test | Primary paired corpus; CC BY-NC 4.0 |
| `ghananlpcommunity/ghana-speech`, Dagbani subset | Overlap and metadata audit | Do not mix until hashes prove it is independent |
| `google/WaxalNLP`, `dag_asr` | None for initial TTS training | ASR data; no verified official `dag_tts` configuration |
| Navigation Dagbani | None | Rejected by native listening |
| `ghananlpcommunity/ghana-speech-eval`, Bible Dagbani | Evaluation only | Never train or tune on it |
| `ghananlpcommunity/dagbani_tts-2025_v2` | Inference baseline only | Benchmark first; provenance, training data, metrics, and license are insufficiently documented |

## Canonical TTS manifest

Both notebooks will use one manifest with at least:

```text
sample_id, source, source_version, audio_locator, text_raw, text_train,
speaker_id, duration_s, sample_rate, split, domain, license, audio_hash,
transcript_hash, alignment_score, clipping_ratio, silence_ratio,
is_accepted, rejection_reason
```

The existing ASR Phase 1 v3 Bible split may seed the TTS split, but it does not replace a TTS-specific audit. TTS additionally needs narrator consistency, recording consistency, silence, clipping, loudness, and alignment checks.

## Notebook 03: data audit and baseline

Proposed file:

```text
notebooks/03_Dagbani_TTS_Data_Audit_and_Baselines_Kaggle.ipynb
```

The notebook will be CPU-first, with an optional single-GPU section for speaker embeddings and model inference.

### Audit work

1. Read metadata before decoding the full corpus.
2. Decode audio in bounded shards with progress, throughput, and ETA.
3. Report actual rows, decoded hours, sample rates, durations, missing audio, and corrupt audio.
4. Detect exact and near-duplicate audio and repeated transcripts.
5. Measure clipping, excessive silence, very low energy, inconsistent loudness, and duration/text outliers.
6. Test audio-text alignment on a stratified sample, using the frozen Dagbani ASR model only as a diagnostic signal.
7. Estimate narrator consistency with speaker embeddings and listening samples. Embedding clusters are review aids, not automatic identity labels.
8. Recover book/chapter or recording-session grouping if the source metadata permits it. Otherwise retain the documented contiguous-content holdout and its limitation.
9. Seal validation and external-test splits before training.
10. Record source licenses and prohibit incompatible data from silent mixing.

### Baseline work

- Run the public `ghananlpcommunity/dagbani_tts-2025_v2` model on a frozen evaluation prompt set.
- Save waveforms, inference time, real-time factor, failures, and model revision.
- Use the frozen Dagbani ASR model to back-transcribe generated audio as a secondary intelligibility proxy.
- Do not select a TTS model using ASR WER alone.

### Frozen prompt set

Create 50-100 native-reviewed sentences covering:

- Everyday conversation
- Short and long utterances
- Questions, statements, and pauses
- Personal and place names
- Numbers and dates
- Dagbani special glyphs
- Similar or context-dependent forms such as `bia` and `bii`
- Bible-domain and non-Bible-domain language

### Phase 1 acceptance gate

- Every admitted audio file decodes.
- Every admitted row has aligned text.
- No audio-hash overlap crosses train, validation, or test.
- Narrator/recording consistency is understood well enough to choose single-speaker or multi-speaker training.
- The evaluation prompts and sealed splits are frozen.
- Licenses and source revisions are recorded.
- At least 20 representative audio-text pairs have been listened to by a native speaker.

Publish artifacts to a separate private dataset repository such as:

```text
ats-tech/dagbani-tts-phase1
```

## Notebook 04: budget-aware VITS training

Proposed file:

```text
notebooks/04_Dagbani_TTS_VITS_Training_Kaggle.ipynb
```

Default architecture: a compact single-speaker VITS-family model with grapheme input. The public Dagbani VITS model remains a baseline until its provenance and license are verified; it must not silently become the training base.

### Mandatory stages

1. **Smoke**
   - 200-500 of the cleanest clips
   - A short training run only
   - Verify glyph round-trip, loss movement, checkpoint save/resume, waveform generation, duration/alignment behavior, and no NaNs or collapsed audio

2. **Clean pilot**
   - Use only a few hours from the most consistent narrator/recording condition
   - Compare several saved checkpoints on unseen text
   - Continue only if native listeners find speech intelligible and the model is not repeating, skipping, or producing noise

3. **Scaled supervised training**
   - Expand first to 10-20 audited hours
   - Expand to the full accepted single-speaker set only after the smaller pilot improves
   - Use length-aware batches, mixed precision where stable, resumable checkpoints, and a fixed session time budget

4. **Optional multi-speaker or adaptation experiment**
   - Allowed only if the audit finds multiple narrators and reliable speaker clusters/identifiers
   - Must remain a separate experiment and repository

### Kaggle safeguards

- Start with one P100 or one T4. Do not waste a T4 x2 allocation unless real DDP is tested.
- Run preprocessing with the accelerator disabled where possible.
- Measure throughput and project completion time early in every GPU run.
- Stop with a safety buffer before the Kaggle session limit.
- Upload the latest resumable state and the best listening checkpoints to private Hugging Face storage.
- Keep optimizer, scheduler, random state, manifest revision, tokenizer vocabulary, configuration, and metric history.
- Do not run expensive full synthesis evaluation at every checkpoint.

Suggested model repository:

```text
ats-tech/dagbani-vits-prototype
```

## Evaluation and model-selection gate

TTS quality cannot be established by training loss alone.

### Automated diagnostics

- Audio decode success
- Clipping and silence rates
- Duration and alignment behavior
- Repetition, omission, and early-stop failures
- Real-time factor and peak memory
- ASR back-transcription WER/CER as a secondary proxy

### Native-speaker listening test

Use at least five listeners if practical, with randomized model labels. Score:

- Intelligibility
- Naturalness
- Pronunciation
- Pace and pauses
- Name pronunciation
- Overall acceptability for a prototype

Record the model commit, prompt ID, listener consent, device, and score. Include failure examples instead of reporting only successful samples.

### Prototype acceptance

- No systematic noise, collapse, repetition, or missing endings.
- Special Dagbani characters synthesize consistently.
- The model handles both seen-domain and ordinary conversational prompts.
- Native listeners can understand most prompts without seeing the text.
- The selected model wins or is clearly more stable than the public baseline.
- All claims remain labelled research/prototype until broader native review and licensing are resolved.

## Integration stage

After the TTS prototype passes listening evaluation:

1. Package a small inference function or Gradio demo.
2. Connect the frozen ASR model to a scripted Dagbani response layer and TTS.
3. Measure end-to-end latency and capture failure examples.
4. Add the Dagbani language/reasoning model later without changing the frozen ASR and TTS evaluations.

The first investor demo can therefore prove the full interaction loop before every model reaches production quality.

## Immediate work order

1. Create Notebook 03.
2. Run its quick audit with CPU and no publishing.
3. Listen to its sampled Bible clips and public-model baseline outputs.
4. Correct source or narrator assumptions.
5. Run the full audit and publish the frozen TTS manifest.
6. Create/run Notebook 04 smoke stage.
7. Run the clean pilot only after smoke acceptance.
8. Scale training only after native listening accepts the pilot.

Do not launch full TTS training before Notebook 03 passes.

## Verified external references

- Dagbani Bible audio-text TTS dataset: https://huggingface.co/datasets/ghananlpcommunity/dagbani-bible-audio-text-tts
- WAXAL dataset configurations: https://huggingface.co/datasets/google/WaxalNLP/blob/main/README.md
- Existing community Dagbani VITS model: https://huggingface.co/ghananlpcommunity/dagbani_tts-2025_v2
- Ghana Speech dataset: https://huggingface.co/datasets/ghananlpcommunity/ghana-speech
- Ghana Speech evaluation-only dataset: https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval
