Yes—the formal real-world evaluation can be deferred. It should block a production claim, but it does not need to block the next research experiment.

The correct next step is:

**Audit additional labeled datasets first → then domain adaptation → then pseudo-labelling → consider Medium last.**

Do not start Medium now.

## Current position

You have completed:

- WAXAL data audit
- Whisper-small smoke test
- Full supervised training
- Statistical checkpoint selection
- Sealed WAXAL test evaluation
- Checkpoint 2,000 promotion and freezing
- Model-card update
- Informal real-world testing

Your frozen supervised-v1 result is:

- Strict WER: 33.86%
- Strict CER: 11.84%
- Normalized WER: 30.39%
- Normalized CER: 10.99%

That model must remain recoverable and unchanged while later experiments happen.

## Full roadmap

| Stage | Work | Compute | Acceptance gate |
|---|---|---:|---|
| 0. Supervised v1 | Completed and frozen | — | WAXAL reproduction gate passed |
| 1. External-data audit | **Next action** | CPU | Alignment, licensing, duplicates and splits pass |
| 2. Domain adaptation | Conditional on Stage 1 | GPU | External validation improves without WAXAL regression |
| 3. Semi-supervised | After best labeled model | GPU | Pseudo-label training measurably improves validation |
| 4. Medium pilot | Only after Small is settled | GPU | At least 1 absolute validation-WER improvement |
| 5. Formal human evaluation | Before production/public claims | Mostly CPU | Native-speaker review and real-world holdout pass |
| 6. ASR deployment | After final model selection | CPU/GPU | Stable latency, chunking and monitoring |
| 7. Voice assistant | Later | Separate project stages | ASR → reasoning/translation → TTS validated |

## Stage 1: audit the additional data

Create a new combined manifest revision instead of overwriting the supervised-v1 manifest:

```text
ats-tech/dagbani-asr-phase1-v2
```

Run Phase 1 with GPU disabled:

```python
enabled_sources = (
    "waxal_dag_asr",
    "dagbani_bible",
    "navigation_dagbani",
)
```

Audit each source for:

- Actual downloadable row count and hours
- Audio decoding
- Audio/transcript alignment
- Missing transcripts and files
- Exact and near-duplicate audio
- Transcript duplicates
- Overlap with WAXAL
- Speaker leakage
- License
- Separate validation split
- Bible book/chapter leakage

Treat the original SciDB ZIP as a probable WAXAL duplicate. Do not add both unless audio hashes demonstrate that they contain independent recordings.

### Licensing decision

The Bible and navigation datasets were reported as CC BY-NC. If this project might become commercial:

- Keep the current WAXAL model as the commercial-compatible branch.
- Put CC BY-NC experiments in a separate research-only model repository.
- Do not silently promote a noncommercial model over the current root.

Suggested repository:

```text
ats-tech/dagbani-whisper-small-domain-research
```

## Stage 2: domain adaptation

Only begin after the combined audit is release-ready.

Recommended configuration:

```text
Starting model: frozen supervised checkpoint 2000
Stage: domain_adaptation
Maximum steps: 500–750
Learning rate: approximately 3e-5
WAXAL batch share: at least 60%
External labeled share: at most 40%
Validation/checkpoint interval: 250 steps
```

Evaluate on:

- Frozen WAXAL validation
- Bible validation
- Navigation validation
- Any new conversational validation set

Accept the domain model only when:

- External-domain WER/CER improves
- WAXAL conversational validation does not materially regress
- Special Dagbani glyph accuracy does not regress

Do not use the WAXAL test results to choose hyperparameters. That test has already served its purpose for supervised-v1.

### Notebook change required before this run

The current Phase 2 notebook uses `model_repo_id` as both the starting model and output destination. That risks overwriting the frozen champion.

Before domain adaptation, add separate settings:

```text
starting_model_id
output_model_repo_id
```

The domain experiment must read from the frozen champion and write to a different repository.

## Stage 3: pseudo-labelled WAXAL audio

After choosing the best labeled model:

1. Calibrate confidence filters on labeled validation.
2. Pseudo-transcribe no more than 20 hours initially.
3. Reject:
   - Silence
   - Repetitions
   - Implausible speech rates
   - Low average token probability
   - Disagreement between greedy and beam decoding
4. Mix pseudo-labelled data at no more than 25% of batches.
5. Keep at least 75% genuine labeled speech.

Accept the pseudo-labelled model only if it improves both conversational and external validation. Otherwise, retain the domain or supervised model.

Do not automatically expand beyond 20 hours.

## Stage 4: Whisper Medium

Medium comes after the Small-data experiments because better data is usually more valuable and cheaper than immediately scaling the model.

Run:

- A matched 100-step Whisper-small pilot
- A matched 100-step Whisper-medium pilot
- Same data, batch exposure and decoding protocol

Allow full Medium training only if:

- It fits memory
- It fits remaining quota
- It improves validation WER by at least one absolute percentage point
- It does not regress external domains
- Its added runtime is justified

Medium must have its own repository. Never write Medium weights into the Whisper-small repository.

## Real-world evaluation

Formal documentation can wait until before release. You already have a basic inference pipeline at [Dagbani_ASR_Inference_Gradio_Colab.ipynb](</D:/ATS Tech/Dagbani AI/notebooks/Dagbani_ASR_Inference_Gradio_Colab.ipynb>).

It supports microphone recording and audio upload, but it is not yet a formal review system because it does not capture:

- Model commit
- Speaker/session identifier
- Recording conditions
- User-approved reference transcription
- Reviewer correction
- Error category
- Consent
- Exportable results

Also remove or disable its optional WAXAL test-sample cell before sharing it. Repeatedly inspecting test examples encourages test leakage.

Later, extend it to export a private CSV/Parquet review manifest.

## Immediate work order

1. Preserve supervised-v1 and its artifacts.
2. Update Phase 2 to separate starting and output repositories.
3. Run a quick CPU audit of Bible and navigation.
4. Review licenses and source overlap.
5. Run the full combined audit into `phase1-v2`.
6. If accepted, train the separate domain-adaptation model.
7. Compare it using validation only.
8. Attempt 20-hour pseudo-labelling only after that.
9. Run the Medium pilot last.
10. Complete formal real-world/native review before production.

So, your next notebook run should be the **Phase 1 external-data audit**, not domain training and not Medium.