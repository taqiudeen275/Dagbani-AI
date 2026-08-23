# Example: Low-Resource ASR Domain Adaptation Gate

This example demonstrates how the Deliberate Engineering Protocol was applied during the Dagbani ASR project to prevent unnecessary GPU spend and protect in-domain transcription accuracy.

---

## Scenario
A 100-step pilot (`pilot-v2`) adapting Whisper Small to the Dagbani Bible produced:
- **Bible Normalized WER**: 68.99% -> 39.84% (-29.16% win on 256 samples).
- **WAXAL Normalized WER**: 29.12% -> 29.22% (+0.10% delta on 256 samples).

---

## Hasty vs. Deliberate Response

### The Hasty (Fast-Gratification) Reflex:
- *"The pilot won by 29%! Let's immediately launch a 750-step training run or train Whisper Medium!"*
- **Flaws**:
  - 256 samples has wide error bars (standard error $\approx \pm 1.5\%$).
  - 750 steps on a 70/30 mix risks severe catastrophic forgetting on conversational WAXAL.
  - Whisper Medium takes $4\times$ more VRAM and compute without proof that Small is saturated.

### The Deliberate (Evidence-First) Protocol:
1. **Action**: Stop training. Run **Evaluation-Only** on the entire 1,750-sample WAXAL validation split (`run_training=False`, `eval_generation_limit=None`).
2. **Metric**: Compute the paired bootstrap 95% confidence interval on WAXAL (`paired_comparisons.json`).
3. **Decision Tree**:
   - **If CI Upper Bound $\le +0.005$ WER**: Accept the 100-step pilot directly as the domain champion! **Zero further training needed.**
   - **If CI shows true regression**: Run another gentle 100-step pilot with $80\%$ in-domain share and $2\text{e-}5$ learning rate.
   - **Whisper Medium**: Remain locked behind the medium gate until Small data experiments are fully completed.\n