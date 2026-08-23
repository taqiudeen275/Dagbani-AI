import os
from pathlib import Path

guide_content = '''# Deliberate Engineering & Evidence-First Guide

## The Psychology of the "Rush for Fix" Trap
Engineers and AI agents often succumb to **Action Bias** (the impulse to write code or run heavy compute before fully understanding the problem) and **The Sunk Scaling Trap** (the assumption that doing *more* of something—more steps, larger models, more data—is always better).

### Core Manifestations & Traps:
1. **Premature Scaling**: Escalating to 750 or 2,500 training steps after a 100-step pilot showed good numbers, without checking whether the 100-step checkpoint had already converged or whether longer exposure degrades primary domain fluency.
2. **Shotgun Hyperparameter Tuning**: Adjusting learning rate, batch size, loss weights, and dataset splits simultaneously, obfuscating causality.
3. **Subset Cherry-Picking**: Concluding equivalence or superiority based on a 100 or 250-sample subset, where standard error is $\\pm 2-3\\%$, without computing a 95% bootstrap confidence interval across the entire distribution.
4. **Catastrophic Regression Blindness**: Celebrating a 30% drop in error rate on a new domain while neglecting a silent 2% regression on the core domain that pays the bills.
5. **Test Set Contamination**: Unsealing test sets during early iterations, destroying their integrity as objective final holdouts.

---

## The 4 Golden Rules of Deliberate Execution

### Rule 1: The Diagnostic Principle
Before scaling or modifying a system, run an **evaluation-only** diagnostic or a **minimal smoke test** to establish exact current state and confidence intervals.

### Rule 2: Single-Variable Isolation
Change **exactly one controlled variable** per experimental cycle. If changing the data mix ratio, keep learning rate, warmup, and batch size constant.

### Rule 3: Asymmetric Risk Guarding
Always weight the cost of regression on Core Domain A significantly higher than the potential gain on Peripheral Domain B. If Core Domain A regresses beyond acceptable noise bounds (e.g. $> +0.005$ WER), reject the candidate regardless of domain B performance.

### Rule 4: Holdout Sanctity
Holdouts exist in strict hierarchies:
- **Smoke Subset**: Pipeline mechanics only.
- **Validation Slice**: Fast diagnostic check.
- **Full Validation Split**: Candidate selection and statistical confidence testing.
- **Sealed Test Split**: Unsealed exactly once at project milestone release.
'''

example_content = '''# Example: Low-Resource ASR Domain Adaptation Gate

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
  - 256 samples has wide error bars (standard error $\\approx \\pm 1.5\\%$).
  - 750 steps on a 70/30 mix risks severe catastrophic forgetting on conversational WAXAL.
  - Whisper Medium takes $4\\times$ more VRAM and compute without proof that Small is saturated.

### The Deliberate (Evidence-First) Protocol:
1. **Action**: Stop training. Run **Evaluation-Only** on the entire 1,750-sample WAXAL validation split (`run_training=False`, `eval_generation_limit=None`).
2. **Metric**: Compute the paired bootstrap 95% confidence interval on WAXAL (`paired_comparisons.json`).
3. **Decision Tree**:
   - **If CI Upper Bound $\\le +0.005$ WER**: Accept the 100-step pilot directly as the domain champion! **Zero further training needed.**
   - **If CI shows true regression**: Run another gentle 100-step pilot with $80\\%$ in-domain share and $2\\text{e-}5$ learning rate.
   - **Whisper Medium**: Remain locked behind the medium gate until Small data experiments are fully completed.
'''

user_home = Path(os.environ['USERPROFILE'])
skill_dirs = [
    user_home / '.gemini' / 'config' / 'skills' / 'deliberate-engineering-protocol',
    user_home / '.gemini' / 'antigravity' / 'builtin' / 'skills' / 'deliberate-engineering-protocol',
    Path('d:/ATS Tech/Dagbani AI/.agents/skills/deliberate-engineering-protocol'),
    Path('d:/ATS Tech/Dagbani AI/skills/deliberate-engineering-protocol')
]

for sdir in skill_dirs:
    ref_dir = sdir / 'references'
    ex_dir = sdir / 'examples'
    ref_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)
    
    (ref_dir / 'deliberate_engineering_guide.md').write_text(guide_content.strip() + '\\n', encoding='utf-8')
    (ex_dir / 'asr_gate_workflow.md').write_text(example_content.strip() + '\\n', encoding='utf-8')
    print(f'Populated references and examples in: {sdir}')
