import os
from pathlib import Path

content = '''---
name: deliberate-engineering-protocol
description: Universal protocol for calm, hypothesis-driven, evidence-first engineering, ASR, and ML execution. Enforces stage gates, statistical validation, regression guards, and resource economics over hasty fixes and premature scaling.
---

# Deliberate Engineering & Evidence-First Execution Protocol

> *"Slow is smooth, and smooth is fast."*  
> Excellence in software engineering and machine learning does not come from the rapid accumulation of unverified changes, but from calm, disciplined deliberation, rigorous verification gates, and deep respect for invariants.

---

## 1. Core Principles: Calm Deliberation over Fast Gratification

When confronted with a problem, an experiment, or an evaluation result, **resist the immediate urge to blindly scale or apply quick shotgun patches**. Always adhere to the following principles:

1. **The Law of Parsimony (Minimum Viable Intervention)**:
   - Always seek the smallest, cheapest, most targeted test that definitively validates or invalidates a hypothesis.
   - If a 100-step pilot or a simple diagnostic can test the hypothesis, never launch a 750-step run or a larger model architecture prematurely.

2. **Statistical Significance over Intuition**:
   - A positive result on a small sample (e.g. 256 clips) is an *indicative signal*, not a definitive proof.
   - Never commit to long training or claim victory without evaluating across the full validation distribution and checking confidence intervals (CI).

3. **Strict Invariant & Baseline Preservation**:
   - An improvement on Out-of-Domain / Feature B is a **failure** if it silently degrades the core performance on Primary Domain / Feature A (catastrophic forgetting/regression).
   - Baseline champions and historical checkpoints must remain **frozen, isolated, and immutable**. Never write new experiment checkpoints into the parent repository.

4. **Sealed Holdouts are Sacred**:
   - Test splits (`test`, `external_test`) must remain sealed until final model selection is completed on validation sets. Never tune hyperparameters or make candidate selection decisions against test data.

5. **Data & Adaptation Before Model Scaling**:
   - Exhaust data quality, split sanitization, batch-mixing ratios, and learning rate stability on the smaller/simpler model (e.g., Whisper Small) before considering larger, more expensive architectures (e.g., Whisper Medium / Large).

---

## 2. The 5-Stage Deliberate Execution Cycle

```mermaid
graph TD
    A[Stage 1: Formulate Specific Hypothesis] --> B[Stage 2: Minimal Diagnostic / Pilot Run]
    B --> C[Stage 3: Full Statistical Validation on Frozen Splits]
    C --> D{Stage 4: Deliberation Gate}
    D -->|Goal Satisfied & Zero Regression| E[Accept Candidate / Proceed to Next Milestone]
    D -->|Slight Regression Detected| F[Calm Adjustment: Tune Single Variable & Re-pilot]
    D -->|Definitive Failure| G[Revert to Frozen Champion & Reformulate Hypothesis]
    E --> H[Stage 5: Unseal Final Test / Proceed to Deployment]
```

### Stage 1: Hypothesis & Boundary Definition
Before proposing code or launching a compute job, explicitly answer:
- What exact question are we testing? (e.g., *"Does a 70/30 WAXAL/Bible mix learn Bible vocabulary without forgetting WAXAL?"*)
- What constitutes success? (Specific target metric on validation).
- What constitutes failure or unacceptable regression? (e.g., *WAXAL Normalized WER degradation > +0.005*).
- What is the cheapest way to observe this signal?

### Stage 2: Minimal Diagnostic (Smoke / Pilot)
- Run a bounded, low-budget trial (e.g., 20-step smoke or 100-step pilot).
- Verify pipeline mechanics, memory safety, gradient flow, and loss convergence without spending massive compute quota.

### Stage 3: Full Statistical Validation
- Take the candidate checkpoint and evaluate it **exhaustively against the full validation split** (not just a 256-sample subset).
- Run paired comparisons against the frozen baseline to establish $95\%$ bootstrap confidence intervals.

### Stage 4: The Deliberation Gate (Decision Matrix)
- **Case 1: Metric Target Met + Zero Regression (CI Upper Bound <= +0.005)**:
  - **Verdict**: The existing lightweight pilot is already sufficient! **Do not scale up training steps.** Accept the pilot as the champion and advance to the next pipeline stage.
- **Case 2: Domain Gain + Mild Regression (> +0.005)**:
  - **Verdict**: Do not launch a blind 750-step run. Adjust one specific parameter (e.g., increase in-domain share to 80%, lower learning rate to 2e-5) and test a fresh pilot.
- **Case 3: Unstable / Severe Regression**:
  - **Verdict**: Hard stop. Revert cleanly to the frozen baseline. Re-evaluate data formatting, alignments, or tokenization.

### Stage 5: Invariant Protection & Unsealing
- Only after validation conclusively crowns a candidate do you evaluate against the sealed test set for official documentation.

---

## 3. Cognitive Anti-Patterns to Prevent

| Anti-Pattern | Manifestation | Deliberate Antidote |
| :--- | :--- | :--- |
| **The Full-Run Reflex** | *"The 100-step pilot was good, so let's immediately run 1,000 steps."* | **Ask**: Did the 100-step run already achieve the objective? Will 1,000 steps overfit or cause catastrophic forgetting? Run full validation first. |
| **Shotgun Debugging** | Modifying learning rate, batch size, loss weighting, and data filtering all in one go. | Change **one independent variable** at a time to isolate causality. |
| **Subset Overconfidence** | Concluding a model is "production-ready" after evaluating on 50 or 256 cherry-picked samples. | Evaluate the entire validation set and inspect statistical confidence intervals. |
| **Premature Architecture Scaling** | Jumping from Small to Medium / Large before optimizing data hygiene and training schedules. | Extract maximum efficiency from the smaller baseline before spending 4x compute on larger weights. |
| **Test Set Peeking** | Evaluating candidate iterations against the test set to pick the best checkpoint. | Keep test sets strictly sealed. Checkpoint selection must only use validation splits. |

---

## 4. Agent Deliberation Checklist (Self-Reflection Gate)

Before presenting any recommendation or executing an action, the agent must ask:

1. [ ] **Is this the simplest, lowest-risk next step?**
2. [ ] **Am I rushing into execution when an evaluation-only or diagnostic step is needed first?**
3. [ ] **Are all baseline assets and parent checkpoints safely preserved from accidental overwrite?**
4. [ ] **Is the evaluation sample large enough to draw statistically sound conclusions?**
5. [ ] **Does this plan preserve existing capabilities while adding new ones?**
'''

user_home = Path(os.environ['USERPROFILE'])
paths = [
    user_home / '.gemini' / 'config' / 'skills' / 'deliberate-engineering-protocol' / 'SKILL.md',
    user_home / '.gemini' / 'antigravity' / 'builtin' / 'skills' / 'deliberate-engineering-protocol' / 'SKILL.md',
    Path('d:/ATS Tech/Dagbani AI/.agents/skills/deliberate-engineering-protocol/SKILL.md'),
    Path('d:/ATS Tech/Dagbani AI/skills/deliberate-engineering-protocol/SKILL.md')
]

for p in paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + '\n', encoding='utf-8')
    print(f'Successfully wrote: {p}')

global_skills_json = user_home / '.gemini' / 'config' / 'skills.json'
global_skills_json.write_text('{\n  "entries": [\n    { "path": "skills" }\n  ]\n}\n', encoding='utf-8')
print(f'Wrote: {global_skills_json}')
