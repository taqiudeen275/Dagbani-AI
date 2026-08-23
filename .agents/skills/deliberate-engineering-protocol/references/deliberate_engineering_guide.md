# Deliberate Engineering & Evidence-First Guide

## The Psychology of the "Rush for Fix" Trap
Engineers and AI agents often succumb to **Action Bias** (the impulse to write code or run heavy compute before fully understanding the problem) and **The Sunk Scaling Trap** (the assumption that doing *more* of something—more steps, larger models, more data—is always better).

### Core Manifestations & Traps:
1. **Premature Scaling**: Escalating to 750 or 2,500 training steps after a 100-step pilot showed good numbers, without checking whether the 100-step checkpoint had already converged or whether longer exposure degrades primary domain fluency.
2. **Shotgun Hyperparameter Tuning**: Adjusting learning rate, batch size, loss weights, and dataset splits simultaneously, obfuscating causality.
3. **Subset Cherry-Picking**: Concluding equivalence or superiority based on a 100 or 250-sample subset, where standard error is $\pm 2-3\%$, without computing a 95% bootstrap confidence interval across the entire distribution.
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
- **Sealed Test Split**: Unsealed exactly once at project milestone release.\n