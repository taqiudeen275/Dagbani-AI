# VITS and Advanced Acoustic Architectures for Dagbani TTS

This reference guide details neural acoustic architectures tailored for Dagbani (*Dagbanli*, `dag`), focusing on **VITS**, **Coqui XTTS-v2**, **Matcha-TTS**, and **Meta MMS-TTS**.

---

## 1. Architectural Taxonomy & Benchmark Comparison

| Dimension | VITS (Variational Inference End-to-End) | Coqui XTTS-v2 (Autoregressive Multi-speaker) | Matcha-TTS (Optimal Transport Flow Matching) | Meta MMS-TTS (Massively Multilingual VITS) |
|---|---|---|---|---|
| **Paradigm** | Conditional VAE + Normalizing Flows + HiFi-GAN GAN | Autoregressive Decoder + HiFi-GAN Vocoder | Non-Autoregressive ODE (Flow Matching) | Multilingual VITS Backbone + Language Adapter |
| **Acoustic Input** | Phoneme IDs + Tone Tier Sequences | Grapheme / Phoneme Byte Tokens | Phoneme Sequence + MAS Durations | Character / Phoneme Sequences |
| **Duration Modeling** | Stochastic Duration Predictor (Flow-based) | Autoregressive Next-Token Sampling | Duration Predictor + Monotonic Alignment | Stochastic Duration Predictor |
| **Alignment Method** | Monotonic Alignment Search (MAS) | Cross-Attention Latent Vectors | Monotonic Alignment Search (MAS) | Monotonic Alignment Search (MAS) |
| **Vocoder Coupling** | Fully Joint End-to-End | 2-Stage Neural Decoders | 2-Stage (External BigVGAN/HiFi-GAN) | Fully Joint End-to-End |
| **Inference Speed** | **Fast (0.05x RTF)** | Moderate (0.35x RTF) | **Ultra-Fast (0.02x RTF)** | **Fast (0.05x RTF)** |
| **Hardware Target** | Server / Desktop / Raspberry Pi 4 | Server GPU / Multi-speaker Cloud | Low-Power Edge Devices / Android | Server & Mobile Embedded |
| **Target Dagbani Data** | 10–25h clean studio audio (single/multi-speaker) | 6s voice prompt + 20h adaptation corpus | 5–15h clean audio | 1–5h low-resource transfer |

---

## 2. VITS Architecture Deep Dive for Dagbani

VITS (*Variational Inference with adversarial learning for end-to-end Text-to-Speech*) optimizes the lower bound of raw audio log-likelihood without intermediate spectrogram estimation, eliminating two-stage synthesis mismatch.

```
                           ┌──────────────────────────────────────────────┐
                           │      Dagbani Phoneme Sequence (c_text)        │
                           └──────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │     Text Encoder      │
                                      │  (Transformer + RPE)  │
                                      └───────────────────────┘
                                                  │
                                        ┌─────────┴─────────┐
                                        ▼                   ▼
                                   Prior Mean μ_p     Prior Variance σ_p
                                        │                   │
                                        ▼                   ▼
    ┌──────────────────────┐      ┌───────────────────────────────┐
    │ Linear Spectrogram x │ ───► │  Monotonic Alignment Search   │ ◄─── Stochastic Duration
    └──────────────────────┘      │            (MAS)              │      Predictor (SDP)
               │                  └───────────────────────────────┘
               ▼                                  │
    ┌──────────────────────┐                      ▼
    │  Posterior Encoder   │            Latent Representation z
    │ (Non-causal WaveNet) │                      │
    └──────────────────────┘                      ▼
               │                        ┌───────────────────┐
               ▼                        │ Normalizing Flows │
         Posterior z                    │ (WaveNet Blocks)  │
               │                        └───────────────────┘
               └──────────────────────────────────┬─┘
                                                  ▼
                                      ┌───────────────────────┐
                                      │   HiFi-GAN Decoder    │
                                      │ (MRF + ConvTranspose) │
                                      └───────────────────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │  Synthesized Waveform │
                                      │   (24 kHz / 48 kHz)   │
                                      └───────────────────────┘
```

### 2.1 Handling Dagbani Linguistic Specifics in VITS

1. **Tone Encoding in Text Prior**:
   - VITS text embeddings concatenate phoneme ID embedding $e_{\text{phn}} \in \mathbb{R}^{192}$ with tone marker embedding $e_{\text{tone}} \in \mathbb{R}^{64}$ (High `H`, Low `L`, Downstep `!H`, Neutral `0`).
   - This ensures the text encoder prior directly parameters $p(z|c)$ with appropriate $F_0$ distribution anchors.

2. **Stochastic Duration Predictor for Vowel Lengths**:
   - Dagbani contrasts short vowels ($/a, e, i, o, u, ɨ/$) with bimoraic long vowels ($/aː, eː, iː, oː, uː/$) and geminates ($/lː, dː/$).
   - The SDP models one-to-many speech rhythm variability via residual normalizing flows, preventing robotic, static vowel durations.

3. **Loss Functions**:
   $$\mathcal{L}_{\text{VITS}} = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{adv}}(G; D) + \mathcal{L}_{\text{fm}}(G; D)$$
   - **Reconstruction Loss**: Mel-spectrogram L1 distance between ground truth $x$ and decoded $\hat{x}$.
   - **KL Divergence**: Regularizes posterior distribution $q(z|x)$ to match text prior $p(z|c_{\text{text}})$.
   - **Duration Loss**: Maximizes log-likelihood of SDP flow transformations.
   - **Adversarial & Feature Matching Losses**: HiFi-GAN Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD).

---

## 3. Coqui XTTS-v2 Multi-Speaker Voice Cloning

XTTS-v2 uses a GPT-style autoregressive speech token decoder conditioned on a 6-second speaker audio prompt.

### Adaptation Recipe for Dagbani:
1. **Vocabulary Expansion**:
   - Insert Dagbani BGL characters (`ɛ`, `ɔ`, `ŋ`, `ɣ`, `ʒ`, `'`) and common syllable units into the SentencePiece vocabulary.
2. **LoRA Adaptation**:
   - Rank $r = 32$, $\alpha = 32$, applying LoRA to attention projection weights in the conditioning autoregressive encoder.
3. **Training Data Requirements**:
   - 10 to 50 hours of multi-speaker Dagbani recordings (e.g., Common Voice v24, SciDB Ghanaian Audio, WAXAL speech corpus).

---

## 4. Matcha-TTS Flow-Matching Architecture

Matcha-TTS replaces traditional diffusion formulations with **Optimal Transport Conditional Flow Matching (OT-CFM)**:
- Generates 80-band mel-spectrograms in as few as 2 to 4 Euler ODE steps.
- Uses a compact 1D CNN + Transformer encoder with 2D U-Net backbone.
- Yields natural prosody with extremely small parameter footprints (~18M parameters), ideal for on-device deployment on low-cost Android phones in Ghana.

---

## 5. Meta MMS-TTS (Massively Multilingual Speech)

Meta MMS-TTS provides a pre-trained VITS checkpoint supporting `dag` (ISO 639-3).
- **Fine-Tuning MMS Checkpoints**:
  - Load base weights `facebook/mms-tts-dag`.
  - Update learning rate: $\text{lr} = 1\times 10^{-4}$ with warmup.
  - Inject custom phoneme mappings to resolve orthographic homographs and tone minimal pairs.
