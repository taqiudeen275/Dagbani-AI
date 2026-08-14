# Dagbani Text-to-Speech (TTS) Acoustic & Vocoder Engineering Playbook

**Document ID**: `DAG-KB-TTS-001`  
**Version**: `1.0.0` (Production Playbook)  
**Target Architectures**: VITS, Coqui XTTS-v2, Matcha-TTS, FastSpeech 2, Meta MMS-TTS  
**Vocoders**: BigVGAN (Snake Activations), HiFi-GAN (Multi-Period / Multi-Scale Discriminators)  
**Audio Standards**: 24.0 kHz / 48.0 kHz 24-bit PCM WAV  

---

## 1. Architectural Blueprint & Paradigms

Text-to-Speech (TTS) synthesis for Dagbani must accurately convey phonemic vowel length contrasts, Advanced Tongue Root ([±ATR]) vowel harmony, and 2-level pitch register melodies (High / Low / Downstep) from text that naturally omits tone marks.

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐     ┌───────────────────────┐
│ Raw Dagbani Text│ ──► │ G2P / Tone Injector  │ ──► │ Acoustic Model         │ ──► │ Neural Vocoder        │
│ "O biɛla Yendi" │     │ [/ó bjɛ̀lá jéndì/]    │     │ (VITS / XTTS-v2 / CFM) │     │ (BigVGAN / HiFi-GAN)  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘     └───────────────────────┘
                                                                 │                              │
                                                     ┌───────────┴───────────┐      ┌───────────┴───────────┐
                                                     │ Monotonic Alignment   │      │ 24kHz / 48kHz Output  │
                                                     │ Flow Latents / SDP    │      │ Studio Synthetic Wave │
                                                     └───────────────────────┘      └───────────────────────┘
```

---

## 2. Acoustic Architecture Zoo & Comparative Evaluation

| Model Architecture | Core Mechanism | Strengths for Dagbani | Limitations / Bottlenecks | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **VITS** (*Variational Inference with adversarial learning*) | Conditional VAE + Normalizing Flows + Stochastic Duration Predictor + HiFi-GAN Decoder | Joint end-to-end training eliminates acoustic mismatch; Monotonic Alignment Search (MAS) handles unaligned audio; stochastic duration models natural tempo variations. | Requires clean single-speaker studio data (15–25h); heavier training computation. | **Primary Choice** for canonical single-speaker studio synthesis (e.g. WAXAL-TTS voice). |
| **Coqui XTTS-v2** | Autoregressive GPT Transformer + Audio Codec RVQ + HiFi-GAN Decoder | 6-second zero-shot voice cloning; multi-speaker conditioning; cross-lingual transfer from 27,000h multilingual base; proven on Gur languages (MOS 4.36 in LoResLM 2026). | Autoregressive latency; potential repetition loops on long sentences without repetition penalties. | **Primary Choice** for multi-speaker conversational bots and dynamic voice cloning. |
| **Matcha-TTS** | Non-autoregressive Optimal Transport Conditional Flow Matching (OT-CFM) | Ultra-fast ODE solver (1–4 inference steps); highly expressive pitch contours; lightweight parameter footprint; robust against alignment collapse. | Requires separate pre-trained vocoder (BigVGAN/HiFi-GAN) and phoneme aligner. | Ideal for **low-latency edge devices** and real-time streaming servers. |
| **FastSpeech 2** | Non-autoregressive Feed-Forward Transformer with explicit Pitch/Energy/Duration Predictors | Deterministic, controllable pitch and speed; extremely fast inference; zero attention-alignment drift. | Pitch and energy predictors can produce robotic, over-smoothed contours if ground-truth $F_0$ is noisy; 2-stage pipeline. | Ideal for **educational apps** requiring variable-speed pronunciation drills. |
| **Meta MMS-TTS** | VITS-based Massively Multilingual Speech (1,107+ languages) | Pre-trained shared backbone across hundreds of African languages; easily adaptable with low data (<5h) via language adapter weights. | Character-based vocabulary may miss Dagbani-specific phonemes if fine-tuning is purely top-layer. | Baseline benchmarking and quick prototype deployment. |

---

## 3. VITS Architecture Deep Dive for Dagbani

VITS optimizes the variational lower bound of the intractable marginal log-likelihood of raw speech waveforms conditioned on text:

$$\log p_\theta(x|c) \ge \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z) - \frac{q_\phi(z|x)}{p_\theta(z|c)}\right]$$

### 3.1 Structural Sub-Modules
1. **Text Encoder**: Multi-head Transformer with relative positional representations encoding IPA phoneme sequences with explicit tone tiers ($\text{H}, \text{L}, \text{M}$).
2. **Posterior Encoder**: Non-causal WaveNet blocks extracting latent representation $z \in \mathbb{R}^{d \times T_{\text{mel}}}$ from linear spectrograms.
3. **Normalizing Flow**: Reversible coupling layers parameterized by WaveNet residual blocks that transform simple prior Gaussian distributions into expressive multimodal distributions.
4. **Stochastic Duration Predictor (SDP)**: Flow-based duration model trained via variational maximum likelihood, capturing the duration dynamics of Dagbani long vowels ($aa, ee, ii, oo, uu$) and geminates ($ll, mm, nn$).
5. **Monotonic Alignment Search (MAS)**: Computes the optimal monotonic alignment matrix $\mathbf{A} \in \mathbb{R}^{T_{\text{text}} \times T_{\text{mel}}}$ without requiring external phone-level forced alignment.
6. **Adversarial Waveform Decoder**: Jointly trained multi-period (MPD) and multi-scale (MSD) HiFi-GAN discriminators.

---

## 4. Coqui XTTS-v2 Adaptation Recipe (LoResLM Protocol)

Adapted from the validated French-Mooré Gur transfer methodology (LoResLM 2026):

### 4.1 Tokenizer Vocabulary Extension
Extend the XTTS-v2 SentencePiece/BPE tokenizer vocabulary to **4,000 tokens** by adding:
1. Special Dagbani BGL characters: `ɛ`, `ɔ`, `ɣ`, `ŋ`, `ʒ`, `Ɛ`, `Ɔ`, `Ɣ`, `Ŋ`, `Ʒ`.
2. Frequent Dagbani digraphs: `kp`, `gb`, `ŋm`, `ny`, `ch`, `sh`.
3. Top 500 Dagbani agglutinative morphemes and root lexemes.

### 4.2 Two-Stage Fine-Tuning Schedule
- **Stage 1 (Single-Speaker Studio Anchor)**:
  - Dataset: 15–20 hours of clean WAXAL-TTS / BibleTTS audio.
  - Epochs: 20 epochs.
  - Optimizer: AdamW ($\text{lr} = 5 \times 10^{-6}$, $\text{weight\_decay} = 10^{-2}$, MultiStepLR decay at epochs 10 and 15).
  - Objective: Autoregressive cross-entropy loss over audio codebook tokens + conditioning speaker latent loss.
- **Stage 2 (Multi-Speaker Adaptation)**:
  - Dataset: 25+ hours of multi-speaker Common Voice & SciDB Dagbani.
  - Epochs: 20 epochs with batch size 8 and gradient accumulation 4.

---

## 5. Text Conditioning, G2P & Tone Modeling

### 5.1 G2P Pipeline Architecture
Feeding raw text directly to acoustic models degrades pronunciation quality. The front-end converts standard BGL text to phonetic IPA with explicit tone tiers:

$$\text{Raw BGL Text} \xrightarrow{\text{Normalization}} \text{NFC Text} \xrightarrow{\text{G2P Rules}} \text{Phonemes} \xrightarrow{\text{Tone Injection}} \text{Tonal Phoneme Stream}$$

### 5.2 Tone Diacritic Restoration (TDR) Module
Because 99% of written Dagbani lacks tone diacritics, an upstream sequence-labeling model (BiLSTM / RoBERTa) predicts syllable tones:
- **Tonal Labels**: High ($\text{H}$), Low ($\text{L}$), Downstep ($!$).
- **Acoustic Pitch Conditioning**: In non-autoregressive models (Matcha-TTS, FastSpeech 2), extracted $F_0$ contours (via PyWorld / CREPE) are pitch-normalized:
  $$\hat{F}_0 = \frac{\log F_0 - \mu_{\log F_0}}{\sigma_{\log F_0}}$$
  The normalized pitch contour conditions the variance adaptor during training.

---

## 6. Neural Vocoder Architectures & Fine-Tuning

### 6.1 BigVGAN with Snake Activations
BigVGAN replaces standard LeakyReLU activations with periodic **Snake activation functions**:

$$f_\alpha(x) = x + \frac{1}{\alpha}\sin^2(\alpha x) = x + \frac{1 - \cos(2\alpha x)}{2\alpha}$$

Snake activations introduce a strong inductive bias for periodic signals, eliminating metallic buzzy artifacts on sustained Dagbani open vowels ($aa, ee, \varepsilon\varepsilon, \rho\rho, oo$) and tonal pitch glides.

```
BigVGAN Generator:
Input Mel (80-band) ──► Conv1D ──► [Upsample Block x4] ──► Snake + Conv1D ──► Waveform (24kHz)
                                           │
                              ┌────────────┴────────────┐
                              │ Anti-Aliased ResBlock   │
                              │ (Snake Activations)     │
                              └─────────────────────────┘
```

### 6.2 Fine-Tuning Recipe for 24kHz / 48kHz Dagbani Voice
- **Sampling Rate**: Resample all recordings to **$24,000\text{ Hz}$** (or $48,000\text{ Hz}$ for high-fidelity studio models).
- **Spectrogram Config**:
  - FFT size ($N_{\text{fft}}$): 1024
  - Hop size ($N_{\text{hop}}$): 256 samples ($10.66\text{ ms}$)
  - Window size: 1024 (Periodic Hann)
  - Mel filterbanks: 80 channels ($f_{\min} = 0\text{ Hz}, f_{\max} = 12,000\text{ Hz}$)
- **Composite Training Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{adv}}(G; D) + 45\,\mathcal{L}_{\text{FM}}(G; D) + 45\,\mathcal{L}_{\text{Mel}}(G)$$
  where $\mathcal{L}_{\text{adv}}$ is LS-GAN adversarial loss, $\mathcal{L}_{\text{FM}}$ is feature matching loss across discriminator layers, and $\mathcal{L}_{\text{Mel}}$ is multi-resolution L1 mel-spectral loss.
- **Optimizer & Schedule**: AdamW ($\beta_1 = 0.8, \beta_2 = 0.99$, $\text{lr} = 2 \times 10^{-4}$, exponential decay $\gamma = 0.999$). Train for 150k steps.

---

## 7. Studio Recording & Audio Hygiene Standards

### 7.1 Recording Protocol for Native Dagbani Voice Talents
To build a gold-standard single-speaker studio voice bank:

| Parameter | Studio Golden Standard (WAXAL-TTS / Studio) | Multi-Speaker Corpus (Common Voice / SciDB) |
| :--- | :--- | :--- |
| **Duration** | **15 – 25 hours** clean single-speaker | **50 – 150 hours** across 30+ speakers |
| **Acoustic Environment** | Sound-dampened anechoic booth ($\text{RT}_{60} < 0.2\text{s}$) | Residential quiet room with pop filter |
| **Hardware** | Large-diaphragm condenser (e.g. Neumann U87 / Rode NT1) | USB condenser / Studio mobile interface |
| **Resolution** | 48.0 kHz / 24-bit PCM Linear WAV | 16.0 kHz or 24.0 kHz / 16-bit PCM WAV |
| **Loudness Normalization** | **-23.0 LUFS** ($\pm 0.5$ LUFS), True Peak $< -1.0\text{ dBTP}$ (EBU R128) | **-20.0 LUFS** integrated loudness |
| **Silence Padding** | Leading silence: $40\text{ ms}$; Trailing silence: $60\text{ ms}$ | VAD-trimmed leading/trailing silence |
| **Utterance Bounds** | $1.0\text{s} \le \text{Duration} \le 12.0\text{s}$ | $0.5\text{s} \le \text{Duration} \le 30.0\text{s}$ |
| **Phonetic Coverage** | 100% Dagbani phonemes, tone patterns, and diphtongs | Natural conversational distribution |

---

## 8. Evaluation Framework: MOS, UTMOS & Pitch Correlation

### 8.1 Subjective Evaluation (Mean Opinion Score - MOS)
- Conduct double-blind MUSHRA or MOS listening tests with native Dagbamba evaluators in Tamale and Yendi.
- Scale: 1 (Completely unnatural / Unintelligible) to 5 (Indistinguishable from native human speaker).
- **Target Threshold**: $\text{MOS} \ge 4.20$.

### 8.2 Objective Metrics (UTMOS & MCD)
1. **UTMOS (Unit-based Speech Assessment Model)**: Deep neural speech assessment predicting MOS automatically ($\text{Target UTMOS} \ge 3.40$).
2. **Mel-Cepstral Distortion (MCD)**:
   $$\text{MCD} = \frac{10\sqrt{2}}{\ln 10} \frac{1}{T}\sum_{t=1}^T \sqrt{\sum_{d=1}^D (c_{t,d} - \hat{c}_{t,d})^2} \quad (\text{dB})$$
   $\text{Target MCD} \le 4.5\text{ dB}$.
3. **$F_0$ Frame Error (FFE) & Tone Correlation**:
   Measures pitch contour alignment against reference recordings to verify tone preservation ($\text{Target FFE} \le 12\%$).

---

## 9. References & Technical Foundations
1. Kim, J., et al. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)*. ICML 2021.
2. Popov, V., et al. (2024). *Matcha-TTS: A Fast and Expressive Non-Autoregressive Flow-Matching Architecture for Speech Synthesis*. ICASSP 2024.
3. Lee, S., et al. (2023). *BigVGAN: A Universal Neural Vocoder with Anti-Aliased Periodic Activation Functions*. ICLR 2023.
4. Coqui AI. (2024). *XTTS-v2: Advanced Multilingual Cross-Lingual Voice Cloning*.
5. Ouédraogo, F. S. A., et al. (2026). *Contributing to Speech-to-Speech Translation for African Low-Resource Languages: Study of French-Mooré Pair (LoResLM)*.
