# Neural Vocoder Fine-Tuning Guide for Dagbani

This reference guide provides setup, architecture configurations, and fine-tuning recipes for **HiFi-GAN** and **BigVGAN** vocoders on Dagbani speech datasets.

---

## 1. Vocoder Architecture Comparison

| Parameter | HiFi-GAN (V1) | BigVGAN (Base / Large) | MelGAN |
|---|---|---|---|
| **Activation Function** | LeakyReLU ($\alpha = 0.1$) | Anti-Aliased Snake ($\alpha$) | LeakyReLU |
| **Discriminator Design** | Multi-Period (MPD) + Multi-Scale (MSD) | MPD + Multi-Resolution Multi-Scale STFT (MRD) | Multi-Scale (MSD) |
| **Pitch Tracking Fidelity** | Good on monotonic languages; can ring on rapid tonal glides | **Exceptional**; continuous harmonic tracking without artifacts | Prone to metallic buzz on nasal vowels |
| **Sampling Rate Target** | 22.05 kHz or 24.0 kHz | **24.0 kHz** (studio default) or 44.1 kHz | 22.05 kHz |
| **Out-of-Distribution Robustness** | Moderate | **High** (handles varied room acoustics) | Low |
| **Compute / Memory** | 13.9M params; runs 10x real-time on CPU | 14.3M (Base) to 112M (Large) | 4.2M params |

---

## 2. DSP & Audio Preprocessing Configuration

Vocoders must be trained on mel-spectrograms matching the acoustic model's feature extraction pipeline.

```json
{
  "audio": {
    "sampling_rate": 24000,
    "max_wav_value": 32768.0,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": 12000.0,
    "mel_fmax_loss": null
  },
  "model": {
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
  }
}
```

---

## 3. BigVGAN Snake Activation Mechanics

Traditional LeakyReLU activations lack periodic inductive bias, leading to phase distortion in tonal languages like Dagbani. BigVGAN replaces them with anti-aliased periodic **Snake** activations:

$$f_\alpha(x) = x + \frac{1}{\alpha}\sin^2(\alpha x) = x + \frac{1 - \cos(2\alpha x)}{2\alpha}$$

Where:
- $\alpha$ controls the frequency of periodic components and is learnable per channel.
- Low-pass filters prevent high-frequency aliasing during nonlinear activation.

---

## 4. Fine-Tuning Recipe & Optimization Protocol

To adapt a pre-trained universal BigVGAN / HiFi-GAN model to Dagbani:

### Step 1: Checkpoint Initialization
- Download pre-trained universal checkpoint (e.g. `bigvgan_24khz_100band` or `hifigan_v1_universal`).

### Step 2: Training Configuration
- **Batch Size**: 16 utterances per GPU (segment size: 8,192 samples = ~341 ms).
- **Optimizer**: AdamW ($\beta_1 = 0.8$, $\beta_2 = 0.99$, $\epsilon = 1\times 10^{-8}$, $\text{weight\_decay} = 0.01$).
- **Learning Rate**: $2.0 \times 10^{-4}$ for Generator, $2.0 \times 10^{-4}$ for Discriminators, decaying with $\gamma = 0.999$ per epoch.

### Step 3: Multi-Task Loss Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{adv}}(G; D) + 45\,\mathcal{L}_{\text{FM}}(G; D) + 45\,\mathcal{L}_{\text{Mel}}(G)$$

Where:
- $\mathcal{L}_{\text{adv}}$: Least-squares GAN adversarial loss.
- $\mathcal{L}_{\text{FM}}$: Feature matching loss across intermediate discriminator layers.
- $\mathcal{L}_{\text{Mel}}$: L1 loss between ground-truth and synthesized 80-band log-mel spectrograms.

### Step 4: Convergence & Evaluation
- Train for **100,000 to 200,000 steps** (~12–24 hours on a single NVIDIA RTX 3090 / A100).
- Monitor validation Mel L1 loss (target $< 0.35$) and inspect spectrogram harmonics for labial-velar stops (`kp`, `gb`) and nasalized vowels.
