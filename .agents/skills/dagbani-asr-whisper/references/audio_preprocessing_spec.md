# Dagbani Audio Ingestion, Preprocessing & Feature Extraction Specification

**Document Version**: 1.0.0  
**Target Architectures**: OpenAI Whisper, Wav2Vec 2.0, Meta MMS, Faster-Whisper  

---

## 1. Acoustic Standard Specifications

All audio signals ingested by the Dagbani AI speech processing pipeline must conform to the following normative acoustic standards:

| Acoustic Dimension | Specification Standard | Validation Boundary / Rule |
| :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | **$16,000\text{ Hz}$ ($16\text{ kHz}$)** | Automatic polyphase sinc resampling from any input rate (44.1k, 48k, 8k). |
| **Channel Format** | **1 Channel (Mono)** | Multichannel input downmixed via uniform arithmetic averaging across channels. |
| **Bit Depth / Format** | **16-bit Signed PCM / Float32** | Linear PCM `int16` or normalized `float32` in dynamic range $[-1.0, 1.0]$. |
| **Duration Bounds** | **$[0.5\text{s}, 30.0\text{s}]$** | Drop clips $<0.5\text{s}$; VAD-chunk or segment clips $>30.0\text{s}$. |
| **Peak Normalization** | **$-1.0\text{ dBFS}$** | Amplitude scaling to prevent clipping while maximizing dynamic range. |
| **DC Offset Removal** | High-pass DC filter ($50\text{ Hz}$) | Eliminates microphone bias and sub-audible mechanical rumble. |

---

## 2. Log-Mel Filterbank Feature Extraction Parameters

Whisper converts 30-second 1D waveform buffers into 2D log-magnitude Mel-scale spectrogram tensors.

### 2.1 Whisper Filterbank Configuration Table

| Parameter | Whisper Tiny / Base / Small / Medium | Whisper Large-v3 | Unit / Formula |
| :--- | :--- | :--- | :--- |
| **Sampling Rate ($f_s$)** | $16,000\text{ Hz}$ | $16,000\text{ Hz}$ | Samples / second |
| **Filterbank Bins ($N_{\text{mels}}$)** | **80 channels** | **128 channels** | Mel frequency channels |
| **STFT Window Size ($N_{\text{fft}}$)** | 400 samples ($25.0\text{ ms}$) | 400 samples ($25.0\text{ ms}$) | $T_{\text{win}} = \frac{N_{\text{fft}}}{f_s}$ |
| **Hop Length ($N_{\text{hop}}$)** | 160 samples ($10.0\text{ ms}$) | 160 samples ($10.0\text{ ms}$) | $T_{\text{hop}} = \frac{N_{\text{hop}}}{f_s}$ |
| **Window Function** | Periodic Hann Window | Periodic Hann Window | $w[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{N}\right)$ |
| **Frequency Range** | $0\text{ Hz} - 8,000\text{ Hz}$ | $0\text{ Hz} - 8,000\text{ Hz}$ | Nyquist limit at $16\text{ kHz}$ |
| **Target Frame Count (30s)** | **3000 frames** | **3000 frames** | $\frac{30.0 \times 16000}{160} = 3000$ |
| **Tensor Dimensions** | $(80, 3000)$ | $(128, 3000)$ | $(\text{Channels}, \text{Frames})$ |
| **Log Compression** | $\log_{10}(\max(M, 10^{-5}))$ scaled to $[-1, 1]$ | $\log_{10}(\max(M, 10^{-5}))$ scaled to $[-1, 1]$ | Bounded dynamic scaling |

---

## 3. Voice Activity Detection (VAD) & Utterance Chunking

### 3.1 VAD Pipeline Mechanics
To process long-form Dagbani field recordings, radio broadcasts, and oral history narrations:
1. **Model**: Silero VAD v4/v5 or adaptive RMS energy thresholding.
2. **Thresholds**:
   - Speech Probability Threshold: $0.5$
   - Minimum Speech Duration: $250\text{ ms}$
   - Minimum Silence Duration: $400\text{ ms}$
   - Speech Boundary Padding: $100\text{ ms}$
3. **Splitting Heuristic**:
   - Traverse speech segment timestamps.
   - Group contiguous segments until total duration approaches target ($24.0\text{s} - 28.0\text{s}$).
   - Split at the highest-confidence silence valley to avoid chopping mid-word or mid-syllable.

---

## 4. Low-Resource Acoustic Data Augmentation

To maximize acoustic generalization on small Dagbani training corpora:

1. **SpecAugment (Spectrogram Masking)**:
   - **Frequency Masking**: Mask $F \le 27$ consecutive Mel channels with 2 masks ($m_F = 2$).
   - **Time Masking**: Mask $T \le 100$ consecutive time frames ($1.0\text{s}$) with max ratio $p = 0.05$.
2. **Waveform Speed Perturbation**:
   - Resample waveforms by factors of $\{0.9, 1.0, 1.1\}$, shifting pitch and tempo without altering phonetic labels.
3. **Additive Background Noise**:
   - Mix background stationary noise (ambient outdoor environment) with SNR between $10\text{ dB}$ and $30\text{ dB}$.
