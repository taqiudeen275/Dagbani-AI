# End-to-End VITS Training Example for Dagbani TTS

This example demonstrates setting up and running single-speaker and multi-speaker VITS training pipelines on Dagbani speech datasets (e.g., WAXAL-TTS, BibleTTS, Common Voice).

---

## 1. Directory Structure Setup

Prepare your workspace directory layout:

```
dagbani_vits_project/
├── data/
│   ├── wavs/                          # 24 kHz Mono 16-bit WAV files
│   │   ├── dag_spk01_0001.wav
│   │   └── dag_spk01_0002.wav
│   ├── train_filelist.txt             # filepath|speaker_id|phonemes
│   └── val_filelist.txt
├── configs/
│   └── dagbani_vits_24k.json          # Hyperparameter configuration
├── checkpoints/                       # Saved model checkpoints
└── logs/                              # TensorBoard logs
```

---

## 2. Preparing Filelists with Dagbani Phonemizer

Generate the pipe-delimited training and validation filelists with full phonetic and tone tokens:

```bash
# Process raw dataset into VITS filelist format
python ../scripts/prepare_tts_dataset.py \
    --audio-dir data/raw_audio/ \
    --transcripts data/raw_transcripts.tsv \
    --output-dir data/processed/ \
    --sample-rate 24000 \
    --val-ratio 0.05 \
    --test-ratio 0.05
```

Generated `train_filelist.txt` entry format:
```
data/wavs/dag_spk01_0001.wav|0|o _ b j ɛ́ l á _ j é n d ì
data/wavs/dag_spk01_0002.wav|0|d á g b á ŋ _ k á y á _ n í _ t á ʔ á d á
```

---

## 3. VITS Configuration File (`configs/dagbani_vits_24k.json`)

```json
{
  "train": {
    "log_interval": 100,
    "eval_interval": 1000,
    "seed": 1234,
    "epochs": 1000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 16,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 8192,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "training_files": "data/train_filelist.txt",
    "validation_files": "data/val_filelist.txt",
    "text_cleaners": ["dagbani_cleaners"],
    "max_wav_value": 32768.0,
    "sampling_rate": 24000,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 1,
    "cleaned_text": true
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 0
  }
}
```

---

## 4. Multi-GPU Distributed Training Launch

Launch training using PyTorch Distributed Data Parallel (DDP):

```bash
python -m torch.distributed.run --nproc_per_node=2 train.py \
    -c configs/dagbani_vits_24k.json \
    -m dagbani_vits_run01
```

### Multi-Speaker Setup:
For multi-speaker training (e.g. Common Voice with 20+ speakers):
1. Set `"n_speakers": 25` and `"gin_channels": 256` in `configs/dagbani_vits_24k.json`.
2. Map speaker IDs in `filelist.txt` (`0` to `24`).
3. Speaker embedding vectors condition both the normalizing flow prior and HiFi-GAN generator blocks.

---

## 5. Monitoring & Checkpoint Validation

Monitor training metrics via TensorBoard:
```bash
tensorboard --logdir logs/dagbani_vits_run01/ --port 6006
```

Key indicators of healthy convergence:
- **`loss_gen_all`**: Drops smoothly from ~35.0 to $< 18.0$.
- **`loss_kl`**: Stabilizes between $1.0$ and $2.5$.
- **`loss_dur`**: Drops below $1.2$ as alignment locks in around 10k steps.
- **Audio evaluation tabs**: MAS alignment matrices show sharp monotonic diagonal trajectories.
