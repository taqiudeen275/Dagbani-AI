## 2026-08-14T20:51:57Z
You are an authoritative Speech Recognition (ASR) & Whisper Architecture Spec Miner for Dagbani.
Your working directory is: d:/ATS Tech/Dagbani AI/.agents/explorer_asr/
Project Root: d:/ATS Tech/Dagbani AI
Read ORIGINAL_REQUEST.md: d:/ATS Tech/Dagbani AI/ORIGINAL_REQUEST.md

Your mission:
Conduct an exhaustive, forensic exploration and specification extraction of Dagbani ASR architectures, Whisper fine-tuning pipelines, and speech datasets from the authoritative sources in the project:
- `resources/dagbani_whisper_asr_colab_v2.ipynb`
- `resources/APSIPA2025_P208.pdf`
- `resources/35_breaking_the_low_resource_barr.pdf`
- `research/Building Dagbani Language Model and TTS.pdf`
- `research/research_source.txt`

Specifically extract and document in full detail:
1. Audio & Preprocessing Pipeline:
   - Audio specifications (sampling rate 16kHz, mono, PCM 16-bit, bitrates).
   - Feature extraction: 80-channel / 128-channel log-mel spectrograms, STFT window/hop sizes (25ms/10ms).
   - Noise handling, normalization, Voice Activity Detection (VAD), chunking/segmentation strategies for long audio.
   - SpecAugment parameters (time masking, frequency masking) and data augmentation for low-resource speech.
2. Whisper Model Architecture & Fine-Tuning Recipes:
   - Pretrained checkpoints (Whisper tiny, base, small, medium, large-v3).
   - Low-Resource adaptation techniques: LoRA (rank, alpha, target modules: q_proj, v_proj, out_proj, fc1, fc2, k_proj), QLoRA 4-bit/8-bit quantization.
   - Tokenizer & Language ID adaptation for Dagbani (handling unrepresented BGL characters like ŋ, ɛ, ɔ, ɣ, ʒ, token vocabulary extension vs ASCII mapping).
   - Hyperparameters: Learning rate schedule (cosine with warmup, peak lr 1e-4 - 1e-5), effective batch size (gradient accumulation steps), weight decay, mixed precision (fp16/bf16).
3. Training & Evaluation Pipeline:
   - Detailed analysis of the Colab v2 notebook (`dagbani_whisper_asr_colab_v2.ipynb`): all cell logic, dependencies, training loops, HuggingFace Seq2SeqTrainer setup.
   - Evaluation metrics: Word Error Rate (WER), Character Error Rate (CER), normalized WER (ignoring punctuation/case vs exact orthography).
   - Common failure modes: Hallucinations in silent chunks, tone confusion, dialectal variation (Tomosili / Western vs Katindu / Eastern Dagbani), code-switching with English.
   - Inference & Deployment: ONNX export, Whisper.cpp / faster-whisper optimizations, streaming ASR considerations.

Output your comprehensive findings and specification report to:
`d:/ATS Tech/Dagbani AI/.agents/explorer_asr/survey_asr_report.md`
Also create `handoff.md` and update `progress.md` in your folder. Send a completion message back when done.
