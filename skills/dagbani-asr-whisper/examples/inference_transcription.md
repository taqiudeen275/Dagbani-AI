# Dagbani Whisper ASR Inference & Transcription Examples

**Document Version**: 1.0.0  
**Supported Engines**: HuggingFace Transformers (PyTorch), `faster-whisper` (CTranslate2 INT8), `whisper.cpp`  

---

## 1. Python Inference with Fine-Tuned LoRA Model

### Example 1: HuggingFace Pipeline with PEFT Adapter
```python
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

base_model_id = "openai/whisper-medium"
adapter_id = "ats-tech/whisper-dagbani-medium-lora"

# 1. Load Processor and Base Model
processor = WhisperProcessor.from_pretrained(base_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. Load Fine-Tuned LoRA Adapter
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

# 3. Ingest and Resample Audio
audio_path = "sample_dagbani.wav"
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# 4. Extract Log-Mel Spectrogram
input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features.to("cuda", dtype=torch.float16)

# 5. Autoregressively Generate Transcription
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        max_length=225,
        forced_decoder_ids=None,
        task="transcribe",
        language=None,
        no_speech_threshold=0.6
    )

# 6. Decode into Dagbani Text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
# Output: "Ti kpalinʒoo n-nyɛla din viɛli pam zaŋ ti Dagbamba zuliya."
```

---

## 2. High-Speed Inference with `faster-whisper` (CTranslate2)

For production deployment and low-latency microservices ($4\times$ speedup, $70\%$ lower memory footprint):

```python
from faster_whisper import WhisperModel

# Load converted INT8 model on GPU or CPU
model = WhisperModel(
    "ats-tech/faster-whisper-dagbani-medium",
    device="cuda",
    compute_type="int8_float16"
)

# Transcribe with Beam Search
segments, info = model.transcribe(
    "interview_audio.wav",
    beam_size=5,
    language=None,
    task="transcribe",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=400)
)

print(f"Detected Duration: {info.duration:.2f}s")
for segment in segments:
    print(f"[{segment.start:05.2f}s -> {segment.end:05.2f}s] {segment.text}")
```

---

## 3. Command-Line Inference Utility

```bash
# Transcribe single audio file via CLI
python skills/dagbani-asr-whisper/scripts/evaluate_asr.py \
  --audio-file ./speech_sample.wav \
  --model-checkpoint ./checkpoints/whisper-dagbani-medium-lora

# Batch evaluate an entire test set directory
python skills/dagbani-asr-whisper/scripts/evaluate_asr.py \
  --test-manifest ./test_manifest.json \
  --output-report ./benchmark_results.json
```
