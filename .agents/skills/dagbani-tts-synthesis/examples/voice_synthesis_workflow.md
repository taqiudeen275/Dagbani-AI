# Voice Synthesis & Audio Export Workflow for Dagbani

This example demonstrates how to perform batch and real-time text-to-speech inference, tone-aware conditioning, and high-fidelity WAV audio export.

---

## 1. Single Sentence Synthesis via CLI

Synthesize a single Dagbani sentence into 24 kHz WAV format using `synthesize_tts.py`:

```bash
python ../scripts/synthesize_tts.py \
    --text "O biɛla Tamale ka nyɛ pukpara ŋun kpaŋsiri kpaŋkpaŋ koobu." \
    --checkpoint checkpoints/vits_dagbani_best.pt \
    --config configs/dagbani_vits_24k.json \
    --output-wav output/farming_advisory.wav \
    --speaker-id 0 \
    --length-scale 1.0 \
    --noise-scale 0.667
```

---

## 2. Programmatic Python API Synthesis

Use the Python SDK to synthesize audio in-memory for integration into web services or conversational voice bots:

```python
import torch
import soundfile as sf
from scripts.dagbani_phonemizer import DagbaniPhonemizer
from scripts.synthesize_tts import DagbaniTTSPipeline

# 1. Initialize Phonemizer & TTS Engine
phonemizer = DagbaniPhonemizer()
engine = DagbaniTTSPipeline(
    checkpoint_path="checkpoints/vits_dagbani_best.pt",
    config_path="configs/dagbani_vits_24k.json",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Text Input
raw_text = "Ti bɔrimi ni ti zaŋ Dagbanli n-ti alaafeei baŋsim zaa."

# 3. G2P & Tone Analysis
phoneme_data = phonemizer.phonemize(raw_text, return_token_ids=True)
print(f"Phonemized: {phoneme_data['ipa']}")
print(f"Token IDs:  {phoneme_data['token_ids']}")

# 4. Neural Waveform Generation
audio_tensor = engine.synthesize(
    phoneme_ids=phoneme_data["token_ids"],
    speaker_id=0,
    speed=1.0,
    noise_scale=0.667
)

# 5. Export to Audio File
engine.save_wav(audio_tensor, "output/health_advisory.wav", sample_rate=24000)
print("Saved synthesized audio to output/health_advisory.wav")
```

---

## 3. Batch Synthesis for Offline Audiobooks & IVR Prompts

Process large batches of educational, cultural, or telephony prompts from a CSV/JSON file:

```python
import json
from pathlib import Path
from scripts.dagbani_phonemizer import DagbaniPhonemizer
from scripts.synthesize_tts import DagbaniTTSPipeline

prompts = [
    {"id": "ivr_01", "text": "A ni saɣi ti, dinbɔŋɔ n-nyɛ Dagbanli kundu."},
    {"id": "ivr_02", "text": "M-bɔhimi bɔhigu kam din kpa alaafeei polo."},
    {"id": "ivr_03", "text": "Suhupiɛlli n-ti a ni kaŋa na."}
]

phonemizer = DagbaniPhonemizer()
engine = DagbaniTTSPipeline(checkpoint_path="checkpoints/vits_dagbani_best.pt")
output_dir = Path("output/ivr_prompts")
output_dir.mkdir(parents=True, exist_ok=True)

for item in prompts:
    tokens = phonemizer.phonemize(item["text"])["token_ids"]
    wav = engine.synthesize(tokens)
    out_path = output_dir / f"{item['id']}.wav"
    engine.save_wav(wav, str(out_path))
    print(f"Synthesized: {out_path}")
```

---

## 4. Synthesis Quality Assurance (MOS & Spectral Inspection)

Verify synthesized audio quality before deployment:
1. **Duration Check**: Ensure audio duration matches expected phoneme count (~100–140 ms per syllable).
2. **True Peak**: Verify True Peak is $< -1.0\text{ dBTP}$ to prevent clipping distortion on mobile speakers.
3. **Tone Intonation**: Verify natural pitch downtrend across long sentences without unnatural robotic pitch flattening.
