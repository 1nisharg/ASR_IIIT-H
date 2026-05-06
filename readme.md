# 🗣️ Gujarati ASR — SPRING-INX Streaming Model
### Edge Deployment Pipeline (Android / Raspberry Pi)

---

## 📌 Project Overview

This project exports the **SPRING-INX Gujarati Streaming ASR model** (originally a monolithic TorchScript `.pt` file) into three separate inference modules optimized for edge devices. The architecture is a **streaming RNN-T Transducer** with a **Zipformer encoder**, designed for real-time chunk-by-chunk speech recognition.

**Source Model:** `SPRING_INX_streaming_k2_Gujarati.pt`  
**Target Platforms:** Android (PyTorch Mobile), Raspberry Pi (PyTorch CPU)  
**Language:** Gujarati (`gu`)  
**Vocabulary Size:** 900 tokens (BPE)

---

## 📁 Repository Structure

```
gujarati-asr/
│
├── models/                          # Exported model files
│   ├── encoder.ptl                  # Encoder — PyTorch Mobile Lite
│   ├── decoder.ptl                  # Decoder — PyTorch Mobile Lite
│   ├── joiner.ptl                   # Joiner  — PyTorch Mobile Lite
│   └── init_states.pt               # Encoder initial states (98 tensors)
│
├── tokens/
│   └── SPRING_INX_Gujarati_tokens.txt   # 903-entry BPE vocabulary
│
├── notebooks/
│   └── gujarati_asr_export.ipynb    # Full export + verification notebook (Kaggle)
│
├── android/
│   └── GujaratiASR.kt               # Android inference class (PyTorch Mobile)
│
├── python/
│   └── transcribe.py                # Python inference script (Raspberry Pi / testing)
│
└── README.md
```

---

## 🏗️ Model Architecture

```
Audio (16kHz PCM)
      │
      ▼
  Kaldi FBank (80-dim mel, 25ms window, 10ms shift)
      │
      ▼
  encoder_embed  [B, 77, 80] → [B, 32, 192]
      │
      ▼
  Streaming Zipformer Encoder
      │  Input:  [B, 77, 80]  + 98 state tensors
      │  Output: [B, 16, 512] + 98 new state tensors
      ▼
  Decoder (Embedding + Linear)
      │  Input:  [B, 2]  (last 2 token context)
      │  Output: [B, 1, 512]
      ▼
  Joiner (project + tanh + linear)
      │  Input:  enc [B, 512] + dec [B, 512]
      │  Output: logits [B, 900]
      ▼
  Greedy / Beam Search → Token IDs → Text
```

---

## ⚙️ Architecture Constants

| Parameter | Value |
|---|---|
| Sample Rate | 16000 Hz |
| Mel Filterbanks | 80 |
| Frame Length | 25ms |
| Frame Shift | 10ms |
| Chunk Frames (input) | 77 |
| Encoder Output Frames | 16 per chunk |
| Encoder Dim | 512 |
| Decoder Context Size | 2 |
| Vocabulary Size | 900 |
| Blank Token ID | 0 |
| Number of Encoder States | 98 |

---

## ⚠️ Critical Notes for Future Developers

### 1. Mel Feature Extraction — NO CMVN
The model was trained with **raw Kaldi FBank features — no CMVN (mean normalization)**. Applying any normalization (mean subtraction, standardization) will cause the model to output only blanks. Always use:

```python
import torchaudio.compliance.kaldi as kaldi

mel = kaldi.fbank(
    waveform,
    num_mel_bins=80,
    frame_length=25.0,
    frame_shift=10.0,
    high_freq=8000,
    low_freq=20,
    sample_frequency=16000,
    use_energy=False,
)
# DO NOT normalize mel features
```

### 2. Encoder States Must Be Carried Between Chunks
The encoder is **stateful**. The 98 state tensors output from one chunk must be fed as input to the next chunk. Resetting states mid-utterance will break transcription. Only reset states between separate utterances.

### 3. Decoder `need_pad=False`
Always call the decoder with `need_pad=False` during inference:
```python
dec_out = decoder(y, need_pad=False)   # Python
decoder(y, torch.tensor(False))        # PTL on Android
```

### 4. Joiner `project_input=True`
Always call the joiner with `project_input=True` and **2D inputs** `[B, 512]`:
```python
logits = joiner(enc_frame, dec_frame, project_input=True)
```

### 5. Init States
On Android/edge devices, load init states from `init_states.pt` — do NOT try to call `model.encoder.get_init_states()` on a `.ptl` file as submodule access is not available after mobile optimization.

---

## 🐍 Python Inference (Raspberry Pi / Testing)

### Requirements
```bash
pip install torch torchaudio
```

### Quick Test
```python
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch

# Load models
encoder = torch.jit.load("models/encoder.ptl", map_location="cpu")
decoder = torch.jit.load("models/decoder.ptl", map_location="cpu")
joiner  = torch.jit.load("models/joiner.ptl",  map_location="cpu")
encoder.eval(); decoder.eval(); joiner.eval()

# Load tokens
with open("tokens/SPRING_INX_Gujarati_tokens.txt") as f:
    tokens = [line.strip().split()[0] for line in f]

# Load init states
state_dict = torch.load("models/init_states.pt")
states     = [state_dict[f"state_{i}"] for i in range(98)]

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    mel = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25.0,
                      frame_shift=10.0, high_freq=8000, low_freq=20,
                      sample_frequency=16000, use_energy=False)

    CHUNK = 77
    pad   = (-mel.shape[0]) % CHUNK
    if pad:
        mel = torch.cat([mel, torch.zeros(pad, 80)], dim=0)
    chunks = mel.reshape(-1, CHUNK, 80)

    hyp = [0, 0]
    with torch.no_grad():
        for chunk in chunks:
            x      = chunk.unsqueeze(0)
            x_lens = torch.tensor([77], dtype=torch.int32)

            enc_out_tuple = encoder(x, x_lens, *states)
            enc_out = enc_out_tuple[0]
            states[:]  = list(enc_out_tuple[2:])

            for t in range(enc_out.shape[1]):
                y         = torch.tensor([[hyp[-2], hyp[-1]]], dtype=torch.int64)
                dec_out   = decoder(y, torch.tensor(False))
                enc_frame = enc_out[:, t, :]
                dec_frame = dec_out[:, 0, :]
                logits    = joiner(enc_frame, dec_frame, True)
                token     = logits[0].argmax().item()
                if token != 0:
                    hyp.append(token)

    text = "".join(
        tokens[t].replace("▁", " ")
        for t in hyp[2:]
        if 0 <= t < len(tokens) and
        tokens[t] not in ["<eps>","<blank>","<unk>","<sos/eos>","#0","#1","#2"]
    )
    return text.strip()

print(transcribe("your_audio.wav"))
```

---

## 🤖 Android Setup

### 1. `build.gradle.kts` dependency
```kotlin
implementation("org.pytorch:pytorch_android_lite:1.13.1")
```

### 2. Assets
Place in `app/src/main/assets/`:
- `encoder.ptl`
- `decoder.ptl`
- `joiner.ptl`
- `init_states.pt`
- `SPRING_INX_Gujarati_tokens.txt`

### 3. Inference Class
See `android/GujaratiASR.kt` for the complete inference implementation.

> **Note:** The Android Kotlin mel frontend (Kaldi FBank equivalent) still needs to be implemented. Options: use `TarsosDSP` library or implement the filterbank manually. This is the next TODO for the Android developer.

---

## 📊 Export Pipeline Summary

The export was done on Kaggle (GPU T4 x2) using the following steps:

| Step | Description | Status |
|---|---|---|
| 1 | Load original `.pt` TorchScript model | ✅ |
| 2 | Scrape architecture constants via forward pass | ✅ |
| 3 | Export `decoder.onnx` | ✅ (not used, see note) |
| 4 | Export `joiner.onnx` | ✅ (not used, see note) |
| 5 | Trace encoder with `EncoderTraceWrapper` | ✅ |
| 6 | Export all 3 as `.ptl` via PyTorch Mobile | ✅ |
| 7 | Verify pipeline produces Gujarati text | ✅ `'કેમ છો'` |

> **Note on ONNX:** Encoder ONNX export was attempted but blocked by `prim::TupleIndex` with non-constant index inside the Zipformer IR — a known limitation of both legacy and dynamo ONNX exporters with this architecture. PyTorch Mobile `.ptl` is the recommended format for this model.

---

## 🔬 Verified Test Result

**Input:** `guj.wav` (3.07s Gujarati speech)  
**Output:** `કેમ છો`  
**Pipeline:** Original model submodules → confirmed working  
**PTL Pipeline:** encoder.ptl + decoder.ptl + joiner.ptl → confirmed identical outputs  
