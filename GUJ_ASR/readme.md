# 🗣️ Gujarati ASR — GUJ_ASR

**Part of:** [ASR_IIIT-H](https://github.com/1nisharg/ASR_IIIT-H) — Multi-Language ASR Export Pipeline  
**Language:** Gujarati (`gu`)  
**Architecture:** Streaming Zipformer RNN-T (k2 / icefall)  
**Source:** SPRING-INX / IIT Madras ASR Group

---

## 📁 Folder Contents

```
GUJ_ASR/
├── encoder.ptl        # Streaming Zipformer encoder (PyTorch Mobile Lite, ~280MB)
├── decoder.ptl        # RNN-T decoder (PyTorch Mobile Lite)
├── joiner.ptl         # RNN-T joiner (PyTorch Mobile Lite)
└── init_states.pt     # 98 encoder initial state tensors
```

> All `.ptl` files are tracked via **Git LFS**.

---

## 🏗️ Architecture at a Glance

```
Audio (16kHz) → Kaldi FBank (80-dim) → Chunked [77 frames]
      │
      ▼
  encoder.ptl   [1, 77, 80] + 98 states → [1, 16, 512] + 98 new states
      │
      ▼
  decoder.ptl   [1, 2] (last 2 tokens) → [1, 1, 512]
      │
      ▼
  joiner.ptl    enc[1,512] + dec[1,512] → logits[1, 900]
      │
      ▼
  Greedy Search → Token IDs → Gujarati Text
```

| Parameter | Value |
|---|---|
| Sample Rate | 16000 Hz |
| Mel Bins | 80 |
| Frame Length | 25ms |
| Frame Shift | 10ms |
| Input Chunk Frames | 77 |
| Encoder Output Frames | 16 per chunk |
| Encoder Dim | 512 |
| Decoder Context Size | 2 |
| Vocabulary Size | 900 BPE tokens |
| Blank Token ID | 0 |
| Encoder State Count | 98 |

---

## ⚠️ Critical Notes

### ❌ DO NOT apply CMVN / mean normalization to mel features
This is the single most important thing. The model was trained on **raw Kaldi FBank features with no normalization**. Applying mean subtraction or standardization will cause the model to output only blanks.

```python
# ✅ CORRECT
mel = kaldi.fbank(waveform, num_mel_bins=80, ...)
# use mel as-is — no normalization

# ❌ WRONG — will break everything
mel = mel - mel.mean()
```

### ✅ Carry encoder states between chunks
The encoder is stateful. Feed the 98 output states of chunk `n` as input states to chunk `n+1`. Reset only between utterances.

### ✅ Decoder — always `need_pad=False`
```python
dec_out = decoder(y, torch.tensor(False))
```

### ✅ Joiner — always `project_input=True`, inputs must be 2D `[B, 512]`
```python
logits = joiner(enc_frame, dec_frame, True)
```

### ✅ Init states — load from `init_states.pt`
The `.ptl` files cannot call `get_init_states()` after mobile optimization. Always load from the saved file:
```python
state_dict = torch.load("init_states.pt")
states = [state_dict[f"state_{i}"] for i in range(98)]
```

---

## 🐍 Python Inference (Testing / Raspberry Pi)

### Install
```bash
pip install torch torchaudio
```

### Run
```python
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch

# Load models
encoder = torch.jit.load("encoder.ptl", map_location="cpu")
decoder = torch.jit.load("decoder.ptl", map_location="cpu")
joiner  = torch.jit.load("joiner.ptl",  map_location="cpu")
encoder.eval(); decoder.eval(); joiner.eval()

# Load vocab
with open("SPRING_INX_Gujarati_tokens.txt") as f:
    tokens = [line.strip().split()[0] for line in f]

# Load init states
state_dict = torch.load("init_states.pt")
states     = [state_dict[f"state_{i}"] for i in range(98)]

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Kaldi FBank — NO normalization
    mel = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25.0,
                      frame_shift=10.0, high_freq=8000, low_freq=20,
                      sample_frequency=16000, use_energy=False)

    # Chunk into 77-frame pieces
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
            enc_out  = enc_out_tuple[0]
            states[:]= list(enc_out_tuple[2:])

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

print(transcribe("your_gujarati_audio.wav"))
```

---

## 🤖 Android Setup

### `build.gradle.kts`
```kotlin
implementation("org.pytorch:pytorch_android_lite:1.13.1")
```

### Assets — place in `app/src/main/assets/`
```
encoder.ptl
decoder.ptl
joiner.ptl
init_states.pt
SPRING_INX_Gujarati_tokens.txt
```

### TODO for Android developer
- [ ] Implement Kaldi FBank mel extraction in Kotlin (no CMVN)
- [ ] Load `init_states.pt` tensors on app start
- [ ] Wire microphone → mel → encoder → decoder → joiner → text
- [ ] See `GujaratiASR.kt` for the inference skeleton

---

## ✅ Verified Results

| Test | Result |
|---|---|
| Source model | `SPRING_INX_streaming_k2_Gujarati.pt` |
| Test audio | `guj.wav` (3.07s Gujarati speech) |
| Transcript | `કેમ છો` |
| PTL vs original max diff | `0.000000` ✅ |
| Platform tested | Kaggle CPU (Python) |
