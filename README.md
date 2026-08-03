# ASR_IIIT-H — Edge Quantization of SPRING-INX Streaming ASR Models

Mobile-ready (`.ptl`) conversions of [SPRING Lab, IIT Madras](https://asr.iitm.ac.in/) **SPRING-INX** streaming ASR models for on-device inference on **Android** and **Raspberry Pi**.

Each model is a streaming **Zipformer + RNN-T (Transducer)** built with the k2/icefall toolkit. This repository takes the original TorchScript `.pt` checkpoints and re-packages them into the **PyTorch Mobile Lite Interpreter** format (`.ptl`), split into three independently loadable modules (encoder / decoder / joiner) plus the encoder's initial streaming state — so the runtime never needs the original multi-hundred-MB `.pt`.

---

## Supported Languages

| Language | Folder | Model vocab | Tokens file | Status |
|----------|--------|:-----------:|:-----------:|:------:|
| Hindi    | `HINDI_ASR/`  | —   | `SPRING_INX_Hindi_tokens.txt`   | ✅ |
| Gujarati | `GUJ_ASR/`    | 900 | `SPRING_INX_Gujarati_tokens.txt`| ✅ |
| Odia     | `ODIA-ASR/`   | 300 | `SPRING_INX_Odia_tokens.txt`    | ✅ |
| Telugu   | `TELUGU-ASR/` | 600 | `SPRING_INX_Telugu_tokens.txt`  | ✅ |

> **Note on vocab size:** the *model* vocab (emission classes) is smaller than the *tokens file* length. SPRING-INX tokens files append k2 lexicon **disambiguation symbols** (`#0`, `#1`, `#2`) that the acoustic model never emits. The count of these symbols varies per language (e.g. Odia appends 2 → 303 file / 300 model; Telugu appends 2 → 602 file / 600 model). Always treat the **joiner output dimension as authoritative**, and exclude `#`-prefixed tokens at decode time.

---

## Architecture

All SPRING-INX streaming models share the same structure. The original `.pt` is a single JIT-scripted `AsrModel` exposing six submodules:

```
AsrModel
├── encoder_embed   (Conv2dSubsampling)      # conv front-end + streaming cache
├── encoder         (StreamingEncoderModel)  # Zipformer, 16 layers, 512-dim
├── decoder         (Decoder)                # RNN-T predictor (stateless, context_size=2)
├── joiner          (Joiner)                 # encoder ⊕ decoder → vocab logits
├── simple_am_proj  (Linear)                 # auxiliary — training only
└── simple_lm_proj  (Linear)                 # auxiliary — training only
```

**Shared constants (verified identical across all four languages):**

| Constant | Value | Meaning |
|----------|:-----:|---------|
| `chunk_size` | 32 | encoder frames processed per streaming step |
| `left_context_len` | 128 | attention left-context window |
| `context_size` | 2 | decoder history tokens |
| `encoder_dim` | 512 | encoder / joiner hidden dim |
| `feat_dim` | 80 | Kaldi mel filterbanks |
| `RAW_FRAMES` | 77 | raw feature frames per chunk (`chunk_size*2 + 13`) |
| `N_STATES` | 98 | streaming state tensors carried between chunks |

**Streaming state contract (98 tensors):**
- `states[0:96]` — Zipformer layer caches (6 tensors per block)
- `states[96]`   — conv embed cache `[1, 128, 3, 19]` (float32)
- `states[97]`   — `processed_lens` `[1]` (int32)

The blank token is `<blk>` at index **0**.

---

## Repository Layout

```
ASR_IIIT-H/
├── HINDI_ASR/
├── GUJ_ASR/
├── ODIA-ASR/
├── TELUGU-ASR/
│   ├── encoder.ptl                     # streaming Zipformer  (~261 MB)
│   ├── decoder.ptl                     # RNN-T predictor      (~1 MB)
│   ├── joiner.ptl                      # joint network        (~3 MB)
│   ├── init_states.pt                  # 98 initial encoder states
│   └── SPRING_INX_<Lang>_tokens.txt    # token → text lookup
├── .gitattributes                      # Git LFS tracking for large .ptl
└── README.md
```

> **Large files:** `encoder.ptl` is ~261 MB — above GitHub's 100 MB limit. These are tracked with **Git LFS** (see `.gitattributes`). Clone with LFS installed:
> ```bash
> git lfs install
> git clone https://github.com/1nisharg/ASR_IIIT-H.git
> ```

---

## Inference

Each `.ptl` loads standalone via the PyTorch Mobile Lite Interpreter. The decode loop is greedy RNN-T: run the encoder once per 77-frame chunk, then step the decoder + joiner per output frame.

```python
import torch, torchaudio
import torchaudio.compliance.kaldi as kaldi

LANG_DIR   = "TELUGU-ASR"
RAW_FRAMES, FEAT_DIM = 77, 80

encoder = torch.jit.load(f"{LANG_DIR}/encoder.ptl", map_location="cpu").eval()
decoder = torch.jit.load(f"{LANG_DIR}/decoder.ptl", map_location="cpu").eval()
joiner  = torch.jit.load(f"{LANG_DIR}/joiner.ptl",  map_location="cpu").eval()
states  = list(torch.load(f"{LANG_DIR}/init_states.pt"))

with open(f"{LANG_DIR}/SPRING_INX_Telugu_tokens.txt", encoding="utf-8") as f:
    tokens = [ln.strip().split()[0] for ln in f if ln.strip()]
EXCLUDE = {"<blk>", "<eps>", "<blank>", "<unk>", "<sos/eos>", "#0", "#1", "#2"}

def transcribe(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)

    mel = kaldi.fbank(wav, num_mel_bins=FEAT_DIM, frame_length=25.0, frame_shift=10.0,
                      high_freq=8000, low_freq=20, sample_frequency=16000, use_energy=False)
    T = mel.shape[0]; pad = (-T) % RAW_FRAMES
    if pad: mel = torch.cat([mel, torch.zeros(pad, FEAT_DIM)], 0)
    chunks = mel.reshape(-1, RAW_FRAMES, FEAT_DIM)

    st, hyp = list(states), [0, 0]
    with torch.no_grad():
        for chunk in chunks:
            out = encoder(chunk.unsqueeze(0),
                          torch.tensor([RAW_FRAMES], dtype=torch.int32), *st)
            enc, st = out[0], list(out[2:])
            for t in range(enc.shape[1]):
                y = torch.tensor([[hyp[-2], hyp[-1]]], dtype=torch.int64)
                dec = decoder(y, torch.tensor(False))          # need_pad=False
                logits = joiner(enc[:, t, :], dec[:, 0, :], True)  # project_input=True
                tok = logits[0].argmax().item()
                if tok != 0: hyp.append(tok)                   # 0 = blank

    return "".join(tokens[t].replace("▁", " ") for t in hyp[2:]
                   if 0 <= t < len(tokens) and tokens[t] not in EXCLUDE).strip()

print(transcribe("sample.wav"))
```

**Audio requirements:** 16 kHz, mono, Kaldi fbank features (80 mel bins, 25 ms / 10 ms), **no CMVN** — matching the original training pipeline.

---

## Conversion Pipeline

The `.pt → .ptl` flow used for every language:

1. **Load & introspect** the JIT-scripted `.pt`; derive all constants from the model (never hardcoded across languages).
2. **Reconcile vocab** — take the joiner output dim as the true vocab; confirm the extra tokens-file entries are `#`-disambig symbols only.
3. **Trace the encoder** via a thin `nn.Module` wrapper that flattens the 98-state list into positional args (list-indexing isn't traceable inside `forward`).
4. **Trace decoder & joiner** directly, preserving the `(y, need_pad)` and `(enc, dec, project_input)` signatures with tensor-valued flags.
5. **Optimize for mobile** (`optimize_for_mobile`) and save via `_save_for_lite_interpreter()`.
6. **Save `init_states.pt`** so deployment never loads the original `.pt`.
7. **Validate** by reloading the `.ptl` files from disk and running a full transcription.

Every stage is verified numerically — encoder, decoder, and joiner traces all reproduce the original outputs to `max_diff = 0.0`.

---

## Acknowledgements

Original models by **SPRING Lab, IIT Madras** ([asr.iitm.ac.in](https://asr.iitm.ac.in/)), released under the **National Language Translation Mission (NLTM)**, funded by MeitY, Government of India. This repository provides only edge-deployment conversions of those publicly released checkpoints; all model credit belongs to SPRING Lab.

**Original data / models:** [Speech-Lab-IITM](https://github.com/Speech-Lab-IITM) · SPRING-INX corpus (arXiv:2310.14654)

---

## License

Conversion scripts and packaging in this repository follow the license terms of the upstream SPRING-INX release (CC BY 4.0). Please cite SPRING Lab, IIT Madras when using these models.
