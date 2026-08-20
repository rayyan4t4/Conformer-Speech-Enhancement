# 🎙️ Conformer Speech Enhancement

A production-oriented, **fully free-tier deployable** speech enhancement system built around a
**Conformer** encoder that predicts a complex ratio mask (cRM) for single-channel noise
suppression. Includes:

- 🧠 A complete PyTorch **Conformer model** (medium config, ~12.4M params)
- 📓 A **single Colab notebook** that downloads the dataset and trains the model end-to-end
- 🖥️ A **Streamlit web app** (upload / mic recording / example clips) ready for Streamlit
  Community Cloud
- 📄 An **IEEE-format LaTeX research paper** describing the method
- ✅ Tested, modular code: `model/`, `utils/`, `app/`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Training on Google Colab](#training-on-google-colab)
5. [Running the Streamlit App Locally](#running-the-streamlit-app-locally)
6. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
7. [Dataset](#dataset)
8. [Model Details](#model-details)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Research Paper](#research-paper)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Architecture Overview

```
Noisy waveform
      │
      ▼
   STFT (n_fft=512, hop=128, 16kHz)
      │
      ▼
 log1p(|STFT|)  ──────────────►  Conformer Encoder (8 blocks, d_model=256)
                                        │
                                        ▼
                          Linear head → complex ratio mask (real, imag)
                                        │
                                        ▼
                       Mask applied to noisy complex STFT
                                        │
                                        ▼
                                    ISTFT
                                        │
                                        ▼
                              Enhanced waveform
```

Each Conformer block follows the standard macaron structure:
`FeedForward → Multi-Head Self-Attention → Convolution Module → FeedForward → LayerNorm`,
combining the local modeling strength of convolutions with the long-range context of
self-attention — well suited to speech, which has both fine-grained (phoneme-level) and
long-range (prosody-level) structure.

---

## Project Structure

```
conformer-speech-enhancement/
├── app/
│   └── streamlit_app.py          # Streamlit web app (upload / mic / examples)
├── model/
│   ├── conformer.py               # Conformer encoder + speech enhancement head
│   ├── dataset.py                 # VoiceBank+DEMAND dataset loader
│   ├── inference.py                # Inference wrapper (chunked, overlap-add)
│   └── losses.py                   # SI-SDR + spectral losses
├── utils/
│   └── audio.py                    # STFT/ISTFT, I/O, feature helpers
├── notebook/
│   └── Conformer_Speech_Enhancement_Training.ipynb   # Full Colab training pipeline
├── paper/
│   ├── paper.tex                   # IEEE-format research paper (LaTeX source)
│   └── paper.pdf                   # Compiled paper
├── assets/
│   └── examples/                   # Bundled example noisy clips for the demo
├── checkpoints/                    # Trained model weights go here (.pt)
├── requirements.txt                 # App dependencies (CPU-only, free-tier friendly)
├── packages.txt                     # System packages for Streamlit Cloud (apt)
├── .streamlit/config.toml           # Streamlit theming
└── README.md
```

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd conformer-speech-enhancement

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (see below) OR use your own checkpoint at
#    checkpoints/conformer_se_best.pt

# 5. Run the app
streamlit run app/streamlit_app.py
```

---

## Training on Google Colab

All training — **including dataset download** — happens in a single notebook:
`notebook/Conformer_Speech_Enhancement_Training.ipynb`.

1. Upload this project as a `.zip` to Colab (or push to GitHub and `git clone` it — both
   options are in the notebook).
2. Open the notebook in Google Colab.
3. `Runtime → Change runtime type → T4 GPU` (free tier).
4. Run all cells top-to-bottom. The notebook will:
   - Install dependencies
   - Download **VoiceBank+DEMAND** directly from the University of Edinburgh DataShare
     (with an automatic HuggingFace mirror fallback)
   - Build the medium Conformer model
   - Train with **mixed precision**, **gradient clipping**, and a **cosine LR schedule**
   - **Checkpoint to Google Drive** every N steps so training survives Colab disconnects,
     and **automatically resumes** on re-run
   - Evaluate with **PESQ**, **STOI**, and **SI-SDR**
   - Export `conformer_se_best.pt` and re-package the project as a ready-to-deploy zip

Training the medium config for 60 epochs on VoiceBank+DEMAND takes roughly **6–10 hours**
on a free T4 — comfortably splittable across multiple free Colab sessions thanks to
checkpoint resuming.

### Adjusting for your compute budget

All hyperparameters live in the `CONFIG` dict in the notebook. To train faster (at some
quality cost), reduce `num_layers` (e.g. 6), `d_model` (e.g. 192), or `epochs`.

---

## Running the Streamlit App Locally

```bash
streamlit run app/streamlit_app.py
```

The app looks for a checkpoint at `checkpoints/conformer_se_best.pt` by default. Override
with an environment variable if needed:

```bash
export CONFORMER_SE_CHECKPOINT=/path/to/your_checkpoint.pt
streamlit run app/streamlit_app.py
```

**Features:**
- 📁 Upload an audio file (wav/mp3/flac/ogg/m4a)
- 🎤 Record directly from your microphone (via `st.audio_input`)
- 🧪 Try bundled synthetic example clips (drop your own into `assets/examples/`)
- 📊 Side-by-side waveform + spectrogram comparison
- ⬇️ Download the enhanced audio

The app processes audio in **overlapping chunks** with cross-fade blending, so it stays
memory-friendly on free hosting tiers even for long recordings, and runs entirely on **CPU**
(no GPU needed for inference).

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Make sure your trained checkpoint is at `checkpoints/conformer_se_best.pt` and committed
   (use [Git LFS](https://git-lfs.github.com/) if it exceeds GitHub's 100MB file limit — the
   medium model checkpoint is typically ~50MB, so plain Git usually works fine).
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click
   **"New app"**.
4. Point it at your repo, branch, and set the main file path to `app/streamlit_app.py`.
5. Deploy — Streamlit Cloud will automatically install `requirements.txt` and `packages.txt`.

Everything used (Streamlit Cloud hosting, CPU PyTorch, the dataset, and the notebook's free
Colab GPU) is free of charge.

---

## Dataset

**VoiceBank+DEMAND** (Valentini-Botinhao et al., 2016) — the standard benchmark corpus for
single-channel speech enhancement:

- 28 speakers, ~11,572 training utterances (clean + noisy pairs at multiple SNRs: 0, 5, 10,
  15 dB), mixed with 10 real-world DEMAND noise types
- 2 held-out speakers, 824 test utterances at SNRs of 2.5, 7.5, 12.5, 17.5 dB with 5 unseen
  noise types
- Distributed freely (CC license) via the University of Edinburgh DataShare — no
  registration or payment required
- Downloaded automatically by the training notebook (~2.5 GB)

---

## Model Details

| Hyperparameter        | Value              |
|------------------------|--------------------|
| Sample rate            | 16 kHz             |
| STFT window / hop      | 512 / 128 samples  |
| Conformer layers       | 8                  |
| Model dimension        | 256                |
| Attention heads        | 4                  |
| Conv kernel size       | 31                 |
| Feed-forward expansion | 4×                 |
| Mask type              | Complex ratio mask (bounded tanh, K=3) |
| Parameters             | ~12.4M             |
| Loss                   | SI-SDR (time) + L1 spectral (magnitude + real/imag) |

This configuration is deliberately sized to train comfortably on a **free Colab T4 GPU**
while giving noticeably better quality than a "small/fast" configuration.

---

## Evaluation Metrics

The notebook reports standard objective speech-quality metrics on the VoiceBank+DEMAND test
set:

- **PESQ** (Perceptual Evaluation of Speech Quality, wideband)
- **STOI** (Short-Time Objective Intelligibility)
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio, dB)

---

## Research Paper

An IEEE-format LaTeX paper describing the method, architecture, training setup, and results
is provided in `paper/paper.tex` (with a compiled `paper/paper.pdf`). To recompile:

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex   # run twice for references/TOC
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: No clean wav files found` | Run the dataset-download cell in the notebook before training. |
| Colab disconnects mid-training | Just re-run the notebook — training resumes automatically from the last Google Drive checkpoint. |
| Streamlit app shows "No trained checkpoint found" | Copy `conformer_se_best.pt` into `checkpoints/`, or set `CONFORMER_SE_CHECKPOINT`. |
| Out-of-memory on Colab | Lower `batch_size` or `segment_seconds` in the notebook `CONFIG` dict. |
| Slow inference on Streamlit Cloud | Increase the sidebar "processing chunk size"; CPU inference is real-time-ish but not instant for very long files. |

---

## License

MIT License — see `LICENSE`. VoiceBank+DEMAND is distributed under its own license terms by
the University of Edinburgh; please review before commercial use.
