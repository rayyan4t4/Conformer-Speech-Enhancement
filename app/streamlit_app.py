"""
Conformer Speech Enhancement — Streamlit App
=============================================
Free-tier friendly demo app for Streamlit Community Cloud.

Features:
  - Upload an audio file (wav/mp3/flac/ogg)
  - Record live from the microphone
  - Try bundled example noisy clips
  - Enhance with the trained Conformer model
  - Compare waveforms/spectrograms and download the result

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import io

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Allow running as `streamlit run app/streamlit_app.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.inference import SpeechEnhancer, SAMPLE_RATE  # noqa: E402

st.set_page_config(
    page_title="Conformer Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
)

CHECKPOINT_PATH = os.environ.get(
    "CONFORMER_SE_CHECKPOINT", "checkpoints/conformer_se_best.pt"
)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")


# --------------------------------------------------------------------------- #
#  Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading Conformer model...")
def load_model(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return None
    return SpeechEnhancer(checkpoint_path=checkpoint_path, device="cpu")


def read_uploaded_audio(uploaded_bytes: bytes):
    audio, sr = sf.read(io.BytesIO(uploaded_bytes), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def plot_waveform_and_spectrogram(noisy, enhanced, sr=SAMPLE_RATE):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    axes[0, 0].set_title("Noisy — Waveform")
    axes[0, 0].plot(np.linspace(0, len(noisy) / sr, len(noisy)), noisy, linewidth=0.5, color="#d62728")
    axes[0, 0].set_xlabel("Time (s)")

    axes[0, 1].set_title("Enhanced — Waveform")
    axes[0, 1].plot(
        np.linspace(0, len(enhanced) / sr, len(enhanced)), enhanced, linewidth=0.5, color="#2ca02c"
    )
    axes[0, 1].set_xlabel("Time (s)")

    noisy_db = librosa.amplitude_to_db(np.abs(librosa.stft(noisy)), ref=np.max)
    img1 = librosa.display.specshow(noisy_db, sr=sr, x_axis="time", y_axis="hz", ax=axes[1, 0], cmap="magma")
    axes[1, 0].set_title("Noisy — Spectrogram")

    enh_db = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
    img2 = librosa.display.specshow(enh_db, sr=sr, x_axis="time", y_axis="hz", ax=axes[1, 1], cmap="magma")
    axes[1, 1].set_title("Enhanced — Spectrogram")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🎙️ Conformer SE")
st.sidebar.markdown(
    "Real-time-style **speech enhancement** powered by a Conformer "
    "time-frequency masking network, trained on VoiceBank+DEMAND."
)
st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "Choose input method",
    ["📁 Upload audio", "🎤 Record from microphone", "🧪 Try an example"],
)

chunk_seconds = st.sidebar.slider(
    "Processing chunk size (seconds)",
    min_value=2.0,
    max_value=16.0,
    value=8.0,
    step=1.0,
    help="Long recordings are processed in overlapping chunks to keep memory "
    "usage low on free hosting tiers.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model runs on CPU for free-tier compatibility. "
    "16 kHz mono input is assumed; other formats are auto-resampled."
)

# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
st.title("Conformer-Based Speech Enhancement")
st.markdown(
    "Upload noisy speech, record from your microphone, or try an example clip — "
    "the Conformer model will remove background noise and enhance speech clarity."
)

model = load_model(CHECKPOINT_PATH)
if model is None:
    st.warning(
        f"⚠️ No trained checkpoint found at `{CHECKPOINT_PATH}`. "
        "Train the model using the provided Colab notebook, then place "
        "`conformer_se_best.pt` inside the `checkpoints/` folder "
        "(or set the `CONFORMER_SE_CHECKPOINT` environment variable)."
    )

noisy_audio = None

if input_mode == "📁 Upload audio":
    uploaded_file = st.file_uploader(
        "Upload a noisy audio file", type=["wav", "mp3", "flac", "ogg", "m4a"]
    )
    if uploaded_file is not None:
        noisy_audio = read_uploaded_audio(uploaded_file.read())

elif input_mode == "🎤 Record from microphone":
    audio_value = st.audio_input("Record noisy speech")
    if audio_value is not None:
        noisy_audio = read_uploaded_audio(audio_value.read())

elif input_mode == "🧪 Try an example":
    if os.path.isdir(EXAMPLES_DIR):
        examples = sorted(
            f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith((".wav", ".flac"))
        )
    else:
        examples = []

    if not examples:
        st.info(
            "No bundled example clips found. Add `.wav` files to `assets/examples/` "
            "in the repository to enable this mode."
        )
    else:
        chosen = st.selectbox("Pick an example noisy clip", examples)
        if chosen:
            noisy_audio = read_uploaded_audio(
                open(os.path.join(EXAMPLES_DIR, chosen), "rb").read()
            )

# --------------------------------------------------------------------------- #
#  Run inference
# --------------------------------------------------------------------------- #
if noisy_audio is not None:
    st.subheader("🔊 Original (Noisy) Audio")
    st.audio(audio_to_wav_bytes(noisy_audio), format="audio/wav")

    if model is not None:
        run = st.button("✨ Enhance Speech", type="primary", use_container_width=True)
        if run:
            with st.spinner("Enhancing audio with the Conformer model..."):
                t0 = time.time()
                enhanced_audio = model.enhance_long(
                    noisy_audio, sr=SAMPLE_RATE, chunk_seconds=chunk_seconds
                )
                elapsed = time.time() - t0

            st.success(f"Done in {elapsed:.2f}s (real-time factor: {elapsed / (len(noisy_audio) / SAMPLE_RATE):.2f}x)")

            st.subheader("✨ Enhanced Audio")
            enhanced_bytes = audio_to_wav_bytes(enhanced_audio)
            st.audio(enhanced_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download enhanced audio",
                data=enhanced_bytes,
                file_name="enhanced_speech.wav",
                mime="audio/wav",
            )

            st.subheader("📊 Waveform & Spectrogram Comparison")
            fig = plot_waveform_and_spectrogram(noisy_audio, enhanced_audio)
            st.pyplot(fig)
    else:
        st.info("Upload a trained checkpoint to enable enhancement.")
else:
    st.info("👆 Choose an input method above to get started.")

st.markdown("---")
st.caption(
    "Conformer Speech Enhancement · Built with PyTorch + Streamlit · "
    "Trained on VoiceBank+DEMAND · Free-tier deployable"
)
