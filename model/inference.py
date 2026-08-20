"""
Inference wrapper for the Conformer speech enhancement model.

Handles:
  - loading a trained checkpoint
  - chunked, memory-friendly inference on arbitrarily long audio
  - overlap-add reconstruction to avoid boundary artifacts
  - CPU-friendly defaults for free-tier deployment (Streamlit Cloud)
"""

import numpy as np
import torch

from model.conformer import build_model
from utils.audio import (
    stft,
    istft,
    complex_to_log_mag,
    apply_crm,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)

DEFAULT_CONFIG = dict(
    n_fft=N_FFT,
    d_model=256,
    num_layers=8,
    num_heads=4,
    conv_kernel=31,
    ff_expansion=4,
    dropout=0.1,
    mask_bound=3.0,
)


class SpeechEnhancer:
    def __init__(self, checkpoint_path: str, device: str = None, config: dict = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or DEFAULT_CONFIG

        self.model = build_model(self.config)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def enhance(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Enhance a single-channel audio waveform.

        Args:
            audio: 1-D numpy array, float32, in range [-1, 1]
            sr: sample rate (must match training sample rate, 16 kHz)
        Returns:
            enhanced audio, same length as input
        """
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected sample rate {SAMPLE_RATE}, got {sr}. Resample before calling enhance()."
            )

        original_len = len(audio)
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)

        noisy_spec = stft(audio_t)  # (1, F, T) complex
        log_mag = complex_to_log_mag(noisy_spec)  # (1, T, F)

        mask_real, mask_imag = self.model(log_mag)
        enh_spec = apply_crm(noisy_spec, mask_real, mask_imag)

        enh_audio = istft(enh_spec, length=original_len)
        return enh_audio.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def enhance_long(
        self,
        audio: np.ndarray,
        sr: int = SAMPLE_RATE,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 0.5,
    ) -> np.ndarray:
        """
        Chunked inference with cross-fade overlap-add, for long recordings
        and low-memory environments (e.g. Streamlit Cloud free tier).
        """
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected sample rate {SAMPLE_RATE}, got {sr}.")

        chunk_len = int(chunk_seconds * sr)
        overlap_len = int(overlap_seconds * sr)
        hop = chunk_len - overlap_len

        if len(audio) <= chunk_len:
            return self.enhance(audio, sr=sr)

        output = np.zeros(len(audio), dtype=np.float32)
        weight = np.zeros(len(audio), dtype=np.float32)

        # linear cross-fade window for smooth blending at chunk boundaries
        fade = np.ones(chunk_len, dtype=np.float32)
        if overlap_len > 0:
            ramp = np.linspace(0, 1, overlap_len, dtype=np.float32)
            fade[:overlap_len] = ramp
            fade[-overlap_len:] = ramp[::-1]

        for start in range(0, len(audio), hop):
            end = min(start + chunk_len, len(audio))
            chunk = audio[start:end]
            pad_len = chunk_len - len(chunk)
            if pad_len > 0:
                chunk = np.pad(chunk, (0, pad_len))

            enhanced_chunk = self.enhance(chunk, sr=sr)
            w = fade if pad_len == 0 else fade[: len(chunk) - pad_len]
            enhanced_chunk = enhanced_chunk[: end - start]
            w = w[: end - start]

            output[start:end] += enhanced_chunk * w
            weight[start:end] += w

            if end >= len(audio):
                break

        weight[weight == 0] = 1.0
        return output / weight
