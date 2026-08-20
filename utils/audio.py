"""
Audio utility functions: STFT/ISTFT, feature extraction, masking, I/O.
"""

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file, resample to target sr, force mono."""
    audio, orig_sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr)


def get_window(win_length: int = WIN_LENGTH, device=None):
    return torch.hann_window(win_length, device=device)


def stft(
    audio: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
) -> torch.Tensor:
    """
    Args:
        audio: (B, samples)
    Returns:
        complex spectrogram (B, F, T) complex64
    """
    window = get_window(win_length, device=audio.device)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    return spec


def istft(
    spec: torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    length: int = None,
) -> torch.Tensor:
    """
    Args:
        spec: complex spectrogram (B, F, T)
    Returns:
        audio: (B, samples)
    """
    window = get_window(win_length, device=spec.device)
    audio = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )
    return audio


def complex_to_log_mag(spec: torch.Tensor) -> torch.Tensor:
    """(B, F, T) complex -> (B, T, F) log1p magnitude, transposed for sequence models."""
    mag = torch.abs(spec)
    log_mag = torch.log1p(mag)
    return log_mag.transpose(1, 2)  # (B, T, F)


def apply_crm(
    noisy_spec: torch.Tensor, mask_real: torch.Tensor, mask_imag: torch.Tensor
) -> torch.Tensor:
    """
    Apply a complex ratio mask to the noisy complex spectrogram.

    Args:
        noisy_spec: (B, F, T) complex
        mask_real, mask_imag: (B, T, F) real-valued mask components
    Returns:
        enhanced complex spectrogram (B, F, T)
    """
    mask_real = mask_real.transpose(1, 2)  # (B, F, T)
    mask_imag = mask_imag.transpose(1, 2)  # (B, F, T)

    noisy_real = noisy_spec.real
    noisy_imag = noisy_spec.imag

    enh_real = mask_real * noisy_real - mask_imag * noisy_imag
    enh_imag = mask_real * noisy_imag + mask_imag * noisy_real

    return torch.complex(enh_real, enh_imag)


def normalize_audio(audio: np.ndarray, target_db: float = -25.0) -> np.ndarray:
    """RMS normalize audio to a target dB level."""
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 10 ** (target_db / 20)
    if rms < 1e-9:
        return audio
    return audio * (target_rms / rms)


def chunk_audio(audio: np.ndarray, chunk_size: int, hop: int = None):
    """Split long audio into overlapping chunks for streaming/low-memory inference."""
    if hop is None:
        hop = chunk_size
    chunks = []
    for start in range(0, len(audio), hop):
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
        if end >= len(audio):
            break
    return chunks
