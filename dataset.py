import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.audio_processing import load_audio_mono, mix_noise_at_snr

class SpeechEnhancementDataset(Dataset):
    """
    High-Performance PyTorch Dataset for Speech Enhancement.
    
    Supports:
    1. Paired Clean/Noisy directories (e.g. VoiceBank-DEMAND).
    2. Dynamic on-the-fly noise mixing with environmental noise datasets (DNS / ESC-50 / Urbansound8K).
    3. Synthetic noise synthesis fallback for immediate testing without external noise files.
    """
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str = None,
        noise_dir: str = None,
        segment_seconds: float = 2.0,
        sample_rate: int = 16000,
        snr_range: tuple[float, float] = (-5.0, 15.0),
        is_train: bool = True
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.noise_dir = noise_dir
        self.segment_len = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.is_train = is_train

        # Collect clean speech files
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "**/*.wav"), recursive=True))
        if not self.clean_files:
            # Also check for flac / mp3
            self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "**/*.*"), recursive=True))

        # Collect noise files if provided
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "**/*.*"), recursive=True))

    def __len__(self) -> int:
        return len(self.clean_files)

    def _generate_synthetic_noise(self, length: int) -> np.ndarray:
        """Generates realistic synthetic stationary & non-stationary noise."""
        noise_type = random.choice(["white", "pink", "brown", "chirp", "multitone"])
        if noise_type == "white":
            noise = np.random.randn(length).astype(np.float32)
        elif noise_type == "pink":
            # 1/f noise filter approximation
            white = np.random.randn(length).astype(np.float32)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            import scipy.signal
            noise = scipy.signal.lfilter(b, a, white).astype(np.float32)
        elif noise_type == "brown":
            # Brownian / Red noise (integrated white noise)
            white = np.random.randn(length).astype(np.float32)
            noise = np.cumsum(white).astype(np.float32)
            noise = noise - np.mean(noise)
        else:
            # Low-frequency rumble + multi-tone hum (mains hum + fan noise)
            t = np.linspace(0, length / self.sample_rate, length, endpoint=False)
            freqs = [50.0, 60.0, 120.0, 300.0]
            noise = sum(np.sin(2 * np.pi * f * t) for f in freqs) + 0.5 * np.random.randn(length)

        # Normalize
        std = np.std(noise) + 1e-8
        return (noise / std).astype(np.float32)

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.segment_len:
            pad_len = self.segment_len - len(audio)
            return np.pad(audio, (0, pad_len), mode='constant')
        elif len(audio) > self.segment_len:
            if self.is_train:
                start = random.randint(0, len(audio) - self.segment_len)
            else:
                start = 0
            return audio[start:start + self.segment_len]
        return audio

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.clean_files[idx]
        clean_audio, _ = load_audio_mono(clean_path, self.sample_rate)

        # 1. Paired Noisy Mode
        if self.noisy_dir:
            file_name = os.path.basename(clean_path)
            noisy_path = os.path.join(self.noisy_dir, file_name)
            if os.path.exists(noisy_path):
                noisy_audio, _ = load_audio_mono(noisy_path, self.sample_rate)
            else:
                # Fallback to noise mixing
                noisy_audio = None
        else:
            noisy_audio = None

        # 2. Dynamic Noise Mixing Mode
        if noisy_audio is None:
            if self.noise_files:
                noise_path = random.choice(self.noise_files)
                noise_audio, _ = load_audio_mono(noise_path, self.sample_rate)
            else:
                noise_audio = self._generate_synthetic_noise(len(clean_audio))

            snr_db = random.uniform(*self.snr_range) if self.is_train else 5.0
            noisy_audio = mix_noise_at_snr(clean_audio, noise_audio, snr_db)

        # Apply segment cropping / padding
        if self.is_train:
            # Synchronous random crop
            if len(clean_audio) > self.segment_len:
                start = random.randint(0, len(clean_audio) - self.segment_len)
                clean_audio = clean_audio[start:start + self.segment_len]
                noisy_audio = noisy_audio[start:start + self.segment_len]
            else:
                clean_audio = self._crop_or_pad(clean_audio)
                noisy_audio = self._crop_or_pad(noisy_audio)
        else:
            clean_audio = self._crop_or_pad(clean_audio)
            noisy_audio = self._crop_or_pad(noisy_audio)

        return torch.from_numpy(noisy_audio.astype(np.float32)), torch.from_numpy(clean_audio.astype(np.float32))
