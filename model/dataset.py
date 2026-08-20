"""
Dataset loader for the VoiceBank+DEMAND speech enhancement corpus.

Expects a directory layout (after running the download script in the
training notebook):

    data/
      clean_trainset_28spk_wav/
      noisy_trainset_28spk_wav/
      clean_testset_wav/
      noisy_testset_wav/

Each clean/noisy file pair shares the same filename.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.audio import load_audio, SAMPLE_RATE


class VoiceBankDemandDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        segment_seconds: float = 4.0,
        sample_rate: int = SAMPLE_RATE,
        random_crop: bool = True,
    ):
        """
        Args:
            root_dir: path to the extracted VoiceBank+DEMAND dataset
            split: 'train' or 'test'
            segment_seconds: fixed-length audio segment for batching
            random_crop: if True, randomly crop segments (training);
                         if False, crop from the start (deterministic eval)
        """
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate)
        self.random_crop = random_crop

        if split == "train":
            clean_dir = os.path.join(root_dir, "clean_trainset_28spk_wav")
            noisy_dir = os.path.join(root_dir, "noisy_trainset_28spk_wav")
        elif split == "test":
            clean_dir = os.path.join(root_dir, "clean_testset_wav")
            noisy_dir = os.path.join(root_dir, "noisy_testset_wav")
        else:
            raise ValueError(f"Unknown split: {split}")

        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        if len(self.clean_files) == 0:
            raise FileNotFoundError(
                f"No clean wav files found in {clean_dir}. "
                f"Did you run the dataset download cell in the training notebook?"
            )
        self.noisy_dir = noisy_dir

        # Validate matching pairs exist
        self.pairs = []
        for clean_path in self.clean_files:
            fname = os.path.basename(clean_path)
            noisy_path = os.path.join(noisy_dir, fname)
            if os.path.exists(noisy_path):
                self.pairs.append((clean_path, noisy_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError("No matching clean/noisy pairs found.")

    def __len__(self):
        return len(self.pairs)

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) >= self.segment_len:
            if self.random_crop:
                start = random.randint(0, len(audio) - self.segment_len)
            else:
                start = 0
            audio = audio[start : start + self.segment_len]
        else:
            audio = np.pad(audio, (0, self.segment_len - len(audio)))
        return audio

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        clean = load_audio(clean_path, sr=self.sample_rate)
        noisy = load_audio(noisy_path, sr=self.sample_rate)

        # Ensure equal length before cropping (align to shorter one)
        min_len = min(len(clean), len(noisy))
        clean, noisy = clean[:min_len], noisy[:min_len]

        if self.random_crop and len(clean) >= self.segment_len:
            start = random.randint(0, len(clean) - self.segment_len)
            clean = clean[start : start + self.segment_len]
            noisy = noisy[start : start + self.segment_len]
        else:
            clean = self._pad(clean)
            noisy = self._pad(noisy)

        return {
            "clean": torch.from_numpy(clean).float(),
            "noisy": torch.from_numpy(noisy).float(),
        }

    def _pad(self, audio):
        if len(audio) < self.segment_len:
            audio = np.pad(audio, (0, self.segment_len - len(audio)))
        else:
            audio = audio[: self.segment_len]
        return audio


def collate_fn(batch):
    clean = torch.stack([b["clean"] for b in batch])
    noisy = torch.stack([b["noisy"] for b in batch])
    return {"clean": clean, "noisy": noisy}
