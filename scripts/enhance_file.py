#!/usr/bin/env python3
"""
CLI utility to enhance a single audio file with a trained Conformer checkpoint.

Usage:
    python scripts/enhance_file.py --input noisy.wav --output enhanced.wav \\
        --checkpoint checkpoints/conformer_se_best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.inference import SpeechEnhancer, SAMPLE_RATE
from utils.audio import load_audio, save_audio


def main():
    parser = argparse.ArgumentParser(description="Enhance a noisy speech file.")
    parser.add_argument("--input", required=True, help="Path to noisy input audio")
    parser.add_argument("--output", required=True, help="Path to write enhanced audio")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/conformer_se_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (auto-detected if omitted)")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train the model first using "
            f"the notebook in notebook/, or point --checkpoint at your .pt file."
        )

    print(f"Loading model from {args.checkpoint} ...")
    enhancer = SpeechEnhancer(checkpoint_path=args.checkpoint, device=args.device)

    print(f"Loading audio from {args.input} ...")
    audio = load_audio(args.input, sr=SAMPLE_RATE)

    print("Enhancing...")
    enhanced = enhancer.enhance_long(audio, sr=SAMPLE_RATE, chunk_seconds=args.chunk_seconds)

    save_audio(args.output, enhanced, sr=SAMPLE_RATE)
    print(f"Saved enhanced audio to {args.output}")


if __name__ == "__main__":
    main()
