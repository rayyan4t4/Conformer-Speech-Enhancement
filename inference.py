import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from models.unet_conformer import ConformerUNet
from utils.audio_processing import load_audio_mono, save_audio, generate_spectrogram_plot
from utils.metrics import compute_si_sdr

def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for Conformer Speech Enhancement")
    parser.add_argument("--input", type=str, required=True, help="Input noisy WAV file")
    parser.add_argument("--output", type=str, default="enhanced_output.wav", help="Output enhanced WAV file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If none, runs initialized model.")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save spectrogram comparison plot")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    return parser.parse_args()


@torch.no_grad()
def enhance_audio_file(model, input_path: str, output_path: str, device: str = "cpu", plot_path: str = None):
    noisy_audio, sr = load_audio_mono(input_path, target_sr=16000)
    audio_tensor = torch.from_numpy(noisy_audio).unsqueeze(0).to(device)
    enhanced_tensor, _ = model(audio_tensor)
    enhanced_audio = enhanced_tensor.squeeze(0).cpu().numpy()
    
    save_audio(output_path, enhanced_audio, sr=16000)
    print(f"✅ Enhanced audio saved to: {output_path}")

    if plot_path:
        fig = generate_spectrogram_plot(noisy_audio, enhanced_audio, sr=16000)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Spectrogram comparison saved to: {plot_path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    model = ConformerUNet(d_model=256, num_conformer_layers=4).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"--> Loaded checkpoint from {args.checkpoint}")
    else:
        print("⚠️ No checkpoint provided; running inference with initial weights.")

    model.eval()
    enhance_audio_file(model, args.input, args.output, device=device, plot_path=args.save_plot)

if __name__ == "__main__":
    main()
