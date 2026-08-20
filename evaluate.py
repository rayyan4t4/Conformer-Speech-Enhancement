import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.unet_conformer import ConformerUNet
from dataset import SpeechEnhancementDataset
from utils.metrics import compute_si_sdr, compute_pesq_score, compute_stoi_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Speech Enhancement Model on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--clean_test_dir", type=str, required=True, help="Path to clean test set")
    parser.add_argument("--noisy_test_dir", type=str, default=None, help="Path to noisy test set")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Running Evaluation on: {device}")

    # Load Model
    model = ConformerUNet(d_model=args.d_model, num_conformer_layers=args.num_layers).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = SpeechEnhancementDataset(clean_dir=args.clean_test_dir, noisy_dir=args.noisy_test_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    noisy_sisdr_list, enh_sisdr_list = [], []
    noisy_pesq_list, enh_pesq_list = [], []
    noisy_stoi_list, enh_stoi_list = [], []
    inference_times = []
    total_audio_duration = 0.0

    print(f"--> Evaluating {len(test_dataset)} test utterances...")

    for i, (noisy_audio, clean_audio) in enumerate(test_loader):
        noisy_audio = noisy_audio.to(device)
        clean_audio = clean_audio.to(device)
        audio_dur = noisy_audio.shape[-1] / 16000.0
        total_audio_duration += audio_dur

        # Measure Latency
        t0 = time.perf_counter()
        enh_audio, _ = model(noisy_audio)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        n_np = noisy_audio.squeeze(0).cpu().numpy()
        c_np = clean_audio.squeeze(0).cpu().numpy()
        e_np = enh_audio.squeeze(0).cpu().numpy()

        # SI-SDR
        noisy_sisdr_list.append(compute_si_sdr(n_np, c_np))
        enh_sisdr_list.append(compute_si_sdr(e_np, c_np))

        # PESQ
        p_n = compute_pesq_score(c_np, n_np, 16000)
        p_e = compute_pesq_score(c_np, e_np, 16000)
        if not np.isnan(p_n) and not np.isnan(p_e):
            noisy_pesq_list.append(p_n)
            enh_pesq_list.append(p_e)

        # STOI
        s_n = compute_stoi_score(c_np, n_np, 16000)
        s_e = compute_stoi_score(c_np, e_np, 16000)
        if not np.isnan(s_n) and not np.isnan(s_e):
            noisy_stoi_list.append(s_n)
            enh_stoi_list.append(s_e)

    total_proc_time = sum(inference_times)
    rtf = total_proc_time / total_audio_duration if total_audio_duration > 0 else 0.0

    print("\n" + "=" * 60)
    print("           BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Test Samples:          {len(test_dataset)}")
    print(f"  Real-Time Factor (RTF):       {rtf:.4f} (lower is faster, <1.0 is real-time)")
    print(f"  Average Latency per Second:  {(rtf * 1000):.2f} ms")
    print("-" * 60)
    print(f"  Metric       | Noisy Input | Enhanced (Conformer U-Net) | Improvement")
    print(f"  SI-SDR (dB)  | {np.mean(noisy_sisdr_list):>11.2f} | {np.mean(enh_sisdr_list):>26.2f} | {np.mean(enh_sisdr_list) - np.mean(noisy_sisdr_list):>+10.2f} dB")
    if noisy_pesq_list:
        print(f"  PESQ (WB)    | {np.mean(noisy_pesq_list):>11.2f} | {np.mean(enh_pesq_list):>26.2f} | {np.mean(enh_pesq_list) - np.mean(noisy_pesq_list):>+10.2f}")
    if noisy_stoi_list:
        print(f"  STOI         | {np.mean(noisy_stoi_list):>11.2f} | {np.mean(enh_stoi_list):>26.2f} | {np.mean(enh_stoi_list) - np.mean(noisy_stoi_list):>+10.2f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
