import os
import sys
import argparse
import time

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet_conformer import ConformerUNet
from dataset import SpeechEnhancementDataset
from utils.losses import HybridSpeechLoss
from utils.metrics import compute_si_sdr

def parse_args():
    parser = argparse.ArgumentParser(description="Train Conformer U-Net Speech Enhancement Model")
    parser.add_argument("--clean_train_dir", type=str, required=True, help="Path to clean training audio files")
    parser.add_argument("--noisy_train_dir", type=str, default=None, help="Path to paired noisy training files (optional)")
    parser.add_argument("--clean_val_dir", type=str, required=True, help="Path to clean validation audio files")
    parser.add_argument("--noisy_val_dir", type=str, default=None, help="Path to paired noisy validation files (optional)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--d_model", type=int, default=256, help="Conformer bottleneck dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Conformer blocks")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, total_spec, total_time = 0.0, 0.0, 0.0
    start_time = time.time()

    for step, (noisy_audio, clean_audio) in enumerate(loader):
        noisy_audio = noisy_audio.to(device, non_blocking=True)
        clean_audio = clean_audio.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
            targ_spec = model.compute_stft(clean_audio)
            pred_audio, pred_spec = model(noisy_audio)
            loss, l_spec, l_time = criterion(pred_audio, clean_audio, pred_spec, targ_spec)

        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        total_spec += l_spec.item()
        total_time += l_time.item()

        if (step + 1) % 25 == 0 or (step + 1) == len(loader):
            print(f"  Step [{step+1}/{len(loader)}] | Loss: {loss.item():.4f} (Spectral: {l_spec.item():.4f}, Time SI-SNR: {l_time.item():.4f})")

    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start_time
    return avg_loss, elapsed


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    si_sdr_improvements = []

    for noisy_audio, clean_audio in loader:
        noisy_audio = noisy_audio.to(device)
        clean_audio = clean_audio.to(device)

        targ_spec = model.compute_stft(clean_audio)
        pred_audio, pred_spec = model(noisy_audio)
        loss, _, _ = criterion(pred_audio, clean_audio, pred_spec, targ_spec)
        total_loss += loss.item()

        pred_np = pred_audio.cpu().numpy()
        clean_np = clean_audio.cpu().numpy()
        noisy_np = noisy_audio.cpu().numpy()

        for p, c, n in zip(pred_np, clean_np, noisy_np):
            sdr_noisy = compute_si_sdr(n, c)
            sdr_pred = compute_si_sdr(p, c)
            si_sdr_improvements.append(sdr_pred - sdr_noisy)

    avg_loss = total_loss / len(loader)
    avg_si_sdr_gain = sum(si_sdr_improvements) / len(si_sdr_improvements) if si_sdr_improvements else 0.0
    return avg_loss, avg_si_sdr_gain


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")

    # Initialize Model & Loss
    model = ConformerUNet(d_model=args.d_model, num_conformer_layers=args.num_layers).to(device)
    criterion = HybridSpeechLoss(gamma=0.3, alpha=0.5, lambda_time=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

    # Data Loaders
    train_dataset = SpeechEnhancementDataset(clean_dir=args.clean_train_dir, noisy_dir=args.noisy_train_dir, is_train=True)
    val_dataset = SpeechEnhancementDataset(clean_dir=args.clean_val_dir, noisy_dir=args.noisy_val_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    print(f"--> Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    best_si_sdr_gain = -float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch [{epoch}/{args.epochs}] (LR: {scheduler.get_last_lr()[0]:.2e}) ===")
        train_loss, train_time = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.use_amp)
        val_loss, val_si_sdr_gain = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch} Summary: Train Loss = {train_loss:.4f} ({train_time:.1f}s) | Val Loss = {val_loss:.4f} | Val ΔSI-SDR = +{val_si_sdr_gain:.2f} dB")

        # Save Checkpoints
        latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_si_sdr_gain': val_si_sdr_gain
        }, latest_path)

        if val_si_sdr_gain > best_si_sdr_gain:
            best_si_sdr_gain = val_si_sdr_gain
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"⭐ New Best Model Saved with +{val_si_sdr_gain:.2f} dB SI-SDR Gain to {best_path}")

    print("\n--> Training Complete!")

if __name__ == "__main__":
    main()
