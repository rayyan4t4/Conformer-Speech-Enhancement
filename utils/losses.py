import torch
import torch.nn as nn
import torch.nn.functional as F

class CompressedSpectralLoss(nn.Module):
    """
    Power-Law Compressed Complex Spectral Loss for Speech Enhancement.
    
    Computes both magnitude and complex spectral error under power-law compression
    (gamma = 0.3), which correlates strongly with human auditory loudness perception (PESQ/STOI).
    """
    def __init__(self, gamma: float = 0.3, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        """
        pred_spec, target_spec: (B, 2, F, T) [Real, Imag]
        """
        pred_r, pred_i = pred_spec[:, 0], pred_spec[:, 1]
        targ_r, targ_i = target_spec[:, 0], target_spec[:, 1]

        # Magnitudes
        pred_mag = torch.clamp(pred_r**2 + pred_i**2, min=1e-12)**(self.gamma / 2.0)
        targ_mag = torch.clamp(targ_r**2 + targ_i**2, min=1e-12)**(self.gamma / 2.0)

        # Magnitude L1 Loss
        loss_mag = F.l1_loss(pred_mag, targ_mag)

        # Compressed Complex Spectrogram
        pred_c_r = pred_mag * (pred_r / torch.clamp(torch.sqrt(pred_r**2 + pred_i**2), min=1e-12))
        pred_c_i = pred_mag * (pred_i / torch.clamp(torch.sqrt(pred_r**2 + pred_i**2), min=1e-12))
        targ_c_r = targ_mag * (targ_r / torch.clamp(torch.sqrt(targ_r**2 + targ_i**2), min=1e-12))
        targ_c_i = targ_mag * (targ_i / torch.clamp(torch.sqrt(targ_r**2 + targ_i**2), min=1e-12))

        loss_complex = F.l1_loss(pred_c_r, targ_c_r) + F.l1_loss(pred_c_i, targ_c_i)

        return self.alpha * loss_mag + (1.0 - self.alpha) * loss_complex


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR) Loss.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        est: estimated waveform (B, L)
        ref: clean reference waveform (B, L)
        """
        # Zero-mean normalization
        est = est - torch.mean(est, dim=-1, keepdim=True)
        ref = ref - torch.mean(ref, dim=-1, keepdim=True)

        # Scale factor alpha = <est, ref> / <ref, ref>
        dot = torch.sum(est * ref, dim=-1, keepdim=True)
        ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / ref_energy) * ref
        e_noise = est - s_target

        s_target_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps

        si_snr = 10 * torch.log10(s_target_energy / e_noise_energy)
        return -torch.mean(si_snr)


class HybridSpeechLoss(nn.Module):
    """
    Composite Hybrid Loss balancing Spectral Domain Quality with Time-Domain Fidelity.
    L_total = L_spec + lambda_time * L_si_snr
    """
    def __init__(self, gamma: float = 0.3, alpha: float = 0.5, lambda_time: float = 0.05):
        super().__init__()
        self.spec_loss = CompressedSpectralLoss(gamma=gamma, alpha=alpha)
        self.time_loss = SISNRLoss()
        self.lambda_time = lambda_time

    def forward(self, pred_audio: torch.Tensor, targ_audio: torch.Tensor,
                pred_spec: torch.Tensor, targ_spec: torch.Tensor):
        l_spec = self.spec_loss(pred_spec, targ_spec)
        l_time = self.time_loss(pred_audio, targ_audio)
        l_total = l_spec + self.lambda_time * l_time
        return l_total, l_spec, l_time
