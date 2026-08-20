"""
Loss functions for Conformer speech enhancement training.

Combines:
  - SI-SDR (scale-invariant signal-to-distortion ratio) loss in time domain
  - L1 magnitude spectral loss in TF domain
  - L1 complex spectral loss (real + imaginary) for phase awareness

The final training loss is a weighted sum, which empirically gives more
stable convergence than any single term alone.
"""

import torch
import torch.nn as nn


def si_sdr_loss(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant SDR loss (negative SI-SDR, to be minimized).
    est, target: (B, samples)
    """
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    s_target = (
        torch.sum(est * target, dim=-1, keepdim=True)
        * target
        / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    )
    e_noise = est - s_target

    si_sdr = 10 * torch.log10(
        (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
    )
    return -si_sdr.mean()


def spectral_l1_loss(
    enh_spec: torch.Tensor, clean_spec: torch.Tensor
) -> torch.Tensor:
    """
    L1 loss on complex spectrograms (real + imaginary) plus magnitude.
    enh_spec, clean_spec: (B, F, T) complex
    """
    mag_loss = torch.abs(torch.abs(enh_spec) - torch.abs(clean_spec)).mean()
    real_loss = torch.abs(enh_spec.real - clean_spec.real).mean()
    imag_loss = torch.abs(enh_spec.imag - clean_spec.imag).mean()
    return mag_loss + real_loss + imag_loss


class ConformerSELoss(nn.Module):
    """Combined loss: weighted SI-SDR (time) + spectral L1 (TF)."""

    def __init__(self, sdr_weight: float = 1.0, spec_weight: float = 1.0):
        super().__init__()
        self.sdr_weight = sdr_weight
        self.spec_weight = spec_weight

    def forward(self, enh_audio, clean_audio, enh_spec=None, clean_spec=None):
        loss = self.sdr_weight * si_sdr_loss(enh_audio, clean_audio)
        components = {"si_sdr_loss": loss.item()}

        if enh_spec is not None and clean_spec is not None:
            spec_loss = self.spec_weight * spectral_l1_loss(enh_spec, clean_spec)
            components["spectral_loss"] = spec_loss.item()
            loss = loss + spec_loss

        components["total_loss"] = loss.item()
        return loss, components
