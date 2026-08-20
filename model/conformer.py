"""
Conformer-based Speech Enhancement Model
==========================================
Implements a Conformer encoder (Gulati et al., 2020) adapted for
time-frequency masking based single-channel speech enhancement.

The model consumes the noisy log-magnitude spectrogram and predicts a
bounded complex ratio mask (cRM) that is applied to the noisy complex
STFT to reconstruct an enhanced complex spectrogram.

Author: Conformer-SE project
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Basic building blocks
# --------------------------------------------------------------------------- #
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 6000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class FeedForwardModule(nn.Module):
    """Macaron-style half-step feed-forward module."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module with GLU + depthwise conv + BatchNorm."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)          # (B, D, T)
        x = self.pointwise_conv1(x)    # (B, 2D, T)
        x = self.glu(x)                # (B, D, T)
        x = self.depthwise_conv(x)     # (B, D, T)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)       # (B, T, D)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        residual = x
        x = self.layer_norm(x)
        out, _ = self.mha(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.dropout(out)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF -> MHSA -> Conv -> FF -> LayerNorm."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
#  Full model
# --------------------------------------------------------------------------- #
class ConformerSE(nn.Module):
    """
    Conformer-based Speech Enhancement network.

    Input:  noisy log-magnitude spectrogram, shape (B, T, F)
    Output: bounded complex ratio mask, two tensors (B, T, F) each
            (real part, imaginary part), values in (-K, K).

    Args:
        input_dim:   number of frequency bins (n_fft // 2 + 1)
        d_model:     Conformer hidden dimension
        num_layers:  number of stacked Conformer blocks
        num_heads:   attention heads
        conv_kernel: depthwise conv kernel size
        ff_expansion:feed-forward expansion factor
        dropout:     dropout probability
        mask_bound:  bound K for tanh-scaled complex ratio mask
    """

    def __init__(
        self,
        input_dim: int = 257,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel: int = 31,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        mask_bound: float = 3.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask_bound = mask_bound

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    conv_kernel=conv_kernel,
                    ff_expansion=ff_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head predicts real & imaginary mask components
        self.output_proj = nn.Linear(d_model, input_dim * 2)

    def forward(self, log_mag: torch.Tensor, key_padding_mask=None):
        """
        Args:
            log_mag: (B, T, F) log1p noisy magnitude spectrogram
        Returns:
            mask_real, mask_imag: (B, T, F) each, bounded in (-mask_bound, mask_bound)
        """
        x = self.input_proj(log_mag)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        out = self.output_proj(x)  # (B, T, 2F)
        mask_real, mask_imag = out.chunk(2, dim=-1)
        mask_real = self.mask_bound * torch.tanh(mask_real)
        mask_imag = self.mask_bound * torch.tanh(mask_imag)
        return mask_real, mask_imag

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> ConformerSE:
    """Factory function that builds a ConformerSE model from a config dict."""
    return ConformerSE(
        input_dim=config.get("n_fft", 512) // 2 + 1,
        d_model=config.get("d_model", 256),
        num_layers=config.get("num_layers", 8),
        num_heads=config.get("num_heads", 4),
        conv_kernel=config.get("conv_kernel", 31),
        ff_expansion=config.get("ff_expansion", 4),
        dropout=config.get("dropout", 0.1),
        mask_bound=config.get("mask_bound", 3.0),
    )


if __name__ == "__main__":
    cfg = dict(n_fft=512, d_model=256, num_layers=8, num_heads=4)
    model = build_model(cfg)
    print(f"Total parameters: {model.count_parameters():,}")
    dummy = torch.randn(2, 100, 257)
    mr, mi = model(dummy)
    print("mask_real:", mr.shape, "mask_imag:", mi.shape)
