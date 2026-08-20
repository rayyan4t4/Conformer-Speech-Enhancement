import torch
import torch.nn as nn
import torch.nn.functional as F
from .conformer import ConformerEncoder

class ConvBlock2d(nn.Module):
    """2D Convolution block with BatchNorm and PReLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TransConvBlock2d(nn.Module):
    """2D Transposed Convolution block with BatchNorm and PReLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0)):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.trans_conv(x)))


class ConformerUNet(nn.Module):
    """
    Conformer-Enhanced Complex Spectral U-Net for Speech Enhancement.
    
    Combines a multi-scale 2D Convolutional U-Net with Conformer attention-convolution 
    bottleneck blocks to perform Bounded Complex Ratio Masking (cRM) on complex STFT inputs.
    
    Args:
        n_fft: FFT window size (default: 512).
        hop_length: STFT hop size (default: 256).
        d_model: Bottleneck feature dimension (default: 256).
        num_conformer_layers: Number of Conformer blocks in the bottleneck (default: 4).
        n_heads: Number of attention heads in each Conformer block (default: 4).
        mask_bound: Maximum bound K for complex ratio mask (M = K * tanh(output)).
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        d_model: int = 256,
        num_conformer_layers: int = 4,
        n_heads: int = 4,
        mask_bound: float = 1.0
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mask_bound = mask_bound
        self.freq_bins = n_fft // 2 + 1  # 257 for n_fft=512

        # Register Hann window buffer for STFT / iSTFT
        self.register_buffer("window", torch.hann_window(n_fft))

        # 2D Conv Encoder: progressively compresses frequency dimension (256 -> 128 -> 64 -> 32 -> 16 -> 8)
        # Note: 257 bins will be sliced or padded to 256 (0:256) inside forward pass for clean power-of-2 downsampling
        self.enc1 = ConvBlock2d(2, 32, stride=(2, 1), padding=(1, 1))    # 256 -> 128
        self.enc2 = ConvBlock2d(32, 64, stride=(2, 1), padding=(1, 1))   # 128 -> 64
        self.enc3 = ConvBlock2d(64, 128, stride=(2, 1), padding=(1, 1))  # 64 -> 32
        self.enc4 = ConvBlock2d(128, 256, stride=(2, 1), padding=(1, 1)) # 32 -> 16
        self.enc5 = ConvBlock2d(256, 256, stride=(2, 1), padding=(1, 1)) # 16 -> 8

        # Bottleneck Projection
        # Shape after enc5: (B, 256, 8, T) -> (B, 2048, T)
        self.bottleneck_dim = 256 * 8
        self.proj_in = nn.Linear(self.bottleneck_dim, d_model)

        # Conformer Sequence Bottleneck
        self.conformer = ConformerEncoder(
            num_layers=num_conformer_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            conv_kernel_size=31,
            dropout=0.1
        )

        # Bottleneck Projection Out
        self.proj_out = nn.Linear(d_model, self.bottleneck_dim)

        # 2D Transposed Conv Decoder with Skip Connections (Concat)
        self.dec5 = TransConvBlock2d(256 + 256, 256, stride=(2, 1), padding=(1, 1), output_padding=(0, 0)) # 8 -> 16
        self.dec4 = TransConvBlock2d(256 + 256, 128, stride=(2, 1), padding=(1, 1), output_padding=(0, 0)) # 16 -> 32
        self.dec3 = TransConvBlock2d(128 + 128, 64, stride=(2, 1), padding=(1, 1), output_padding=(0, 0))  # 32 -> 64
        self.dec2 = TransConvBlock2d(64 + 64, 32, stride=(2, 1), padding=(1, 1), output_padding=(0, 0))    # 64 -> 128
        self.dec1 = nn.ConvTranspose2d(32 + 32, 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0)) # 128 -> 256

    def compute_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Computes Complex STFT for time-domain audio.
        Audio: (B, L) -> Returns (B, 2, F, T) [Real, Imag]
        """
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True
        )
        real = stft.real.unsqueeze(1)  # (B, 1, F, T)
        imag = stft.imag.unsqueeze(1)  # (B, 1, F, T)
        return torch.cat([real, imag], dim=1)

    def compute_istft(self, complex_spec: torch.Tensor, length: int = None) -> torch.Tensor:
        """
        Reconstructs time-domain audio from Complex Spectrogram.
        complex_spec: (B, 2, F, T) -> Returns audio: (B, L)
        """
        real = complex_spec[:, 0, :, :]
        imag = complex_spec[:, 1, :, :]
        c_spec = torch.complex(real, imag)
        audio = torch.istft(
            c_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=length
        )
        return audio

    def forward_spec(self, spec_in: torch.Tensor) -> torch.Tensor:
        """
        Estimates enhanced complex spectrogram from noisy complex spectrogram.
        spec_in: (B, 2, 257, T)
        """
        B, C, F_orig, T = spec_in.shape
        # Slice the 257th Nyquist bin to get 256 for symmetric U-Net downsampling
        x = spec_in[:, :, :256, :]
        nyquist = spec_in[:, :, 256:, :]  # (B, 2, 1, T)

        # Encoder Path with Skip Connections
        e1 = self.enc1(x)       # (B, 32, 128, T)
        e2 = self.enc2(e1)      # (B, 64, 64, T)
        e3 = self.enc3(e2)      # (B, 128, 32, T)
        e4 = self.enc4(e3)      # (B, 256, 16, T)
        e5 = self.enc5(e4)      # (B, 256, 8, T)

        # Bottleneck Sequence Processing
        # (B, 256, 8, T) -> (B, 2048, T) -> (B, T, 2048)
        feat = e5.permute(0, 3, 1, 2).contiguous().view(B, T, self.bottleneck_dim)
        feat_proj = self.proj_in(feat)
        feat_conf = self.conformer(feat_proj)
        feat_out = self.proj_out(feat_conf)
        feat_dec = feat_out.view(B, T, 256, 8).permute(0, 2, 3, 1).contiguous()  # (B, 256, 8, T)

        # Decoder Path with Skip Connections
        d5 = self.dec5(torch.cat([feat_dec, e5], dim=1))  # (B, 256, 16, T)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))        # (B, 128, 32, T)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))        # (B, 64, 64, T)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))        # (B, 32, 128, T)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))        # (B, 2, 256, T)

        # Pad Nyquist bin back to 257
        mask_256 = self.mask_bound * torch.tanh(d1)
        nyquist_mask = torch.zeros(B, 2, 1, T, device=spec_in.device, dtype=spec_in.dtype)
        mask = torch.cat([mask_256, nyquist_mask], dim=2)  # (B, 2, 257, T)

        # Complex Multiplication (Bounded Complex Ratio Masking):
        # S_hat_real = M_r * X_r - M_i * X_i
        # S_hat_imag = M_r * X_i + M_i * X_r
        Mr = mask[:, 0:1, :, :]
        Mi = mask[:, 1:2, :, :]
        Xr = spec_in[:, 0:1, :, :]
        Xi = spec_in[:, 1:2, :, :]

        Sr_hat = Mr * Xr - Mi * Xi
        Si_hat = Mr * Xi + Mi * Xr
        spec_out = torch.cat([Sr_hat, Si_hat], dim=1)

        return spec_out, mask

    def forward(self, audio_noisy: torch.Tensor) -> torch.Tensor:
        """
        End-to-End Forward Pass:
        audio_noisy: (B, L) -> Returns audio_enhanced: (B, L), spec_enhanced: (B, 2, F, T)
        """
        orig_len = audio_noisy.shape[-1]
        spec_noisy = self.compute_stft(audio_noisy)
        spec_enhanced, mask = self.forward_spec(spec_noisy)
        audio_enhanced = self.compute_istft(spec_enhanced, length=orig_len)
        return audio_enhanced, spec_enhanced
