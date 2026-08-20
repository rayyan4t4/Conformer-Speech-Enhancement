import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish (SiLU) activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit along the channel dimension."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module (Macaron-style).
    FFN(x) = LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention with pre-LayerNorm and residual connection.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        normed = self.layer_norm(x)
        attn_out, _ = self.mha(normed, normed, normed, key_padding_mask=mask)
        return residual + self.dropout(attn_out)


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module:
    x -> LayerNorm -> Pointwise Conv (1x1, 2*d) -> GLU -> 1D Depthwise Conv -> 
         BatchNorm1d -> Swish -> Pointwise Conv (1x1, d) -> Dropout -> + residual
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for same-length padding"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Channels/d_model)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Single Conformer Block composed of:
    1. FFN (Macaron style, 0.5 scale)
    2. Multi-Head Self Attention
    3. Convolution Module
    4. FFN (0.5 scale)
    5. LayerNorm
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mha(x, mask=mask)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Stack of N Conformer Blocks."""
    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
