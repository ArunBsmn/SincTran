"""SincTran: Multi-scale sinc-filter CNN-Transformer for EEG classification."""
from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BandConfig = Dict[Tuple[float, float], int]


class SincTran(nn.Module):
    """
    Multi-scale sinc-filter CNN-Transformer for imagined-speech EEG classification.

    Args:
        n_chans          (int): EEG input channels.
        n_times          (int): Temporal samples per epoch.
        n_outputs        (int): Number of output classes.
        sfreq            (float): Sampling frequency in Hz.
        eeg_bands        (BandConfig, optional): ``{(low_hz, high_hz): kernel_size}`` per band.
                                                 All kernel sizes must be odd.
        n_filters        (int): Sinc filters per band. Default: 8.
        depth_multiplier (int): Channel expansion; d_model = n_filters × depth_multiplier. Default: 2.
        t_kern           (int): Temporal depthwise kernel width, must be odd. Default: 15.
        pool1            (int): Temporal stride after spatial conv. Default: 8.
        pool2            (int): Adaptive temporal pooling target. Default: 7.
        drop_cnn         (float): CNN dropout. Default: 0.3.
        num_heads        (int): Transformer attention heads. Default: 4.
        ff_ratio         (float): FFN width = d_model × ff_ratio. Default: 1.0.
        drop_trans       (float): Transformer dropout. Default: 0.3.
        num_layers        (int): Transformer encoder depth. Default: 2.
        use_cls          (bool): CLS token aggregation; else mean-pool. Default: True.
        trans_act        (str): FFN activation, ``"relu"`` or ``"gelu"``. Default: ``"relu"``.
        norm_first       (bool): Pre-norm transformer layers. Default: False.
        embedding_dim    (int): Hidden dim in MLP head. Default: 64.
    """

    DEFAULT_EEG_BANDS: BandConfig = {
        (0.5,  4): 129,
        (4,    8):  65,
        (8,   13):  33,
        (13,  30):  17,
        (30, 100):   9,
    }

    def __init__(
        self,
        n_chans:          int,
        n_times:          int,
        n_outputs:        int,
        sfreq:            float,
        eeg_bands:        Optional[BandConfig] = None,
        n_filters:        int   = 8,
        depth_multiplier: int   = 2,
        t_kern:           int   = 15,
        pool1:            int   = 8,
        pool2:            int   = 7,
        drop_cnn:         float = 0.3,
        num_heads:        int   = 4,
        ff_ratio:         float = 1.0,
        drop_trans:       float = 0.3,
        num_layers:       int   = 2,
        use_cls:          bool  = True,
        trans_act:        Literal["relu", "gelu"] = "relu",
        norm_first:       bool  = False,
        embedding_dim:    int   = 64,
    ) -> None:
        super().__init__()

        bands   = eeg_bands or self.DEFAULT_EEG_BANDS
        d_model = n_filters * depth_multiplier
        seq_len = n_times // pool1 // pool2

        for k in [t_kern, *bands.values()]:
            if k % 2 == 0:
                raise ValueError(f"All kernel sizes must be odd, got {k}.")

        # Per-band sinc filter banks: (B, n_bands, n_filters, C, T)
        self.sinc_banks = nn.ModuleList([
            SincBandFilter(n_filters, sfreq, lo, hi, k)
            for (lo, hi), k in bands.items()
        ])

        # Band attention scores: (B, n_bands, 1)
        self.band_score = nn.Linear(n_filters, 1, bias=False)
        nn.init.uniform_(self.band_score.weight, -0.01, 0.01)

        # Spatial depthwise conv + temporal pool: (B, d_model, 1, T//pool1)
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters, d_model, (n_chans, 1), groups=n_filters, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(drop_cnn),
        )

        # Temporal depthwise conv + BN: (B, d_model, 1, T//pool1)
        self.temporal_dw = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, t_kern),
                      padding=(0, t_kern // 2), groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
        )

        # Pointwise conv + temporal pool: (B, d_model, 1, T//pool1//pool2)
        self.temporal_pw = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(drop_cnn),
        )

        self.use_cls = use_cls

        # Temporal positional embeddings: (1, seq_len, d_model)
        self.time_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        if use_cls:

            # CLS token: (1, 1, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

            # CLS positional embedding: (1, 1, d_model)
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder: (B, seq_len [+1], d_model) -> (B, seq_len [+1], d_model)
        self.encoder = nn.TransformerEncoder(
            TransparentEncoderLayer(
                d_model         = d_model,
                nhead           = num_heads,
                dim_feedforward = max(1, int(d_model * ff_ratio)),
                dropout         = drop_trans,
                activation      = trans_act,
                batch_first     = True,
                norm_first      = norm_first,
            ),
            num_layers           = num_layers,
            norm                 = nn.LayerNorm(d_model),
            enable_nested_tensor = not norm_first,
        )

        # Classification head: (B, d_model) -> (B, n_outputs)
        self.head = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, n_outputs),
        )

    def features(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, d_model) pre-classifier embedding."""
        bands = torch.stack([blk(x) for blk in self.sinc_banks], dim=1)
        band_repr = bands.mean(dim=(-1, -2))
        w = F.softplus(self.band_score(band_repr))
        attn = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        x = (bands * attn.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        x = self.spatial(x)
        x = self.temporal_dw(x)
        x = self.temporal_pw(x)
        x = x.squeeze(2).permute(0, 2, 1)
        B = x.shape[0]
        pos = self.time_pos
        if self.use_cls:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            pos = torch.cat([self.cls_pos, pos], dim=1)
        x = self.encoder(x + pos)
        return x[:, 0] if self.use_cls else x.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, n_outputs) logits."""
        return self.head(self.features(x))


class SincBandFilter(nn.Module):
    """
    Learnable sinc bandpass filter bank constrained to a physiological band.

    Args:
        n_filters   (int): Number of sinc filters.
        sfreq       (float): Sampling frequency in Hz.
        band_low    (float): Band lower bound in Hz.
        band_high   (float): Band upper bound in Hz.
        kernel_size (int): Filter length (must be odd).
    """

    def __init__(
        self,
        n_filters:   int,
        sfreq:       float,
        band_low:    float,
        band_high:   float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.band_low  = band_low
        self.band_high = band_high

        half = kernel_size // 2

        # Hamming window (causal half) for sinc kernel: (half, 1)
        self.register_buffer(
            "window",
            torch.hamming_window(kernel_size, periodic=False)[:half].unsqueeze(-1),
        )

        # Time indices scaled to radians: (half, 1)
        self.register_buffer(
            "n_pi",
            (torch.arange(-half, 0, dtype=torch.float32) / sfreq * 2 * math.pi).unsqueeze(-1),
        )

        stride = (band_high - band_low) / (n_filters + 1)

        # Evenly spaced initial lower cutoff frequencies: (1, n_filters)
        self.register_buffer(
            "low_freq_init",
            torch.linspace(band_low + stride, band_high - stride, n_filters).unsqueeze(0),
        )
        # Learnable lower-cutoff offsets: (1, n_filters)
        self.low_freq_offsets = nn.Parameter(torch.zeros(1, n_filters))

        # Learnable filter bandwidths: (1, n_filters)
        self.bandwidths = nn.Parameter(
            torch.full((1, n_filters), min(4.0, (band_high - band_low) / n_filters))
        )

        # DC placeholder for symmetric filter construction: (1, 1, 1, n_filters)
        self.register_buffer("ones", torch.ones(1, 1, 1, n_filters))

        # Per-filter normalisation + activation: (B, n_filters, C, T)
        self.bn  = nn.BatchNorm2d(n_filters)
        self.act = nn.ELU()

    def _build_filters(self) -> Tensor:
        """(1, K, 1, F) normalised sinc kernels."""
        low  = (self.low_freq_init + self.low_freq_offsets).clamp(self.band_low, self.band_high)
        high = (low + self.bandwidths.abs()).clamp(self.band_low, self.band_high)
        low  = torch.minimum(low, high - 0.1)
        bw   = high - low
        half = (torch.sin(self.n_pi * high) - torch.sin(self.n_pi * low))
        half = half / (self.n_pi / 2.0) * self.window / (2.0 * bw)
        half = half.unsqueeze(0).unsqueeze(2)
        filt = torch.cat([half, self.ones, torch.flip(half, [1])], dim=1)

        # Normalise each filter to unit std for stable training
        return filt / (filt.std(dim=1, keepdim=True) + 1e-7)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, n_filters, C, T)"""
        filt = self._build_filters().permute(3, 2, 0, 1)
        return self.act(self.bn(F.conv2d(x.unsqueeze(1), filt, padding="same")))


class TransparentEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer that caches per-head attention weights.

    Attributes:
        last_attn_weights (Tensor | None): Shape (B, n_heads, seq, seq).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Cached per-head attention weights: (B, n_heads, seq, seq)
        self.last_attn_weights: Optional[Tensor] = None

    def forward(
        self,
        src:                  Tensor,
        src_mask:             Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal:            bool = False,
        **kwargs,
    ) -> Tensor:
        """(B, seq, d_model) -> (B, seq, d_model); caches per-head attention weights."""
        x = src
        if self.norm_first:
            attn_out, w = self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                attn_mask            = src_mask,
                key_padding_mask     = src_key_padding_mask,
                need_weights         = True,
                average_attn_weights = False,
                is_causal            = is_causal,
            )
            x = x + self.dropout1(attn_out)
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out, w = self.self_attn(
                x, x, x,
                attn_mask            = src_mask,
                key_padding_mask     = src_key_padding_mask,
                need_weights         = True,
                average_attn_weights = False,
                is_causal            = is_causal,
            )
            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self._ff_block(x))
        self.last_attn_weights = w.detach()
        return x