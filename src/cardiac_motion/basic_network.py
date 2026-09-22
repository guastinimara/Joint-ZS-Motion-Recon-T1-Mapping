"""Core U-Net building blocks with attention-gated skip connections.

Reference:
    Attention U-Net (Oktay et al., 2018): https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionBlock(nn.Module):
    """Gated attention on skip features conditioned by decoder features."""

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: channels of the gating (decoder) features.
            F_l: channels of the skip (encoder) features.
            F_int: channels of the intermediate layer.
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, g, x, return_attention=False):
        gated_features = self.W_g(g)
        skip_features = self.W_x(x)
        fused_features = self.relu(gated_features + skip_features)
        attention = self.psi(fused_features) * self.scale

        if return_attention:
            return x * attention, attention
        return x * attention


class ConvBlockBN(nn.Module):
    """Stack of convolutions with BatchNorm and LeakyReLU."""

    def __init__(self, dim, n_ch_in, n_ch_out, n_convs, kernel_size=3, bias=False, padding_mode="zeros", dropout_prob=0.):
        super().__init__()

        if dim == 1:
            conv_op = nn.Conv1d
            norm_op = nn.BatchNorm1d
        elif dim == 2:
            conv_op = nn.Conv2d
            norm_op = nn.BatchNorm2d
        elif dim == 3:
            conv_op = nn.Conv3d
            norm_op = nn.BatchNorm3d

        padding = int(np.floor(kernel_size / 2))

        conv_block_list = []
        for i in range(n_convs):
            conv_block_list.append(conv_op(n_ch_in if i == 0 else n_ch_out, n_ch_out, kernel_size, padding=padding, bias=bias, padding_mode=padding_mode))
            conv_block_list.append(norm_op(n_ch_out))  # Add BatchNorm
            conv_block_list.append(nn.LeakyReLU())

        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, x):
        return self.conv_block(x)

class EncoderBN(nn.Module):
    """Encoder path that returns multiscale features for skip connections."""

    def __init__(self, dim, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size=3, bias=False, padding_mode="zeros", dropout_prob=0.5):
        super().__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters) * 2 ** ne)

        self.enc_blocks = nn.ModuleList(
            [ConvBlockBN(dim, n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode, dropout_prob=dropout_prob) for i in range(len(n_ch_list) - 1)]
        )

        if dim == 1:
            pool_op = nn.MaxPool1d(2)
        elif dim == 2:
            pool_op = nn.MaxPool2d(2)
        elif dim == 3:
            pool_op = nn.MaxPool3d(2)

        self.pool = pool_op

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class DecoderBN_attention(nn.Module):
    """Decoder path with attention-modulated skip connection weighting."""

    def __init__(self, dim, n_ch_in, n_dec_stages, n_convs_per_stage, n_filters, kernel_size=3, bias=False, padding_mode="zeros", dropout_prob=0.5):
        super().__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in * (1 / 2) ** ne))

        if dim == 1:
            conv_op = nn.Conv1d
            interp_mode = "linear"
        elif dim == 2:
            conv_op = nn.Conv2d
            interp_mode = "bilinear"
        elif dim == 3:
            interp_mode = "trilinear"
            conv_op = nn.Conv3d

        self.interp_mode = interp_mode

        padding = int(np.floor(kernel_size / 2))
        self.upconvs = nn.ModuleList([conv_op(n_ch_list[i], n_ch_list[i + 1], kernel_size=kernel_size, padding=padding, bias=bias, padding_mode=padding_mode) for i in range(len(n_ch_list) - 1)])
        self.dec_blocks = nn.ModuleList(
            [ConvBlockBN(dim, n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode, dropout_prob=dropout_prob) for i in range(len(n_ch_list) - 1)]
        )

        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(n_ch_list[i + 1], n_ch_list[i + 1], n_ch_list[i + 1] // 2) for i in range(len(n_ch_list) - 1)]
        )

    def forward(self, x, encoder_features, log_attention=False):
        attention_maps = []

        for i in range(len(self.dec_blocks)):
            skip_features = encoder_features[i]
            skip_shape = skip_features.shape
            x = F.interpolate(x, skip_shape[2:], mode=self.interp_mode, align_corners=False)
            x = self.upconvs[i](x)

            if log_attention:
                skip_features, attention = self.attention_blocks[i](g=x, x=skip_features, return_attention=True)
                attention_maps.append(attention)
            else:
                skip_features = self.attention_blocks[i](g=x, x=skip_features)

            x = torch.cat([x, skip_features], dim=1)
            x = self.dec_blocks[i](x)

        if log_attention:
            return x, attention_maps
        return x


class UNetBN_attention(nn.Module):
    """U-Net with BatchNorm and optional residual output connection."""

    def __init__(self, dim, n_ch_in=2, n_ch_out=2, n_enc_stages=5, n_convs_per_stage=2, n_filters=16, kernel_size=3, res_connection=False, bias=True, padding_mode="zeros", dropout_prob=0.5):
        super().__init__()
        self.encoder = EncoderBN(dim, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode, dropout_prob=dropout_prob)
        self.decoder = DecoderBN_attention(
            dim, n_filters * (2 ** (n_enc_stages - 1)), n_enc_stages, n_convs_per_stage, n_filters * (n_enc_stages * 2), kernel_size=kernel_size, bias=bias, padding_mode=padding_mode, dropout_prob=dropout_prob,
        )

        if dim == 1:
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        self.c1x1 = conv_op(n_filters, n_ch_out, kernel_size=1, padding=0, bias=bias)
        if res_connection:
            if n_ch_in == n_ch_out:
                self.res_connection = lambda x: x
            else:
                self.res_connection = conv_op(n_ch_in, n_ch_out, 1)
        else:
            self.res_connection = False

    def forward(self, x, log_attention=False):
        enc_features = self.encoder(x)
        if log_attention:
            dec, attention_maps = self.decoder(enc_features[-1], enc_features[::-1][1:], log_attention=log_attention)
        else:
            dec = self.decoder(enc_features[-1], enc_features[::-1][1:], log_attention=log_attention)

        out = self.c1x1(dec)
        if self.res_connection:
            out = out + self.res_connection(x)

        if log_attention:
            return out, attention_maps
        return out


