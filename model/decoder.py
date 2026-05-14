# -*- coding: utf-8 -*-
"""
model/decoder.py

Query-guided change-mask decoder for VSCDNet.

The decoder is split into two parts:
1. ChangeFeatureHead builds a per-candidate change feature on the SAM token grid.
2. UpsampleMaskHead decodes the confidence-fused feature once per query frame and
   injects the query RGB image at high-resolution stages to recover fine details.

Expected feature resolution is 64 x 64 for SAM ViT-B with 1024 x 1024 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChangeDecoderConfig:
    """Configuration for the VSCD change decoder."""

    in_dim: int
    feat_dim: int = 128
    reduce_layers: int = 5
    patch_hw: Tuple[int, int] = (64, 64)
    fuse_ch: int = 64
    q_inj_ch: int = 16

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {self.in_dim}")
        if self.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {self.feat_dim}")
        if self.reduce_layers <= 0:
            raise ValueError(f"reduce_layers must be positive, got {self.reduce_layers}")
        if len(self.patch_hw) != 2 or self.patch_hw[0] <= 0 or self.patch_hw[1] <= 0:
            raise ValueError(f"patch_hw must be a positive (H, W) tuple, got {self.patch_hw}")
        if self.fuse_ch <= 0:
            raise ValueError(f"fuse_ch must be positive, got {self.fuse_ch}")
        if self.q_inj_ch <= 0:
            raise ValueError(f"q_inj_ch must be positive, got {self.q_inj_ch}")


def _make_reduce_stack(in_dim: int, feat_dim: int, layers: int) -> nn.Sequential:
    modules = []
    channels = in_dim
    for _ in range(layers - 1):
        modules.extend(
            [
                nn.Conv2d(channels, feat_dim, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        )
        channels = feat_dim
    modules.append(nn.Conv2d(channels, feat_dim, kernel_size=1, bias=True))
    return nn.Sequential(*modules)


def _check_feature_map(name: str, x: torch.Tensor, channels: int, hw: Tuple[int, int]) -> None:
    if x.dim() != 4:
        raise ValueError(f"{name} must be [B, C, H, W], got {tuple(x.shape)}")
    if x.size(1) != channels:
        raise ValueError(f"{name} channel mismatch: expected {channels}, got {x.size(1)}")
    if tuple(x.shape[-2:]) != tuple(hw):
        raise ValueError(f"{name} spatial size mismatch: expected {hw}, got {tuple(x.shape[-2:])}")


class ChangeFeatureHead(nn.Module):
    """
    Build a per-reference-candidate change feature on the patch-token grid.

    Inputs:
        Eq_map:   [B, D, H, W] query token map
        Er_w_map: [B, D, H, W] reference token map warped to the query grid

    Output:
        feat_ts:  [B, C, H, W] candidate-level change feature
    """

    def __init__(self, cfg: ChangeDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.reduce_q = _make_reduce_stack(cfg.in_dim, cfg.feat_dim, cfg.reduce_layers)
        self.reduce_r = _make_reduce_stack(cfg.in_dim, cfg.feat_dim, cfg.reduce_layers)

        in_channels = cfg.feat_dim * 3 + 1
        self.fuse64 = nn.Sequential(
            nn.Conv2d(in_channels, cfg.fuse_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.fuse_ch, cfg.fuse_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, Eq_map: torch.Tensor, Er_w_map: torch.Tensor) -> torch.Tensor:
        _check_feature_map("Eq_map", Eq_map, self.cfg.in_dim, self.cfg.patch_hw)
        _check_feature_map("Er_w_map", Er_w_map, self.cfg.in_dim, self.cfg.patch_hw)
        if Eq_map.shape[0] != Er_w_map.shape[0]:
            raise ValueError(
                f"Batch mismatch between Eq_map and Er_w_map: {Eq_map.shape[0]} vs {Er_w_map.shape[0]}"
            )

        Fq = self.reduce_q(Eq_map)
        Fr = self.reduce_r(Er_w_map)

        Fq_n = F.normalize(Fq, dim=1)
        Fr_n = F.normalize(Fr, dim=1)
        cosine = (Fq_n * Fr_n).sum(dim=1, keepdim=True)
        diff = Fr - Fq

        x = torch.cat([Fq, Fr, cosine, diff], dim=1)
        return self.fuse64(x)


class UpsampleMaskHead(nn.Module):
    """
    Decode one fused change feature per query frame to a full-resolution logit mask.

    Inputs:
        feat_fuse: [B, C, H, W]
        qry_rgb:   [B, 3, 1024, 1024]

    Output:
        logits:    [B, 1, 1024, 1024]
    """

    def __init__(self, cfg: ChangeDecoderConfig):
        super().__init__()
        self.cfg = cfg
        channels = cfg.fuse_ch
        q_channels = cfg.q_inj_ch

        self.qproj_256 = nn.Sequential(
            nn.Conv2d(3, q_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.qproj_1024 = nn.Sequential(
            nn.Conv2d(3, q_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up_64_128 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up_128_256 = nn.Sequential(
            nn.Conv2d(channels + q_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up_256_512 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up_512_1024 = nn.Sequential(
            nn.Conv2d(channels + q_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_logits = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, feat_fuse: torch.Tensor, qry_rgb: torch.Tensor) -> torch.Tensor:
        _check_feature_map("feat_fuse", feat_fuse, self.cfg.fuse_ch, self.cfg.patch_hw)
        if qry_rgb.dim() != 4 or qry_rgb.size(1) != 3:
            raise ValueError(f"qry_rgb must be [B, 3, H, W], got {tuple(qry_rgb.shape)}")
        if qry_rgb.shape[0] != feat_fuse.shape[0]:
            raise ValueError(
                f"Batch mismatch between feat_fuse and qry_rgb: {feat_fuse.shape[0]} vs {qry_rgb.shape[0]}"
            )
        if tuple(qry_rgb.shape[-2:]) != (1024, 1024):
            raise ValueError(f"qry_rgb must have spatial size (1024, 1024), got {tuple(qry_rgb.shape[-2:])}")

        feat128 = F.interpolate(feat_fuse, scale_factor=2, mode="bilinear", align_corners=False)
        feat128 = self.up_64_128(feat128)

        feat256 = F.interpolate(feat128, scale_factor=2, mode="bilinear", align_corners=False)
        qry_256 = F.interpolate(qry_rgb, size=(256, 256), mode="bilinear", align_corners=False)
        feat256 = torch.cat([feat256, self.qproj_256(qry_256)], dim=1)
        feat256 = self.up_128_256(feat256)

        feat512 = F.interpolate(feat256, scale_factor=2, mode="bilinear", align_corners=False)
        feat512 = self.up_256_512(feat512)

        feat1024 = F.interpolate(feat512, scale_factor=2, mode="bilinear", align_corners=False)
        feat1024 = torch.cat([feat1024, self.qproj_1024(qry_rgb)], dim=1)
        feat1024 = self.up_512_1024(feat1024)

        return self.out_logits(feat1024)


class ChangeMaskDecoder(nn.Module):
    """
    VSCD change decoder wrapper.

    make_feat() is applied to each query-reference candidate pair. After candidate
    features are fused by confidence-aware fusion, decode_1024() predicts the final
    query-frame change logit mask.
    """

    def __init__(self, cfg: ChangeDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.feat_head = ChangeFeatureHead(cfg)
        self.up_head = UpsampleMaskHead(cfg)

    def make_feat(self, Eq_map: torch.Tensor, Er_w_map: torch.Tensor) -> torch.Tensor:
        return self.feat_head(Eq_map, Er_w_map)

    def decode_1024(self, feat_fuse_64: torch.Tensor, qry_rgb: torch.Tensor) -> torch.Tensor:
        return self.up_head(feat_fuse_64, qry_rgb)
