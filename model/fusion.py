# -*- coding: utf-8 -*-
"""
model/fusion.py

Confidence utilities for VSCDNet.

The model fuses per-candidate change features using two confidence terms:
  - frame-level confidence Cf from P_frame
  - patch-level confidence Csp from the local patch-matching distribution

This file intentionally keeps only the confidence computation and a small
feature-fusion helper used by the paper implementation. Mask-level fusion from
older prototypes is kept as a deprecated compatibility wrapper so existing
imports do not break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class FusionConfig:
    """Configuration for confidence-aware fusion."""

    cp: float = 1.0
    c: float = 1.0
    eps: float = 1e-6


def patch_confidence(P_patch: torch.Tensor, cfg: FusionConfig) -> torch.Tensor:
    """
    Compute patch-level confidence from a local patch-matching distribution.

    Args:
        P_patch: Tensor of shape [B, K*K, H, W]. The values are expected to be
            probabilities over local offsets at each spatial location.
        cfg: Fusion configuration.

    Returns:
        Tensor of shape [B, 1, H, W]. Higher values indicate more reliable
        local correspondence.
    """
    if P_patch.dim() != 4:
        raise ValueError(f"P_patch must be [B, KK, H, W], got {tuple(P_patch.shape)}")

    _, kk, _, _ = P_patch.shape
    if kk <= 1:
        raise ValueError(f"P_patch must contain at least two offsets, got KK={kk}")

    p = P_patch.clamp_min(1e-9)
    peak = p.max(dim=1, keepdim=True).values
    entropy = -(p * p.log()).sum(dim=1, keepdim=True)
    entropy = entropy / (torch.log(torch.tensor(float(kk), device=P_patch.device, dtype=P_patch.dtype)) + 1e-9)

    return float(cfg.cp) * peak + float(cfg.c) * (1.0 - entropy)


def fuse_features(
    features: Sequence[torch.Tensor],
    frame_confidences: Sequence[torch.Tensor],
    patch_confidences: Sequence[torch.Tensor],
    cfg: Optional[FusionConfig] = None,
) -> torch.Tensor:
    """
    Fuse per-candidate feature maps with frame- and patch-level confidence.

    Args:
        features: Sequence of tensors, each [B, C, H, W].
        frame_confidences: Sequence of tensors broadcastable to [B, 1, H, W],
            typically [B, 1, 1, 1] values from P_frame.
        patch_confidences: Sequence of tensors, each [B, 1, H, W].
        cfg: Fusion configuration. If omitted, defaults are used.

    Returns:
        Fused feature tensor of shape [B, C, H, W].
    """
    cfg = cfg if cfg is not None else FusionConfig()

    if not (len(features) == len(frame_confidences) == len(patch_confidences)):
        raise ValueError(
            "features, frame_confidences, and patch_confidences must have the same length"
        )
    if len(features) == 0:
        raise ValueError("At least one candidate feature is required for fusion")

    numerator = None
    denominator = None

    for feat, cf, csp in zip(features, frame_confidences, patch_confidences):
        if feat.dim() != 4:
            raise ValueError(f"Each feature must be [B, C, H, W], got {tuple(feat.shape)}")
        weight = cf * csp
        weighted_feat = weight * feat

        if numerator is None:
            numerator = weighted_feat
            denominator = weight
        else:
            numerator = numerator + weighted_feat
            denominator = denominator + weight

    return numerator / (denominator + float(cfg.eps))


def fuse_masks(
    Ms_list: Sequence[torch.Tensor],
    Cf_list: Sequence[torch.Tensor],
    Csp_list: Sequence[torch.Tensor],
    cfg: FusionConfig,
) -> torch.Tensor:
    """
    Deprecated compatibility wrapper.

    Older scripts imported ``fuse_masks`` even though the current VSCDNet
    implementation performs feature-level fusion before decoding. The operation
    is mathematically identical to ``fuse_features`` for tensors shaped like mask
    logits, so we keep this wrapper to avoid breaking existing code.
    """
    return fuse_features(Ms_list, Cf_list, Csp_list, cfg)
