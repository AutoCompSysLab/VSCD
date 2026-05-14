# -*- coding: utf-8 -*-
"""
Deterministic preprocessing utilities for VSCDSequenceDataset.

The dataset loader already resizes RGB frames and masks and normalizes RGB
frames with ImageNet statistics. This transform module is intentionally kept
minimal for the public release:

- validate the expected tensor shapes;
- optionally resize tensors as a safety check if their spatial size differs
  from the configured output resolution;
- avoid stochastic data augmentation so that training and evaluation are
  reproducible and match the paper setting.

Input sample format:
    {
        "ref_frames": Tensor [T_r, 3, H, W],
        "qry_frames": Tensor [T_q, 3, H, W],
        "qry_masks":  Tensor [T_q, 1, H, W],
        ...
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for deterministic VSCD preprocessing."""

    out_hw: Tuple[int, int] = (1024, 1024)  # (H, W)

    def __post_init__(self) -> None:
        if len(self.out_hw) != 2:
            raise ValueError(f"out_hw must be a tuple of length 2, got {self.out_hw}")
        if int(self.out_hw[0]) <= 0 or int(self.out_hw[1]) <= 0:
            raise ValueError(f"out_hw must contain positive integers, got {self.out_hw}")


def _check_video_tensor(x: torch.Tensor, name: str) -> None:
    """Validate a video tensor shaped [T, 3, H, W]."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() != 4:
        raise ValueError(f"{name} must be [T, 3, H, W], got {tuple(x.shape)}")
    if x.size(1) != 3:
        raise ValueError(f"{name} must have 3 channels, got {x.size(1)}")


def _check_mask_tensor(x: torch.Tensor, name: str) -> None:
    """Validate a binary mask tensor shaped [T, 1, H, W]."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() != 4:
        raise ValueError(f"{name} must be [T, 1, H, W], got {tuple(x.shape)}")
    if x.size(1) != 1:
        raise ValueError(f"{name} must have 1 channel, got {x.size(1)}")


def _resize_video_if_needed(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize a [T, C, H, W] video tensor with bilinear interpolation if needed."""
    if tuple(x.shape[-2:]) == tuple(out_hw):
        return x
    return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


def _resize_mask_if_needed(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize a [T, 1, H, W] mask tensor with nearest-neighbor interpolation if needed."""
    if tuple(x.shape[-2:]) == tuple(out_hw):
        return x
    return F.interpolate(x, size=out_hw, mode="nearest")


class VideoPairTransforms:
    """Apply deterministic safety preprocessing to a VSCD sample."""

    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "ref_frames" not in sample or "qry_frames" not in sample or "qry_masks" not in sample:
            raise KeyError("sample must contain ref_frames, qry_frames, and qry_masks")

        ref = sample["ref_frames"]
        qry = sample["qry_frames"]
        masks = sample["qry_masks"]

        _check_video_tensor(ref, "ref_frames")
        _check_video_tensor(qry, "qry_frames")
        _check_mask_tensor(masks, "qry_masks")

        out_hw = (int(self.cfg.out_hw[0]), int(self.cfg.out_hw[1]))
        sample["ref_frames"] = _resize_video_if_needed(ref, out_hw)
        sample["qry_frames"] = _resize_video_if_needed(qry, out_hw)
        sample["qry_masks"] = _resize_mask_if_needed(masks, out_hw)

        # Kept for compatibility with older evaluation utilities. The cleaned
        # train/eval path does not require full query sequences.
        if "qry_frames_full" in sample:
            qfull = sample["qry_frames_full"]
            _check_video_tensor(qfull, "qry_frames_full")
            sample["qry_frames_full"] = _resize_video_if_needed(qfull, out_hw)

        return sample


def build_transforms(is_train: bool, cfg: Any) -> VideoPairTransforms:
    """
    Build deterministic VSCD transforms.

    The ``is_train`` argument is accepted for API compatibility. This public
    preprocessing path does not apply random augmentation in either training or
    evaluation.
    """
    del is_train

    if hasattr(cfg, "img_h") and hasattr(cfg, "img_w"):
        out_hw = (int(cfg.img_h), int(cfg.img_w))
    elif hasattr(cfg, "img_size"):
        out_hw = (int(cfg.img_size), int(cfg.img_size))
    else:
        out_hw = (1024, 1024)

    return VideoPairTransforms(TransformConfig(out_hw=out_hw))
